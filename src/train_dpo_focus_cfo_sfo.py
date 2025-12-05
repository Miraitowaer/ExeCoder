import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Union, Any
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig

# =============================================================================
# 1. 导入原生工具
# =============================================================================
try:
    from trl.trainer.utils import DPODataCollatorWithPadding
except ImportError:
    from trl.trainer import DPODataCollatorWithPadding

# =============================================================================
# 2. 组合式 Data Collator
# =============================================================================
class FocusedDPODataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # 复用 trl 原生逻辑
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.base_collator = DPODataCollatorWithPadding(
            pad_token_id=pad_token_id,
            label_pad_token_id=-100,
            is_encoder_decoder=False 
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 暂存自定义字段
        rejected_masks = [f.pop("extra_rejected_mask") for f in features]
        
        # 2. 调用 trl 处理标准字段
        batch = self.base_collator(features)
        
        # 3. 简单补齐 mask
        target_len = batch["rejected_input_ids"].shape[1]
        padding_side = self.tokenizer.padding_side
        
        padded_masks = []
        for mask in rejected_masks:
            mask = torch.tensor(mask, dtype=torch.float32)
            if len(mask) > target_len: mask = mask[-target_len:]
            pad_len = target_len - len(mask)
            if pad_len > 0:
                zeros = torch.zeros(pad_len, dtype=torch.float32)
                if padding_side == "left":
                    mask = torch.cat([zeros, mask])
                else:
                    mask = torch.cat([mask, zeros])
            padded_masks.append(mask)
            
        batch["extra_rejected_mask"] = torch.stack(padded_masks)
        return batch

# =============================================================================
# 3. 自定义 Trainer (修复返回值数量)
# =============================================================================
class FocusedDPOTrainer(DPOTrainer):
    
    # 自定义 Logps 计算逻辑 (支持 Mask 加权)
    def _get_batch_logps_masked(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        is_rejected: bool = False, 
        focused_mask: torch.FloatTensor = None, 
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")

        labels = labels.clone()
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # --- 核心创新点：应用加权掩码 ---
        if is_rejected and focused_mask is not None:
            if focused_mask.device != per_token_logps.device:
                focused_mask = focused_mask.to(per_token_logps.device)
            
            per_token_logps = per_token_logps * focused_mask

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), loss_mask
        else:
            return (per_token_logps * loss_mask).sum(-1), loss_mask

    def concatenated_forward(
        self, model, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        # 1. 拼接输入
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        
        len_chosen = batch["chosen_labels"].shape[0]

        # 2. 前向传播
        all_logits = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch.get("concatenated_attention_mask", None),
            use_cache=False,
        ).logits

        # 3. 计算原始 Logps (Chosen部分直接用原生的)
        all_logps_results = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        
        if isinstance(all_logps_results, tuple):
            all_logps = all_logps_results[0]
        else:
            all_logps = all_logps_results

        chosen_logps = all_logps[:len_chosen]
        
        # 4. 核心创新点：重算 Rejected Logps (带 Mask)
        rejected_logits = all_logits[len_chosen:]
        rejected_labels = concatenated_batch["concatenated_labels"][len_chosen:]
        
        extra_mask = batch["extra_rejected_mask"].to(self.accelerator.device)
        
        # 动态对齐形状 (防止 Trainer 内部 padding 导致形状不一致)
        # 这是一个常见的 Edge Case，必须处理
        if extra_mask.shape[1] != rejected_logits.shape[1]:
            diff = rejected_logits.shape[1] - extra_mask.shape[1]
            if diff > 0: # Mask 短了，补 0
                zeros = torch.zeros((extra_mask.shape[0], diff), device=extra_mask.device)
                if self.tokenizer.padding_side == "left":
                    extra_mask = torch.cat([zeros, extra_mask], dim=1)
                else:
                    extra_mask = torch.cat([extra_mask, zeros], dim=1)
            elif diff < 0: # Mask 长了，截断
                if self.tokenizer.padding_side == "left":
                    extra_mask = extra_mask[:, -rejected_logits.shape[1]:]
                else:
                    extra_mask = extra_mask[:, :rejected_logits.shape[1]]

        # 调用自定义计算
        focused_rejected_logps, _ = self._get_batch_logps_masked(
            rejected_logits,
            rejected_labels,
            average_log_prob=(self.loss_type == "ipo"), 
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            is_rejected=True,
            focused_mask=extra_mask 
        )

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        
        # ✅ 关键修复：构建第 5 个返回值 (all_logps)
        # 将 Chosen 和 Focused-Rejected 拼接起来
        final_all_logps = torch.cat([chosen_logps, focused_rejected_logps], dim=0)

        return (chosen_logps, focused_rejected_logps, chosen_logits, rejected_logits, final_all_logps)

# =============================================================================
# 4. 数据预处理
# =============================================================================
def preprocess_data(examples, tokenizer, max_length=2048):
    new_examples = {"prompt": [], "chosen": [], "rejected": [], "extra_rejected_mask": []}
    
    for prompt, chosen, rejected, error_lines in zip(examples['prompt'], examples['chosen'], examples['rejected'], examples['error_lines']):
        full_rejected = prompt + rejected
        tokenized = tokenizer(full_rejected, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=max_length)
        offsets = tokenized['offset_mapping']
        
        prompt_len = len(tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)['input_ids'])
        
        # Adaptive Masking: Global Error -> 1.0, Local Error -> 0.1
        default_weight = 1.0 if not error_lines else 0.1
        mask = [default_weight] * len(tokenized['input_ids'])
        
        if error_lines:
            rej_lines = rejected.split('\n')
            line_ranges = []
            curr = 0
            for line in rej_lines:
                line_ranges.append((curr, curr + len(line)))
                curr += len(line) + 1
            
            prompt_char_len = len(prompt)
            target_lines = set(error_lines)
            
            for i, (start, end) in enumerate(offsets):
                if i < prompt_len: 
                    mask[i] = 0.0
                    continue
                rel_start, rel_end = start - prompt_char_len, end - prompt_char_len
                if rel_start < 0: 
                    mask[i] = 0.0
                    continue
                mid = (rel_start + rel_end) / 2
                for line_idx, (l_s, l_e) in enumerate(line_ranges):
                    if l_s <= mid < l_e:
                        if line_idx in target_lines:
                            mask[i] = 1.0 
                        break
        
        if len(mask) > 0:
            new_examples['prompt'].append(prompt)
            new_examples['chosen'].append(chosen)
            new_examples['rejected'].append(rejected)
            new_examples['extra_rejected_mask'].append(mask)
            
    return new_examples

# =============================================================================
# Main
# =============================================================================
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "SFT Model Path"})
    data_path: str = field(metadata={"help": "Data Path"})

def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.remove_unused_columns = False 

    model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, use_cache=False)
    ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path, trust_remote_code=True, use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset('json', data_files=script_args.data_path, split='train')
    dataset = dataset.map(preprocess_data, fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.max_length}, batched=True, batch_size=1000, remove_columns=dataset.column_names)

    collator = FocusedDPODataCollator(tokenizer)

    trainer = FocusedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()