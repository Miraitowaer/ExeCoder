import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Union, Any

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin
from trl import DPOTrainer, DPOConfig

# =============================================================================
# 1. 自定义 Data Collator (处理新增的 Mask 字段)
# =============================================================================
@dataclass
class FocusedDPODataCollatorWithPadding(DataCollatorMixin):
    tokenizer: AutoTokenizer
    # 我们需要特殊处理 extra_rejected_mask 的 padding
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. 提取自定义 mask，防止被默认 collator 丢弃或报错
        rejected_masks = [f.pop("extra_rejected_mask") for f in features]
        
        # 2. 调用默认的 padding 逻辑处理 input_ids 等
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        
        # 3. 手动对 extra_rejected_mask 进行 padding
        # 它的长度应该和 rejected_input_ids 一致
        max_len = batch["rejected_input_ids"].shape[1]
        padded_masks = []
        for mask in rejected_masks:
            # 截断
            mask = mask[:max_len]
            # 填充 (用 0.0 填充，表示 padding 部分不计算权重)
            padded_mask = mask + [0.0] * (max_len - len(mask))
            padded_masks.append(padded_mask)
            
        batch["extra_rejected_mask"] = torch.tensor(padded_masks, dtype=torch.float32)
        return batch

# =============================================================================
# 2. 自定义 Trainer (核心创新点: 重写 Logps 计算)
# =============================================================================
class FocusedDPOTrainer(DPOTrainer):
    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        is_rejected: bool = False, # 我们修改源码逻辑，增加这个标记判断
        focused_mask: torch.FloatTensor = None, # 接收我们的自定义 mask
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        
        # 调用父类逻辑计算标准的 token-level logps
        # 注意：为了不破坏父类签名，通常我们需要 trick 一下或者重写 forward
        # 但 trl 的结构比较紧耦合。这里我们采用更底层的重写方式。
        
        # 标准 Logits 处理
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits and labels must have the same shape.")

        labels = labels.clone()
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        # ==========================================
        # 🔥 INNOVATION: Focused Weighting Applied Here
        # ==========================================
        if is_rejected and focused_mask is not None:
            # focused_mask: [batch, seq_len]
            # 1.0 = Error Token (Full Penalty)
            # 0.1 = Correct Token (Reduced Penalty)
            
            # 确保 mask 和 logps 在同一设备
            focused_mask = focused_mask.to(per_token_logps.device)
            
            # 应用加权：
            # 我们希望 Error Token 的 logp 贡献保持原样 (weight 1.0)
            # 非 Error Token 的 logp 贡献变小 (weight 0.1) -> 对 loss 贡献变小 -> 梯度变小
            # 也就是让模型 "主要去优化 Error Token 的概率"
            per_token_logps = per_token_logps * focused_mask

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1), loss_mask
        else:
            return (per_token_logps * loss_mask).sum(-1), loss_mask

    # 重写 concatenated_forward 以便传入 is_rejected 和 focused_mask
    def concatenated_forward(
        self, model, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        
        # 1. 构建各种 Input
        len_chosen = batch["chosen_labels"].shape[0]
        
        # 拼接 batch (Chosen + Rejected)
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        
        # 2. 模型前向传播
        all_logits = model(
            input_ids=concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch.get("concatenated_attention_mask", None),
            use_cache=False,
        ).logits

        # 3. 切分 Logits
        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        
        # ==========================================
        # 🔥 HACK: 这里我们需要重新计算 Rejected 的 Logps 
        # 因为父类方法 get_batch_logps 没法传 mask，我们在上面计算了一次标准的
        # 现在我们要手动重算一次 "加权版" 的 rejected_logps
        # ==========================================
        
        # 提取 Rejected 部分的 Logits 和 Labels
        rejected_logits = all_logits[len_chosen:]
        rejected_labels = concatenated_batch["concatenated_labels"][len_chosen:]
        
        # 提取我们的自定义 Mask
        extra_mask = batch["extra_rejected_mask"].to(self.accelerator.device)
        
        # 调用我们魔改的 _get_batch_logps
        focused_rejected_logps, _ = self._get_batch_logps(
            rejected_logits,
            rejected_labels,
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            is_rejected=True,       # <--- 开启 Focused 模式
            focused_mask=extra_mask # <--- 传入 Mask
        )

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        # 返回修改后的 rejected_logps
        return (chosen_logps, focused_rejected_logps, chosen_logits, rejected_logits)


# =============================================================================
# 3. 数据预处理 (行号 -> Token Mask 映射)
# =============================================================================
def preprocess_data(examples, tokenizer, max_length=2048):
    new_examples = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "extra_rejected_mask": []
    }
    
    for prompt, chosen, rejected, error_lines in zip(examples['prompt'], examples['chosen'], examples['rejected'], examples['error_lines']):
        
        # 1. Tokenize Rejected (带 Offset Mapping)
        # 我们需要手动拼接 Prompt + Rejected 才能得到完整的 input_ids
        # DPO Trainer 内部是 Prompt + Response，所以我们这里模拟一下
        
        # 这里的逻辑稍微复杂：DPOTrainer 会再次 Tokenize。
        # 为了不出错，我们必须预先 Tokenize 好，然后以 `input_ids` 形式传给 Trainer。
        # 但 trl 支持预处理好的 dataset。
        
        # 构造完整文本
        full_rejected_text = prompt + rejected
        
        tokenized_rej = tokenizer(
            full_rejected_text,
            return_offsets_mapping=True,
            add_special_tokens=False, # 后面统一加
            truncation=True,
            max_length=max_length
        )
        
        # 2. 计算 Prompt 的长度 (Token 数)
        tokenized_prompt = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length
        )
        prompt_len = len(tokenized_prompt['input_ids'])
        
        # 3. 生成 Mask
        # 默认权重 0.1 (Correct Code Protection)
        # 错误行权重 1.0 (Full Penalty)
        if not error_lines:
            mask = [1.0] * len(tokenized_rej['input_ids'])
        else:
            # 有具体错误行，执行 Focused 逻辑
            mask = [0.1] * len(tokenized_rej['input_ids'])
        
        offsets = tokenized_rej['offset_mapping']
        
        # 计算 Rejected 部分每一行的字符范围
        # 注意：error_lines 是相对于 `rejected` 字符串的行号
        # Prompt 部分我们不关心，我们只看 Rejected 部分的 Token
        
        # 先计算 Rejected 字符串各行的 offset
        rej_lines = rejected.split('\n')
        line_char_ranges = []
        curr = 0
        for line in rej_lines:
            line_char_ranges.append((curr, curr + len(line)))
            curr += len(line) + 1 # +1 for \n
            
        prompt_char_len = len(prompt)
        
        target_lines = set(error_lines)
        
        for i, (start, end) in enumerate(offsets):
            # 如果这个 token 属于 prompt，给 0 权重 (DPO 默认也不算 prompt loss，这里双保险)
            if i < prompt_len:
                mask[i] = 0.0
                continue
                
            # 这个 token 在 rejected 字符串中的相对位置
            rel_start = start - prompt_char_len
            rel_end = end - prompt_char_len
            
            if rel_start < 0: # 还在 prompt 里
                mask[i] = 0.0
                continue
                
            token_mid = (rel_start + rel_end) / 2
            
            # 判断属于哪一行
            for line_idx, (l_start, l_end) in enumerate(line_char_ranges):
                if l_start <= token_mid < l_end:
                    if line_idx in target_lines:
                        mask[i] = 1.0 # 命中错误行，全额惩罚
                    break
        
        new_examples['prompt'].append(prompt)
        new_examples['chosen'].append(chosen)
        new_examples['rejected'].append(rejected)
        new_examples['extra_rejected_mask'].append(mask)

    return new_examples

# =============================================================================
# 4. Main Execution
# =============================================================================
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "SFT Model Path"})
    data_path: str = field(metadata={"help": "Data Path (must contain error_lines)"})

def main():
    parser = HfArgumentParser((ScriptArguments, DPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 1. Load Models
    print(f"Loading model from {script_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True,
        use_cache=False
    )
    
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load & Process Data
    dataset = load_dataset('json', data_files=script_args.data_path, split='train')
    
    print("Preprocessing data for Focused DPO...")
    # 使用 map 进行预处理，生成 extra_rejected_mask
    dataset = dataset.map(
        preprocess_data,
        fn_kwargs={"tokenizer": tokenizer, "max_length": training_args.max_length},
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names # 移除旧列，换成新生成的
    )
    
    # 3. Trainer
    trainer = FocusedDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=FocusedDPODataCollatorWithPadding(tokenizer), # 使用自定义 collator
    )

    print("Starting Focused-DPO Training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()