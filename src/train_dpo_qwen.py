import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import deepspeed
from datasets import load_dataset
import argparse
from tqdm import tqdm

# ================= DPO Loss 实现 =================
def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """原生 PyTorch 实现的 DPO Loss"""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    losses = -F.logsigmoid(beta * logits)
    rewards = beta * (pi_logratios - ref_logratios).detach()
    return losses.mean(), rewards.mean()

def get_batch_logps(logits, labels, average_log_prob=False):
    """计算 Log Probabilities"""
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # 忽略 padding (-100)
    token_logps = -loss_fct(shift_logits, shift_labels)
    token_logps = token_logps.view(labels.shape[0], -1)
    
    # Sum over sequence
    # 注意：这里假设 padding 的 loss 已经是 0（CrossEntropyLoss 默认行为）
    if average_log_prob:
        return token_logps.sum(-1) / (shift_labels != -100).sum(-1)
    else:
        return token_logps.sum(-1)

# ================= 数据集类 =================
class QwenChatMLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        self.data = load_dataset("json", data_files=data_path, split="train")
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 预先构建 ChatML 模板部分
        self.system = "<|im_start|>system\nYou are a helpful and efficient AI programming assistant.<|im_end|>\n"
        self.user_start = "<|im_start|>user\n"
        self.user_end = "<|im_end|>\n"
        self.assist_start = "<|im_start|>assistant\n"
        self.assist_end = "<|im_end|>\n"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 手动构建 Prompt (避免依赖 tokenizer.apply_chat_template 导致的不确定性)
        prompt_str = f"{self.system}{self.user_start}{item['prompt']}{self.user_end}{self.assist_start}"
        
        def tokenize_pair(p_str, answer_str):
            full_text = p_str + answer_str + self.assist_end
            
            # 这里的 padding=False，我们在 collate_fn 里做 padding
            enc = self.tokenizer(
                full_text, 
                max_length=self.max_length, 
                truncation=True, 
                add_special_tokens=False
            )
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']
            
            # 构建 Labels：Prompt 部分设为 -100
            prompt_enc = self.tokenizer(p_str, add_special_tokens=False)['input_ids']
            labels = list(input_ids)
            if len(prompt_enc) < len(labels):
                for i in range(len(prompt_enc)):
                    labels[i] = -100
            else:
                # 极端情况：Prompt 被截断
                return None
                
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }

        chosen = tokenize_pair(prompt_str, item['chosen'])
        rejected = tokenize_pair(prompt_str, item['rejected'])
        
        if chosen is None or rejected is None:
            return None
            
        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_labels": chosen["labels"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_labels": rejected["labels"],
        }

def collate_fn(batch):
    # 过滤 None
    batch = [x for x in batch if x is not None]
    if len(batch) == 0: return None
    
    pad_id = 151643 # Qwen pad_token_id (或通过 tokenizer 获取)
    
    out = {}
    for key in batch[0].keys():
        # labels 用 -100 填充，其他用 pad_id 填充
        padding_value = -100 if "labels" in key else pad_id
        # attention_mask 用 0 填充
        if "attention_mask" in key: padding_value = 0
            
        tensors = [x[key] for x in batch]
        out[key] = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)
    return out

# ================= 主程序 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--num_epochs", type=int, default=1)
    
    # ================= 🔧 关键修改：添加 DeepSpeed 必须的参数 =================
    # DeepSpeed 需要这些来计算 "auto" 的配置
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    # ======================================================================

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 1. 初始化 Dataset 和 DataLoader
    dataset = QwenChatMLDataset(args.data_path, tokenizer, args.max_length)
    
    # 使用 args 里的 batch size
    train_dataloader = DataLoader(
        dataset, 
        batch_size=args.per_device_train_batch_size, # 这里使用传入的参数
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=0
    )

    # 2. 加载模型 (Policy)
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    policy_model.gradient_checkpointing_enable()

    # 3. 加载参考模型 (Ref)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    ref_model.eval()

    # 4. 初始化 DeepSpeed
    # 只需要把 Policy Model 传给 DeepSpeed 进行优化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=policy_model,
        model_parameters=policy_model.parameters()
    )
    
    # Reference Model 也需要放到正确的设备上，但不需要 DeepSpeed 优化器
    # 简单做法：我们让 DeepSpeed 也管理它（作为 Inference Engine）或者手动放
    # 为了兼容 ZeRO-3，我们最好也用 deepspeed.init_inference 或者简单的 .to(device)
    # 注意：ZeRO-3 下 ref_model 显存占用是个问题。这里简化处理，假设显存足够或依靠 offload。
    ref_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=ref_model,
        optimizer=None # Ref model 不优化
    )

    # 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        model_engine.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}", disable=(args.local_rank != 0))
        
        for batch in progress_bar:
            if batch is None: continue
            
            # Move batch to device
            device = model_engine.device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # --- Forward Pass (Policy) ---
            chosen_logps = get_batch_logps(
                model_engine(input_ids=batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits,
                batch['chosen_labels']
            )
            rejected_logps = get_batch_logps(
                model_engine(input_ids=batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask']).logits,
                batch['rejected_labels']
            )
            
            # --- Forward Pass (Reference) ---
            with torch.no_grad():
                ref_chosen_logps = get_batch_logps(
                    ref_engine(input_ids=batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits,
                    batch['chosen_labels']
                )
                ref_rejected_logps = get_batch_logps(
                    ref_engine(input_ids=batch['rejected_input_ids'], attention_mask=batch['rejected_attention_mask']).logits,
                    batch['rejected_labels']
                )

            # --- Loss Calculation ---
            loss, reward = dpo_loss(
                chosen_logps, rejected_logps, 
                ref_chosen_logps, ref_rejected_logps, 
                beta=args.beta
            )
            
            # --- Backward ---
            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1
            if args.local_rank == 0 and global_step % 5 == 0:
                progress_bar.set_postfix(loss=loss.item(), reward=reward.item())
                
            if global_step % 100 == 0 and args.local_rank == 0:
                print(f"Step {global_step} | Loss: {loss.item():.4f} | Reward: {reward.item():.4f}")

        # 保存
        if args.local_rank == 0:
            print(f"Saving epoch {epoch}...")
            # 注意：ZeRO-3 保存需要特殊处理，这里简单示意，建议使用 model_engine.save_checkpoint
            # 为了简单，我们只保存 tokenizer
            tokenizer.save_pretrained(args.output_dir)

    # 最终保存
    model_engine.save_checkpoint(args.output_dir)

if __name__ == "__main__":
    main()