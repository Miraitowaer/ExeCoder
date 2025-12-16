import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= 配置区 =================
# 1. 你的 SFT 基座模型路径 (用于初始化结构)
BASE_MODEL_PATH = "/data/private/ExeCoder/results/Deepseek-coder-6.7b-instruct-code/checkpoint-327"

# 2. 你的 DeepSpeed Checkpoint 路径
# 注意：指向 checkpoint-500 目录即可
CHECKPOINT_PATH = "/data/private/ExeCoder/results/dpo_online_mask_v1/checkpoint-500"

# 3. 最终输出路径
OUTPUT_PATH = "/data/private/ExeCoder/results/dpo_online_mask_v1/checkpoint-500-hf"
# =========================================

def main():
    print(f"🚀 开始转换 ZeRO-2 Checkpoint...")
    print(f"📂 Checkpoint 路径: {CHECKPOINT_PATH}")

    # 1. 寻找 Rank 0 的权重文件
    # DeepSpeed ZeRO-2 通常命名为 mp_rank_00_model_states.pt
    target_file = os.path.join(CHECKPOINT_PATH, "mp_rank_00_model_states.pt")
    
    if not os.path.exists(target_file):
        # 备选方案：有时候可能是 global_stepXXX/mp_rank_00...
        print(f"⚠️ 未直接找到 {target_file}，尝试搜索子目录...")
        found = False
        for root, dirs, files in os.walk(CHECKPOINT_PATH):
            if "mp_rank_00_model_states.pt" in files:
                target_file = os.path.join(root, "mp_rank_00_model_states.pt")
                found = True
                break
        if not found:
            raise FileNotFoundError(f"❌ 无法在 {CHECKPOINT_PATH} 中找到 mp_rank_00_model_states.pt 文件！请检查目录是否为空。")

    print(f"✅ 找到权重文件: {target_file}")

    # 2. 加载基座模型结构 (加载到 CPU 内存，避免爆显存)
    print("⏳ 正在初始化基座模型结构...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16, # 保持和训练时一致
        trust_remote_code=True,
        device_map="cpu" # 强制 CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 3. 加载 DeepSpeed 权重
    print(f"⏳ 正在加载 DeepSpeed 权重 (这可能需要几分钟)...")
    # map_location='cpu' 关键，防止占用 GPU
    state_dict = torch.load(target_file, map_location='cpu')

    # DeepSpeed 保存的 state_dict 通常包裹在 'module' 键下
    if "module" in state_dict:
        print("ℹ️ 检测到 'module' 前缀，正在剥离...")
        state_dict = state_dict["module"]
    
    # 有时候 key 会带有 'module.' 前缀 (DDP 遗留)，需要去除
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    # 4. 覆盖权重
    print("⏳ 正在将权重应用到模型...")
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"📄 权重加载报告: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    
    if len(missing) > 0:
        print(f"⚠️ 警告: 丢失了 {len(missing)} 个键 (可能是 LoRA 或非关键参数，如果数量很大请警惕)")
    
    # 5. 保存为 HF 格式
    print(f"💾 正在保存为 Safetensors 格式到: {OUTPUT_PATH}")
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True, max_shard_size="10GB")
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print("🎉 转换完成！现在可以使用该模型进行推理了。")

if __name__ == "__main__":
    main()