import torch
from transformers import AutoTokenizer

# 加载保存的文件
# output_path = "/work/hdd/beib/gwang3/bdiq/AsFT/check_v/v_data/ours_v_gemma_2-9b_instr_base_nonorm_new_top50.pt"
# output_path = "/work/hdd/beib/gwang3/bdiq/AsFT/check_v/v_data/ours_v_llama_8b_instr_base_nonorm_new_top50.pt"
# output_path = "/work/hdd/beib/gwang3/bdiq/AsFT/check_v/v_data/ours_v_llama3_1b_instr_base_nonorm_new_top50.pt"
output_path = "/work/hdd/beib/gwang3/bdiq/DiffuGuard/find_token_set/v_dir_dllm_refusal.pt"


save_dict = torch.load(output_path, map_location="cpu")

# 提取 v_dir 向量和元信息
v_dir = save_dict["v_dir"]
metadata = save_dict["metadata"]

# 打印元信息
print("=== Metadata ===")
for key, value in metadata.items():
    print(f"  {key}: {value}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(metadata["aligned_model_path"])

# 转为 float32 方便查看
v_float = v_dir.float()

print(f"\n=== v_dir Info ===")
print(f"Shape: {v_float.shape}")
print(f"Min:  {v_float.min().item():.6f}")
print(f"Max:  {v_float.max().item():.6f}")
print(f"Mean: {v_float.mean().item():.6f}")
print(f"Std:  {v_float.std().item():.6f}")
print(f"Norm: {v_float.norm().item():.6f}")

# 非零元素统计
nonzero_mask = v_float.abs() > 1e-9
nonzero_count = nonzero_mask.sum().item()
print(f"Non-zero elements: {nonzero_count} / {v_float.numel()}")

# 查看 top-k
k = 20

# Top-k 正值（aligned model 偏好的 token）
topk_vals, topk_ids = torch.topk(v_float, k=k)
print(f"\n{'='*60}")
print(f"Top {k} POSITIVE Values (Aligned model prefers these tokens)")
print(f"{'='*60}")
print(f"{'Rank':<6} {'Token ID':<12} {'Value':<12} {'Token'}")
print("-" * 60)
for i in range(k):
    token_id = topk_ids[i].item()
    token_text = tokenizer.decode([token_id])
    # 显示不可见字符的 repr
    token_repr = repr(token_text)
    print(f"{i+1:<6} {token_id:<12} {topk_vals[i].item():<12.6f} {token_repr}")

# Top-k 负值（base model 偏好的 token）
bottomk_vals, bottomk_ids = torch.topk(v_float, k=k, largest=False)
print(f"\n{'='*60}")
print(f"Top {k} NEGATIVE Values (Base model prefers these tokens)")
print(f"{'='*60}")
print(f"{'Rank':<6} {'Token ID':<12} {'Value':<12} {'Token'}")
print("-" * 60)
for i in range(k):
    token_id = bottomk_ids[i].item()
    token_text = tokenizer.decode([token_id])
    token_repr = repr(token_text)
    print(f"{i+1:<6} {token_id:<12} {bottomk_vals[i].item():<12.6f} {token_repr}")

# 额外：查看所有非零 token 的分布
print(f"\n{'='*60}")
print("All Non-zero Token Statistics")
print(f"{'='*60}")
nonzero_vals = v_float[nonzero_mask]
print(f"Count: {len(nonzero_vals)}")
print(f"Positive count: {(nonzero_vals > 0).sum().item()}")
print(f"Negative count: {(nonzero_vals < 0).sum().item()}")