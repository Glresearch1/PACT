import os

import fire
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from ft_datasets.pure_bad_dataset.pure_bad_dataset import InstructionDataset


@torch.no_grad()
def _build_logit_direction_T1_D1(
    model_ref,
    model_base,
    tokenizer_ref,
    tokenizer_base,
    dataset,
    device,
    K=128,
    max_resp_len=128,
    temperature=1.0,
    diff_mode="p_diff",
    topk=200,
    eps=1e-6,
    require_same_vocab=True,
    compute_dtype=torch.bfloat16,
):
    """Build the T1/D1 logit-space direction from teacher-forced responses."""
    assert diff_mode in ["p_diff", "pref_logratio"], f"Unknown diff_mode={diff_mode}"

    model_ref.eval().to(device)
    model_base.eval().to(device)

    V_ref = getattr(model_ref.config, "vocab_size", None)
    V_base = getattr(model_base.config, "vocab_size", None)
    if require_same_vocab:
        assert V_ref == V_base, f"Model vocab mismatch: ref={V_ref}, base={V_base}"
        assert len(tokenizer_ref) == len(tokenizer_base) == V_ref, (
            f"Tokenizer vocab mismatch: ref_tok={len(tokenizer_ref)}, "
            f"base_tok={len(tokenizer_base)}, model={V_ref}"
        )

    diff_global = torch.zeros(V_ref, device=device, dtype=torch.float32)
    p_ref_global = torch.zeros(V_ref, device=device, dtype=torch.float32)
    cnt = 0

    for i in range(min(K, len(dataset))):
        if i % 100 == 0:
            print(f"Processing sample {i}/{min(K, len(dataset))}")

        sample = dataset[i]
        prompt_messages = sample.get("prompt_messages", None)
        response_text = sample.get("response_text", "")

        if not prompt_messages or not isinstance(prompt_messages, list):
            continue
        if response_text is None:
            response_text = ""
        response_text = str(response_text).strip()
        if len(response_text) == 0:
            continue

        resp_ids_ref = tokenizer_ref(
            response_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]
        resp_ids_base = tokenizer_base(
            response_text,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"]

        R_ref = resp_ids_ref.size(1)
        R_base = resp_ids_base.size(1)
        R = min(R_ref, R_base, max_resp_len)
        if R <= 0:
            continue
        resp_ids_ref = resp_ids_ref[:, :R].to(device)
        resp_ids_base = resp_ids_base[:, :R].to(device)

        if require_same_vocab:
            if not torch.equal(resp_ids_ref.cpu(), resp_ids_base.cpu()):
                raise ValueError(
                    f"[T1] response tokenization mismatch at sample {i}. "
                    "Ref and base tokenizers likely not identical."
                )

        prompt_ids_ref = tokenizer_ref.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        P_ref = prompt_ids_ref.size(1)

        input_ids_ref = torch.cat([prompt_ids_ref, resp_ids_ref], dim=1)
        attn_ref = torch.ones_like(input_ids_ref)

        base_prompt = ""
        for m in prompt_messages:
            if m.get("role") == "system":
                continue
            if m.get("role") == "user":
                base_prompt += f"User: {m.get('content', '')}\n\n"
        base_prompt += "Assistant: "

        prompt_ids_base = tokenizer_base(
            base_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )["input_ids"].to(device)
        P_base = prompt_ids_base.size(1)

        input_ids_base = torch.cat([prompt_ids_base, resp_ids_base], dim=1)
        attn_base = torch.ones_like(input_ids_base)

        out_ref = model_ref(input_ids=input_ids_ref, attention_mask=attn_ref)
        out_base = model_base(input_ids=input_ids_base, attention_mask=attn_base)

        logits_ref_full = (out_ref.logits / temperature).float()
        logits_base_full = (out_base.logits / temperature).float()

        if P_ref < 1 or P_base < 1:
            continue

        idx_ref_start = P_ref - 1
        idx_ref_end = P_ref + R - 1
        idx_base_start = P_base - 1
        idx_base_end = P_base + R - 1

        logits_ref = logits_ref_full[:, idx_ref_start:idx_ref_end, :]
        logits_base = logits_base_full[:, idx_base_start:idx_base_end, :]

        if logits_ref.size(1) != R or logits_base.size(1) != R:
            continue

        diff_sum = torch.zeros(V_ref, device=device, dtype=torch.float32)
        p_ref_sum = torch.zeros(V_ref, device=device, dtype=torch.float32)

        for t in range(R):
            lr = logits_ref[:, t, :]
            lb = logits_base[:, t, :]

            p_ref = F.softmax(lr, dim=-1)
            p_base = F.softmax(lb, dim=-1)

            if diff_mode == "p_diff":
                diff_step = p_ref - p_base
            else:
                logp_ref = F.log_softmax(lr, dim=-1)
                logp_base = F.log_softmax(lb, dim=-1)
                diff_step = p_ref * (logp_base - logp_ref)

            diff_sum += diff_step.squeeze(0)
            p_ref_sum += p_ref.squeeze(0)

        diff_global += diff_sum / max(R, 1)
        p_ref_global += p_ref_sum / max(R, 1)
        cnt += 1

    if cnt == 0:
        raise ValueError("No valid samples to build direction (all missing response_text?).")

    diff_global = diff_global / cnt
    p_ref_global = p_ref_global / cnt

    if topk is not None and topk > 0 and topk < diff_global.size(0):
        topk_ids = torch.topk(p_ref_global, k=topk, dim=-1).indices
        mask = torch.zeros_like(diff_global)
        mask.scatter_(0, topk_ids, 1.0)
        diff_global = diff_global * mask

    nonzero_mask = diff_global.abs() > 1e-9
    if nonzero_mask.sum() > 0:
        nonzero_vals = diff_global[nonzero_mask]
        centered_vals = nonzero_vals - nonzero_vals.mean()
        standardized_vals = centered_vals / (centered_vals.std() + eps)
        diff_global[nonzero_mask] = standardized_vals

    v = diff_global / (diff_global.norm() + eps)
    v = v.to(compute_dtype)

    return v.detach()


def compute_and_save_v_dir(
    base_model_path: str = "meta-llama/Llama-2-7b-hf",
    aligned_model_path: str = "meta-llama/Llama-2-7b-chat-hf",
    dataset_path: str = "safe_direction.json",
    output_path: str = "safety tokens path",
    K: int = 1000,
    max_resp_len: int = 64,
    temperature: float = 1.0,
    diff_mode: str = "p_diff",
    topk: int = 200,
    device: str = "cuda:0",
    use_bfloat16: bool = True,
):
    """Compute v_dir offline and save it to disk."""
    print("=" * 60)
    print("Offline v_dir Computation")
    print("=" * 60)
    print(f"Base model: {base_model_path}")
    print(f"Aligned model: {aligned_model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")
    print(f"K={K}, max_resp_len={max_resp_len}, topk={topk}")
    print("=" * 60)

    dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    print("\nLoading tokenizers...")
    tokenizer_instruct = AutoTokenizer.from_pretrained(aligned_model_path, use_fast=True)
    if tokenizer_instruct.pad_token is None:
        tokenizer_instruct.pad_token = tokenizer_instruct.eos_token
    tokenizer_instruct.padding_side = "right"

    tokenizer_base = AutoTokenizer.from_pretrained(aligned_model_path, use_fast=True)
    if tokenizer_base.pad_token is None:
        tokenizer_base.pad_token = tokenizer_base.eos_token
    tokenizer_base.padding_side = "right"
    tokenizer_base.chat_template = (
        "{% for m in messages %}"
        "{% if m['role'] == 'user' %}"
        "{{ 'User: ' + m['content'] + '\\n\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ 'Assistant: ' }}"
        "{% endif %}"
    )

    print("\nLoading dataset...")
    dataset = InstructionDataset(
        train_dataset_path=dataset_path,
        mode="prompt",
        keep_system=True,
    )
    print(f"Dataset size: {len(dataset)}")

    print(f"\nLoading base model with dtype={dtype}...")
    model_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"Loading aligned model with dtype={dtype}...")
    model_ref = AutoModelForCausalLM.from_pretrained(
        aligned_model_path,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print("\nComputing v_dir...")
    v_dir = _build_logit_direction_T1_D1(
        model_ref=model_ref,
        model_base=model_base,
        tokenizer_ref=tokenizer_instruct,
        tokenizer_base=tokenizer_base,
        dataset=dataset,
        device=device,
        K=K,
        max_resp_len=max_resp_len,
        temperature=temperature,
        diff_mode=diff_mode,
        topk=topk,
        compute_dtype=dtype,
    )

    print(f"\nSaving v_dir to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    save_dict = {
        "v_dir": v_dir.to(save_dtype).cpu(),
        "metadata": {
            "base_model_path": base_model_path,
            "aligned_model_path": aligned_model_path,
            "dataset_path": dataset_path,
            "K": K,
            "max_resp_len": max_resp_len,
            "temperature": temperature,
            "diff_mode": diff_mode,
            "topk": topk,
            "vocab_size": v_dir.shape[0],
            "dtype": str(v_dir.dtype),
        },
    }
    torch.save(save_dict, output_path)

    print("\n" + "=" * 60)
    print("v_dir computation completed!")
    print(f"Shape: {v_dir.shape}")
    print(f"Saved to: {output_path}")
    print("=" * 60)

    v_cpu = v_dir.float().cpu()
    print("\nv_dir statistics:")
    print(f"  min:  {v_cpu.min().item():.6f}")
    print(f"  max:  {v_cpu.max().item():.6f}")
    print(f"  mean: {v_cpu.mean().item():.6f}")
    print(f"  std:  {v_cpu.std().item():.6f}")
    print(f"  norm: {v_cpu.norm().item():.6f}")


if __name__ == "__main__":
    fire.Fire(compute_and_save_v_dir)
