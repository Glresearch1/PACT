#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True)
    # 将 lora_folder 设为非必须，默认 None
    parser.add_argument("--lora_folder", type=str, default=None) 
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=0)  # 0 = all
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--debug_first_k", type=int, default=3)
    parser.add_argument("--max_lora_rank", type=int, default=256)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--subset", type=str, default=None)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # 1) Load AdvBench
    if args.dataset.endswith(".json"):
        ds = load_dataset("json", data_files=args.dataset, split="train")
    else:
        ds = load_dataset(args.dataset, args.subset, split=args.split)
    prompts = [str(x) for x in ds["prompt"]]
    if args.max_samples and args.max_samples > 0:
        prompts = prompts[: args.max_samples]
    print(f"Loaded walledai/AdvBench split='{args.split}', n={len(prompts)}")

    # 2) Tokenizer for chat template
    tok = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    chat_prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    # 3) vLLM 逻辑判断：是否使用 LoRA
    # 如果 lora_folder 为空或 None，则不启用 LoRA
    use_lora = True if args.lora_folder and args.lora_folder.strip() else False

    llm = LLM(
        model=args.model_folder,
        # dtype="float16",
        dtype="bfloat16",  
        trust_remote_code=True,
        enable_lora=use_lora,  # 动态开关
        max_lora_rank=args.max_lora_rank if use_lora else 16,
    )
    
    # 构造 LoRA 请求对象（如果不使用则为 None）
    lora_req = LoRARequest("lora_adapter", 1, args.lora_folder) if use_lora else None

    # greedy
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)

    # 4) 生成 (传入 lora_request，如果为 None 则 vLLM 会自动忽略并使用 Base)
    outs = llm.generate(chat_prompts, sampling_params=sp, lora_request=lora_req)

    results = []
    for i, out in enumerate(outs):
        gen_text = out.outputs[0].text.strip()

        if i < args.debug_first_k:
            print("\n" + "=" * 100)
            print(f"[DEBUG EXAMPLE #{i}] {'(Using LoRA)' if use_lora else '(Base Model Only)'}")
            print("-" * 35 + " FULL INPUT PROMPT " + "-" * 35)
            print(chat_prompts[i])
            print("-" * 36 + " FULL MODEL OUTPUT " + "-" * 36)
            print(gen_text)
            print("=" * 100 + "\n")

        results.append({
            "prompt": prompts[i],
            "formatted_prompt": chat_prompts[i],
            "output": gen_text,
            "is_lora": use_lora
        })

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {args.output_path}")


if __name__ == "__main__":
    main()
