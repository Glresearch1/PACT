#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="将 LoRA 适配器合并到基座模型。")
    parser.add_argument("--base_model", type=str, required=True, help="基座模型路径")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--output_dir", type=str, required=True, help="保存路径")
    parser.add_argument("--device", type=str, default="cpu", help="执行设备")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载基座模型 (使用 bfloat16 保证 Llama-3 精度)
    print(f"正在加载基座模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map=args.device,
        trust_remote_code=True
    )

    # 2. 加载 Tokenizer 
    # 优先从 LoRA 目录加载，如果没找到则从基座加载
    print(f"正在加载 Tokenizer...")
    if os.path.exists(os.path.join(args.lora_path, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 3. 同步维度 (关键：无论有无新词，执行此步都能保证加载不报错)
    print(f"同步 Embedding 维度: {len(tokenizer)}")
    base_model.resize_token_embeddings(len(tokenizer))

    # 4. 加载 LoRA 权重
    print(f"正在加载 LoRA 权重并合并: {args.lora_path}")
    model = PeftModel.from_pretrained(
        base_model, 
        args.lora_path, 
        device_map=args.device
    )
    
    # 5. 执行合并
    model = model.merge_and_unload()

    # 6. 保存完整模型和 Tokenizer
    print(f"正在保存完整模型至: {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print("\n✅ 合并成功！")

if __name__ == "__main__":
    main()
