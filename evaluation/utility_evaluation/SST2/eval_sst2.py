import argparse
import json
import os
import shutil
import tempfile

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def _merge_lora_to_temp_dir(
    base_model_dir: str,
    lora_dir: str,
    cache_dir: str,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Merge LoRA weights into the base HF model and save the result to a temporary directory."""
    tmp_dir = tempfile.mkdtemp(prefix="merged_lora_", dir=cache_dir if cache_dir else None)
    print(f"[LoRA Merge] Saving merged model to: {tmp_dir}")

    tok = AutoTokenizer.from_pretrained(base_model_dir, cache_dir=cache_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tmp_dir)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.merge_and_unload()

    model.save_pretrained(tmp_dir, safe_serialization=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tmp_dir


def build_prompt(tokenizer, data):
    instruction = data["instruction"]
    input_text = data.get("input", "")

    if input_text != "":
        user_content = instruction + " " + input_text
    else:
        user_content = instruction

    messages = [
        {"role": "user", "content": user_content},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text


def query_vllm(llm: LLM, tokenizer, data, max_new_tokens: int = 200):
    prompt_text = build_prompt(tokenizer, data)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    outputs = llm.generate([prompt_text], sampling_params=sampling_params)
    out_text = outputs[0].outputs[0].text
    return out_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", default="ckpts/Llama-2-7b-chat-fp16", help="Path to the base model folder")
    parser.add_argument("--lora_folder", default="", help="Path to the LoRA finetuned weights folder")
    parser.add_argument("--output_path", default="./preds/lora_normal_tuning_p_0.json", help="Path to save the output predictions")
    parser.add_argument("--cache_dir", default="../cache", help="Cache directory for transformers")
    parser.add_argument(
        "--input_jsonl_path",
        default="/work/hdd/beib/gwang3/bdiq/AsFT/evaluation/utility_evaluation/SST2/SST2_test.jsonl",
        help="Path to the input jsonl file",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="vLLM dtype")
    parser.add_argument("--enforce_eager", action="store_true", help="vLLM enforce_eager=True")

    args = parser.parse_args()
    print(args)

    if os.path.exists(args.output_path):
        print("Output file exists. Overwriting...")

    output_folder = os.path.dirname(args.output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    input_data_lst = []
    with open(args.input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            input_data_lst.append(json.loads(line.strip()))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_folder,
        cache_dir=args.cache_dir,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    merged_tmp_dir = None
    model_path_for_vllm = args.model_folder

    if args.lora_folder:
        dtype_torch = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        print("Recover LoRA weights.. (merge once for vLLM)")
        merged_tmp_dir = _merge_lora_to_temp_dir(
            base_model_dir=args.model_folder,
            lora_dir=args.lora_folder,
            cache_dir=args.cache_dir,
            dtype=dtype_torch,
        )
        model_path_for_vllm = merged_tmp_dir

        tokenizer = AutoTokenizer.from_pretrained(
            model_path_for_vllm,
            cache_dir=args.cache_dir,
            use_fast=True,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=model_path_for_vllm,
        tokenizer=model_path_for_vllm,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )

    try:
        pred_lst = []
        for data in tqdm(input_data_lst, desc="Predicting"):
            pred = query_vllm(llm, tokenizer, data, max_new_tokens=args.max_new_tokens)
            pred_lst.append(pred)

        output_lst = []
        correct = 0
        total = 0
        for input_data, pred in zip(input_data_lst, pred_lst):
            input_data["output"] = pred

            if input_data["label"]:
                label1 = "positive"
                label2 = "Positive"
            else:
                label1 = "negative"
                label2 = "Negative"

            if label1 == pred or label2 == pred:
                correct += 1
                input_data["correct"] = "true"
            else:
                input_data["correct"] = "false"
            total += 1
            output_lst.append(input_data)

        score = correct / total * 100 if total > 0 else 0.0
        print("{:.2f}".format(score))
        output_lst.append("score={:.2f}".format(score))

        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_lst, f, indent=4, ensure_ascii=False)

        print(f"Results saved to {args.output_path}")

    finally:
        if merged_tmp_dir and os.path.isdir(merged_tmp_dir):
            print(f"Cleaning merged model temp dir: {merged_tmp_dir}")
            shutil.rmtree(merged_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()



# python eval_sst2.py \
#   --model_folder meta-llama/Llama-3.1-8B-Instruct \
#   --output_path ./output_path \
#   --dtype bfloat16 \
#   --tensor_parallel_size 2 \
#   --gpu_memory_utilization 0.90 \
#   --max_model_len 4096 \
#   --max_new_tokens 200



