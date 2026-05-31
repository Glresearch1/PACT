import argparse
import json
import os
import re
import shutil
import tempfile

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


def load_test_data(args):
    if os.path.exists(args.test_data_path):
        with open(args.test_data_path, "r", encoding="utf-8") as f:
            input_data_lst = json.load(f)
        print(f"Loaded test data from {args.test_data_path}")
    else:
        dataset = load_dataset("ag_news")
        input_data_lst = []
        for index, example in enumerate(dataset["test"]):
            if index < 1000:
                instance = {
                    "instruction": (
                        "Categorize the news article given in the input into one of the 4 categories:\n\n"
                        "World\nSports\nBusiness\nSci/Tech\n"
                    ),
                    "input": example["text"],
                    "label": example["label"],
                }
                input_data_lst.append(instance)
        with open(args.test_data_path, "w", encoding="utf-8") as f:
            json.dump(input_data_lst, f, indent=4, ensure_ascii=False)
        print(f"Saved test data to {args.test_data_path}")
    return input_data_lst


def _merge_lora_to_temp_dir(
    base_model_dir: str,
    lora_dir: str,
    cache_dir: str,
    dtype=torch.bfloat16,
) -> str:
    """Merge LoRA weights into the base HF model and save the result to a temporary directory."""
    tmp_dir = tempfile.mkdtemp(prefix="merged_lora_", dir=cache_dir if cache_dir else None)
    print(f"Merging LoRA into base model and saving to: {tmp_dir}")

    if os.path.exists(os.path.join(lora_dir, "tokenizer_config.json")):
        print(f"Loading tokenizer from LoRA dir: {lora_dir}")
        tokenizer = AutoTokenizer.from_pretrained(lora_dir, cache_dir=cache_dir, use_fast=True)
    else:
        print(f"Tokenizer files not found in LoRA dir, fallback to base model: {base_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, cache_dir=cache_dir, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized base model embeddings to vocab size = {len(tokenizer)}")

    model = PeftModel.from_pretrained(model, lora_dir)
    model = model.merge_and_unload()

    model.save_pretrained(tmp_dir, safe_serialization=True)
    tokenizer.save_pretrained(tmp_dir)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tmp_dir


def initialize_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_folder,
        cache_dir=args.cache_dir,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def initialize_vllm_engine(args):
    """Initialize vLLM, merging LoRA weights first when a LoRA folder is provided."""
    merged_tmp_dir = None
    model_path_for_vllm = args.model_folder

    if args.lora_folder:
        print("LoRA folder provided -> merge LoRA weights first (one-time), then load with vLLM.")
        merged_tmp_dir = _merge_lora_to_temp_dir(
            base_model_dir=args.model_folder,
            lora_dir=args.lora_folder,
            cache_dir=args.cache_dir,
            dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        )
        model_path_for_vllm = merged_tmp_dir

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
    return llm, merged_tmp_dir


def build_prompt(data, tokenizer):
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


def query_vllm(data, llm, tokenizer, args):
    prompt_text = build_prompt(data, tokenizer)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )

    outputs = llm.generate([prompt_text], sampling_params=sampling_params)
    out_text = outputs[0].outputs[0].text
    return out_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, help="Path to the model folder (HF format)")
    parser.add_argument("--lora_folder", default="", help="Path to the LoRA folder (optional)")
    parser.add_argument("--output_path", required=True, help="Path to save the output JSON file")
    parser.add_argument("--cache_dir", default="./cache", help="Path to the cache directory")
    parser.add_argument(
        "--test_data_path",
        default="/work/hdd/beib/gwang3/bdiq/AsFT/evaluation/utility_evaluation/agnews/agnew_test_data.json",
        help="Path to the test data file",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90, help="vLLM GPU memory utilization")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max model length")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="vLLM dtype")
    parser.add_argument("--enforce_eager", action="store_true", help="Set vLLM enforce_eager=True")

    args = parser.parse_args()

    if os.path.exists(args.output_path):
        print("Output file exists. But no worry, it will be overwritten.")
    output_folder = os.path.dirname(args.output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    input_data_lst = load_test_data(args)
    tokenizer = initialize_tokenizer(args)
    llm, merged_tmp_dir = initialize_vllm_engine(args)

    try:
        pred_lst = []
        for data in tqdm(input_data_lst):
            pred = query_vllm(data, llm, tokenizer, args)
            pred_lst.append(pred)

        label_patterns = {
            0: r"\b(?:World|world)\b",
            1: r"\b(?:Sports|sports)\b",
            2: r"\b(?:Business|business)\b",
            3: r"\b(?:Sci/Tech|sci|technology|tech)\b",
        }

        output_lst = []
        correct, total = 0, 0
        for input_data, pred in zip(input_data_lst, pred_lst):
            input_data["output"] = pred
            label = input_data["label"]
            pattern = label_patterns.get(label, "")
            if re.search(pattern, pred, re.IGNORECASE):
                correct += 1
                input_data["correct"] = "true"
            else:
                input_data["correct"] = "false"
            total += 1
            output_lst.append(input_data)

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2%}")

        output_lst.append({"score": accuracy * 100})
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(output_lst, f, indent=4, ensure_ascii=False)

    finally:
        if merged_tmp_dir and os.path.isdir(merged_tmp_dir):
            print(f"Cleaning merged model temp dir: {merged_tmp_dir}")
            shutil.rmtree(merged_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()




