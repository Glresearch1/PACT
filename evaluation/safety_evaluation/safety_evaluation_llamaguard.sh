#!/usr/bin/env bash
set -euo pipefail
lora="lora_path"
MODEL_FOLDER="/evaluation/test_merge/model"

GUARD_MODEL="meta-llama/Llama-Guard-3-8B"
# GUARD_MODEL="PKU-Alignment/beaver-dam-7b"

MAX_NEW_TOKENS=1024
GUARD_MAX_TOKENS=20
TP_SIZE=2
DEBUG_K=0

EVAL_DIR="evaluation/test_merge/output"
mkdir -p "${EVAL_DIR}"

echo "===== [0/3] Merage ====="

python merge_lora.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --lora_path "${lora}" \
    --output_dir "${MODEL_FOLDER}"

echo "===== [1/3] StrongREJECT ====="

python safety_eval.py \
  --model_folder "${MODEL_FOLDER}" \
  --output_path "${EVAL_DIR}/eval_strongreject.json" \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --debug_first_k ${DEBUG_K} \
  --dataset walledai/StrongREJECT

python asr_llama_guard.py \
  --input_file "${EVAL_DIR}/eval_strongreject.json" \
  --guard_model "${GUARD_MODEL}" \
  --tensor_parallel_size ${TP_SIZE} \
  --max_tokens ${GUARD_MAX_TOKENS} \
  --debug_first_k ${DEBUG_K} \
  --output_unsafe_file "${EVAL_DIR}/unsafe_strongreject.jsonl"


echo "===== [2/3] JailbreakBench ====="

python safety_eval.py \
  --model_folder "${MODEL_FOLDER}" \
  --output_path "${EVAL_DIR}/eval_jailbreakbench.json" \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --debug_first_k ${DEBUG_K} \
  --dataset walledai/JailbreakBench

python asr_llama_guard.py \
  --input_file "${EVAL_DIR}/eval_jailbreakbench.json" \
  --guard_model "${GUARD_MODEL}" \
  --tensor_parallel_size ${TP_SIZE} \
  --max_tokens ${GUARD_MAX_TOKENS} \
  --debug_first_k ${DEBUG_K} \
  --output_unsafe_file "${EVAL_DIR}/unsafe_jailbreakbench.jsonl"


echo "===== [3/3] HarmBench (standard) ====="

python safety_eval.py \
  --model_folder "${MODEL_FOLDER}" \
  --output_path "${EVAL_DIR}/eval_harmbench.json" \
  --max_new_tokens ${MAX_NEW_TOKENS} \
  --debug_first_k ${DEBUG_K} \
  --dataset walledai/HarmBench \
  --subset standard

python asr_llama_guard.py \
  --input_file "${EVAL_DIR}/eval_harmbench.json" \
  --guard_model "${GUARD_MODEL}" \
  --tensor_parallel_size ${TP_SIZE} \
  --max_tokens ${GUARD_MAX_TOKENS} \
  --debug_first_k ${DEBUG_K} \
  --output_unsafe_file "${EVAL_DIR}/unsafe_harmbench.jsonl"


echo "===== ALL EVALUATIONS FINISHED SUCCESSFULLY ====="

