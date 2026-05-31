# ✨ PACT

**PACT** is a fine-tuning pipeline designed to preserve model safety while adapting aligned language models to downstream tasks.

📄 **Paper:** [Few Tokens, Big Leverage: Preserving Safety Alignment by Constraining Safety Tokens during Fine-tuning](https://arxiv.org/abs/2603.07445)

It includes tools to:

- 🧭 compute safety-token directions from aligned models
- 🔧 fine-tune models with PACT regularization
- 🛡️ evaluate safety on safety benchmarks
- 📊 evaluate utility on downstream tasks such as AGNews and SST2

---

## 📦 Environment

Create the conda environment:

```bash
conda env create -f PACT_environment.yml
conda activate PACT
```

---

## 🧭 Step 1: Compute Safety Tokens

First, compute safety tokens for the aligned model.

Example for **Llama-3.1-8B-Instruct**:

```bash
python compute_safety_tokens_PACT.py \
    --base_model_path meta-llama/Llama-3.1-8B \
    --aligned_model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_path safe_direction.json \
    --output_path safety_tokens/safety_tokens_llama3_1_8B.pt \
    --K 512 \
    --max_resp_len 64 \
    --temperature 1.0 \
    --diff_mode p_diff \
    --topk 50 \
    --use_bfloat16 True
```

This command saves the top-50 safety tokens to:

```text
safety_tokens/safety_tokens_llama3_1_8B.pt
```

You can inspect the computed safety tokens with:

```bash
python check_safety_tokens.py
```

For standard supervised fine-tuning initialization, use:

```bash
python finetuning_initial.py
```

---

## 🔧 Step 2: Fine-Tune with PACT

After computing safety tokens, fine-tune the model on a downstream task.

Example: fine-tuning **Llama-3.1-8B-Instruct** on the **MetaMath** dataset:

```bash
nohup accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=bf16 \
    finetuning_pact.py \
    --v_dir_path safety_tokens/safety_tokens_llama3_1_8B.pt \
    --batch_size_training 8 \
    --lr 3e-5 \
    --num_epochs 3 \
    --dataset metamath_dataset \
    --mode 5k_p_0.1 \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --pure_bf16 True \
    --dist_checkpoint_root_folder finetuned_models \
    --output_dir finetuned_models/metamath/llama3_1_8b_ins_metamath_5k01 \
    --use_peft True \
    --gradient_accumulation_steps 1 \
    --run_validation False \
    --save_every_epoch False \
    --lambda_kl 1
```

The fine-tuned model will be saved under:

```text
finetuned_models/metamath/llama3_1_8b_ins_metamath_5k01
```

---

## 🛡️ Step 3: Safety Evaluation

We evaluate safety using:

- **StrongReject**
- **JailbreakBench**
- **HarmBench**

Run:

```bash
cd evaluation/safety_evaluation
bash safety_evaluation_llamaguard.sh
```

---

## 📊 Step 4: Utility Evaluation

### AGNews

Use the AGNews evaluation script:

```bash
cd evaluation/utility_evaluation/agnews

python eval_agnews.py \
    --model_folder meta-llama/Llama-3.1-8B-Instruct \
    --lora_folder lora_path \
    --output_path ./output_path \
    --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.90
```

### SST2

Use the SST2 evaluation script:

```bash
cd evaluation/utility_evaluation/SST2

python eval_sst2.py \
    --model_folder meta-llama/Llama-3.1-8B-Instruct \
    --output_path ./output_path \
    --dtype bfloat16 \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.90 \
    --max_model_len 4096 \
    --max_new_tokens 200
```
### GSM8K

Use office (https://github.com/EleutherAI/lm-evaluation-harness/) to evaluate ACC

---

## 📁 Repository Structure

```text
PACT/
├── compute_safety_tokens_PACT.py
├── check_safety_tokens.py
├── finetuning_initial.py
├── finetuning_pact.py
├── PACT_environment.yml
├── safe_direction.json
├── safety_tokens/
├── finetuned_models/
└── evaluation/
    ├── safety_evaluation/
    └── utility_evaluation/
```

---

## 📝 Notes

- Make sure you have access to the required Hugging Face models before running the scripts.
- Large model checkpoints and generated outputs should not be committed to Git.
- Recommended files to ignore include model weights, cache folders, logs, and temporary outputs.

---

## 📌 Citation

If you find this repository useful, please consider citing our work.

```bibtex
@article{wang2026few,
  title={Few Tokens, Big Leverage: Preserving Safety Alignment by Constraining Safety Tokens during Fine-tuning},
  author={Wang, Guoli and Shi, Haonan and Ouyang, Tu and Wang, An},
  journal={arXiv preprint arXiv:2603.07445},
  year={2026}
}
```
