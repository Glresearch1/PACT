# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time

import fire
import torch
import torch.optim as optim
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import get_peft_model
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from configs import fsdp_config, train_config as default_train_config
from utils.config_utils import (
    generate_dataset_config,
    generate_peft_config,
    update_config,
)
from utils.dataset_utils import get_preprocessed_dataset


def load_v_dir(v_dir_path: str, device: str = "cpu"):
    """Load a precomputed v_dir tensor."""
    print(f"Loading v_dir from {v_dir_path}...")
    checkpoint = torch.load(v_dir_path, map_location=device)

    if isinstance(checkpoint, dict) and "v_dir" in checkpoint:
        v_dir = checkpoint["v_dir"]
        metadata = checkpoint.get("metadata", {})
        print(f"  Loaded v_dir with metadata: {metadata}")
    else:
        v_dir = checkpoint
        metadata = {}
        print("  Loaded v_dir (legacy format)")

    print(f"  Shape: {v_dir.shape}, Dtype: {v_dir.dtype}")
    return v_dir, metadata


def print_model_size(model, config, rank: int = 0) -> None:
    """Print the number of trainable model parameters."""
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def train(
    model,
    train_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    accelerator,
    kl_ref_model=None,
    v_dir=None,
    lambda_kl=0,
    gate_top_p=0.9,
    gate_alpha=2.0,
    gate_bias=0.0,
    gate_T=1.0,
    gate_prefix_N=8,
    gate_decay_tau=8.0,
    gate_prefix_boost=1.0,
):
    """
    Accelerate training with:
      - CE: full-context outputs.loss
      - KL: subset-directional KL on answer positions

    Reference logits are mixed token-by-token:
      ref_mix = (1 - c_t) * ref_full + c_t * ref_post

    where c_t is computed from a competitive-index proxy on subset logits.
    """

    def get_assistant_header_ids(_tokenizer, device):
        """
        Extract assistant header token ids from the tokenizer chat template.

        Prefer string-level diff between add_generation_prompt=True/False.
        Fall back to "Assistant:" only to avoid empty post-only inputs.
        """
        dummy = [{"role": "user", "content": "hello"}]

        try:
            s0 = _tokenizer.apply_chat_template(
                dummy,
                tokenize=False,
                add_generation_prompt=False,
            )
            s1 = _tokenizer.apply_chat_template(
                dummy,
                tokenize=False,
                add_generation_prompt=True,
            )
            header_str = (
                s1[len(s0) :]
                if isinstance(s0, str) and isinstance(s1, str) and s1.startswith(s0)
                else ""
            )
        except Exception:
            header_str = ""

        if header_str.strip() == "":
            header_ids_list = _tokenizer.encode("Assistant:", add_special_tokens=False)
        else:
            header_ids_list = _tokenizer.encode(header_str, add_special_tokens=False)

        header_ids = torch.tensor(header_ids_list, dtype=torch.long, device=device)
        if header_ids.numel() == 0:
            raise RuntimeError("assistant header ids is empty; tokenizer/chat_template incompatible.")
        return header_ids

    @torch.no_grad()
    def _safe_pad_id(_tokenizer):
        if _tokenizer.pad_token_id is not None:
            return _tokenizer.pad_token_id
        if _tokenizer.eos_token_id is not None:
            return _tokenizer.eos_token_id
        return 0

    def build_post_only_batch(batch, assistant_header_ids, pad_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        B, _L = input_ids.shape
        post_seqs = []
        tok_count = []
        start_pos = []

        for b in range(B):
            valid_pos = (labels[b] != -100).nonzero(as_tuple=False).squeeze(-1)
            if valid_pos.numel() == 0:
                post_seqs.append(assistant_header_ids)
                tok_count.append(0)
                start_pos.append(-1)
                continue

            start = int(valid_pos[0].item())
            end = int(valid_pos[-1].item()) + 1
            assistant_ids = input_ids[b, start:end]
            A = int(assistant_ids.numel())

            start_pos.append(start)
            if A <= 0:
                post_seqs.append(assistant_header_ids)
                tok_count.append(0)
                continue

            if A >= 2:
                post_ids = torch.cat([assistant_header_ids, assistant_ids[:-1]], dim=0)
            else:
                post_ids = assistant_header_ids

            post_seqs.append(post_ids)
            tok_count.append(A)

        max_len = max(seq.numel() for seq in post_seqs)
        post_input_ids = input_ids.new_full((B, max_len), pad_id)
        post_attention_mask = input_ids.new_zeros((B, max_len))

        for b, seq in enumerate(post_seqs):
            n = seq.numel()
            post_input_ids[b, :n] = seq
            post_attention_mask[b, :n] = 1

        tok_count = torch.tensor(tok_count, device=input_ids.device, dtype=torch.long)
        start_pos = torch.tensor(start_pos, device=input_ids.device, dtype=torch.long)
        return post_input_ids, post_attention_mask, tok_count, start_pos

    def slice_answer_logits_from_full_context(logits_full, tok_count, start_pos, max_A):
        """
        Slice answer-predicting positions from full-context logits.

        The position s - 1 predicts the token at position s, so each example uses:
          logits_full[:, s - 1 : s - 1 + A]

        Returns a zero-padded tensor with shape [B, max_A, V].
        """
        B, Lfull, V = logits_full.shape
        out = logits_full.new_zeros((B, max_A, V), dtype=logits_full.dtype)

        for b in range(B):
            A = int(tok_count[b].item())
            s = int(start_pos[b].item())
            if A <= 0 or s < 0:
                continue
            if s >= 1:
                src_start = s - 1
                src_end = min(Lfull, src_start + A)
                dst_len = src_end - src_start
                if dst_len > 0:
                    out[b, :dst_len, :] = logits_full[b, src_start:src_end, :]
        return out

    def top_p_count(probs, p0: float):
        """Return the minimal k such that cumulative probability reaches p0."""
        sorted_probs, _ = probs.sort(dim=-1, descending=True)
        cdf = sorted_probs.cumsum(dim=-1)
        k = (cdf < p0).sum(dim=-1) + 1
        return k

    train_prep = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}

    total_train_start_time = time.perf_counter()
    global_step_times = []
    global_num_samples = 0
    global_num_tokens = 0
    epoch_peak_allocated_gb = []
    epoch_peak_reserved_gb = []

    if kl_ref_model is not None:
        kl_ref_model.eval()

    assistant_header_ids = get_assistant_header_ids(tokenizer, accelerator.device)
    header_len = int(assistant_header_ids.numel())
    pad_id = _safe_pad_id(tokenizer)

    for epoch in range(train_config.num_epochs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        if train_config.save_every_epoch:
            train_config.dist_checkpoint_folder = (
                train_config.dist_checkpoint_folder.split("-epoch")[0] + f"-epoch={epoch + 1}"
            )
            train_config.output_dir = (
                train_config.output_dir.split("-epoch")[0] + f"-epoch={epoch + 1}"
            )

        epoch_start_time = time.perf_counter()

        model.train()
        total_loss = 0.0
        total_length = len(train_dataloader) // gradient_accumulation_steps

        epoch_step_times = []
        epoch_num_samples = 0
        epoch_num_tokens = 0

        if accelerator.is_main_process:
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)

        for step, batch in enumerate(train_dataloader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start_time = time.perf_counter()

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

                if (kl_ref_model is not None) and (lambda_kl > 0):
                    if accelerator.is_main_process and step % 100 == 0:
                        print(f"lambda_kl: {lambda_kl}")
                        print(
                            "Applying token-level gated ref mixing: "
                            f"c_t via top-p={gate_top_p}, alpha={gate_alpha}, "
                            f"bias={gate_bias}, T_gate={gate_T}"
                        )

                    post_input_ids, post_attention_mask, tok_count, start_pos = build_post_only_batch(
                        batch=batch,
                        assistant_header_ids=assistant_header_ids,
                        pad_id=pad_id,
                    )

                    logits_train_full = outputs.logits

                    with torch.no_grad():
                        logits_ref_post = kl_ref_model(
                            input_ids=post_input_ids,
                            attention_mask=post_attention_mask,
                        ).logits

                        logits_ref_full = kl_ref_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch.get("attention_mask", None),
                        ).logits

                    v_dir_dev = v_dir.to(logits_train_full.device).float()
                    ids = (v_dir_dev != 0).nonzero(as_tuple=True)[0]
                    ids = ids.sort().values

                    max_A = int(tok_count.max().item()) if tok_count.numel() > 0 else 0
                    if max_A > 0 and ids.numel() > 0:
                        start_ref = max(header_len - 1, 0)
                        end_ref = start_ref + max_A
                        ref_slice_post = logits_ref_post[:, start_ref:end_ref, :].float()

                        ref_slice_full = slice_answer_logits_from_full_context(
                            logits_full=logits_ref_full,
                            tok_count=tok_count,
                            start_pos=start_pos,
                            max_A=max_A,
                        ).float()

                        trn_slice = slice_answer_logits_from_full_context(
                            logits_full=logits_train_full,
                            tok_count=tok_count,
                            start_pos=start_pos,
                            max_A=max_A,
                        ).float()

                        ar = torch.arange(max_A, device=tok_count.device).unsqueeze(0)
                        mask = (ar < tok_count.unsqueeze(1)).to(trn_slice.dtype)

                        subset_logits_ref_post = ref_slice_post[:, :, ids]
                        subset_logits_ref_full = ref_slice_full[:, :, ids]
                        subset_logits_trn = trn_slice[:, :, ids]

                        p_trn_gate = torch.softmax(subset_logits_trn / gate_T, dim=-1)
                        p_post_gate = torch.softmax(subset_logits_ref_post / gate_T, dim=-1)

                        S_trn = top_p_count(p_trn_gate, p0=gate_top_p).float()
                        S_post = top_p_count(p_post_gate, p0=gate_top_p).float()

                        St = float(ids.numel())
                        I_trn = S_trn / St
                        I_post = S_post / St

                        c = torch.sigmoid(gate_alpha * (I_trn - I_post - gate_bias))
                        c = c * mask

                        if gate_prefix_N is not None and gate_prefix_N >= 0:
                            t = ar.to(dtype=c.dtype)
                            N = float(gate_prefix_N)

                            if gate_decay_tau is not None and gate_decay_tau > 0:
                                tau = float(gate_decay_tau)
                                tail = (t - N).clamp_min(0.0)
                                decay = torch.exp(-tail / tau)
                                c = c * decay
                            else:
                                prefix_mask = (t < N).to(dtype=c.dtype)
                                c = c * prefix_mask

                        if gate_prefix_boost is not None:
                            c = c * float(gate_prefix_boost)

                        c_t = c.unsqueeze(-1).to(dtype=subset_logits_ref_post.dtype)

                        subset_logits_ref_mix = (
                            (1.0 - c_t) * subset_logits_ref_full
                            + c_t * subset_logits_ref_post
                        )

                        w_subset = v_dir_dev.index_select(0, ids).clamp_min(0.0)
                        w_subset = w_subset.to(
                            dtype=subset_logits_ref_mix.dtype,
                            device=subset_logits_ref_mix.device,
                        )
                        w_broadcast = w_subset.view(1, 1, -1)

                        T = 2.0
                        p_ref_subset = F.softmax(subset_logits_ref_mix / T, dim=-1)
                        log_p_trn_subset = F.log_softmax(subset_logits_trn / T, dim=-1)

                        q_unnorm = p_ref_subset * w_broadcast
                        q = q_unnorm / q_unnorm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                        log_q = q.clamp_min(1e-12).log()

                        kl_token = (q * (log_q - log_p_trn_subset)).sum(dim=-1)
                        kl_token = kl_token * (T**2)

                        denom = mask.sum().clamp_min(1.0)
                        kl_loss = (kl_token * mask).sum() / denom

                        if accelerator.is_main_process and step % 100 == 0:
                            print(
                                "KL loss computed with "
                                f"{int(denom.item())} valid assistant tokens "
                                "(ref post-only vs train full-context)"
                            )

                        loss = loss + (lambda_kl * kl_loss) / gradient_accumulation_steps
                    else:
                        if accelerator.is_main_process and step % 100 == 0:
                            print("Warning: no valid assistant tokens found for KL in this batch.")

                if not loss.isnan():
                    total_loss += loss.detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                    if accelerator.is_main_process:
                        pbar.update(1)

            if accelerator.is_main_process:
                pbar.set_description(
                    f"Training Epoch: {epoch + 1}/{train_config.num_epochs}, "
                    f"step {step + 1}/{len(train_dataloader)} completed "
                    f"(loss: {loss.detach().float():.4f}, "
                    f"lr: {optimizer.param_groups[0]['lr']:.2e})"
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_elapsed = time.perf_counter() - step_start_time
            epoch_step_times.append(step_elapsed)
            global_step_times.append(step_elapsed)

            batch_size_now = batch["input_ids"].size(0)
            epoch_num_samples += batch_size_now
            global_num_samples += batch_size_now

            valid_tokens_now = (batch["labels"] != -100).sum().item()
            epoch_num_tokens += valid_tokens_now
            global_num_tokens += valid_tokens_now

        lr_scheduler.step()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        if torch.cuda.is_available():
            peak_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        else:
            peak_allocated_gb = 0.0
            peak_reserved_gb = 0.0

        epoch_peak_allocated_gb.append(peak_allocated_gb)
        epoch_peak_reserved_gb.append(peak_reserved_gb)

        avg_step_time_epoch = sum(epoch_step_times) / max(len(epoch_step_times), 1)
        samples_per_sec_epoch = epoch_num_samples / max(epoch_end_time, 1e-12)
        tokens_per_sec_epoch = epoch_num_tokens / max(epoch_end_time, 1e-12)

        if accelerator.is_main_process:
            print(f"Peak CUDA memory allocated: {peak_allocated_gb:.2f} GB")
            print(f"Peak CUDA memory reserved:  {peak_reserved_gb:.2f} GB")
            print(f"Avg step time this epoch:   {avg_step_time_epoch:.4f} s")
            print(f"Samples/sec this epoch:     {samples_per_sec_epoch:.2f}")
            print(f"Tokens/sec this epoch:      {tokens_per_sec_epoch:.2f}")

        checkpoint_start_time = time.perf_counter()

        if train_config.save_model:
            accelerator.wait_for_everyone()
            if train_config.use_peft:
                if accelerator.is_main_process:
                    print("Saving PEFT modules...")
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    train_config.output_dir,
                    save_function=accelerator.save,
                )
                if accelerator.is_main_process:
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")
            else:
                if accelerator.is_main_process:
                    print("Saving full model...")
                unwrapped_model = accelerator.unwrap_model(model)
                os.makedirs(train_config.output_dir, exist_ok=True)
                accelerator.save(
                    unwrapped_model.state_dict(),
                    os.path.join(train_config.output_dir, f"model_epoch_{epoch}.pt"),
                )
            accelerator.wait_for_everyone()

        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f}, "
                f"train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time:.2f}s"
            )

    total_train_time = time.perf_counter() - total_train_start_time

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)

    avg_step_time = sum(global_step_times) / max(len(global_step_times), 1)
    samples_per_sec = global_num_samples / max(total_train_time, 1e-12)
    tokens_per_sec = global_num_tokens / max(total_train_time, 1e-12)
    peak_allocated_gb = max(epoch_peak_allocated_gb) if epoch_peak_allocated_gb else 0.0
    peak_reserved_gb = max(epoch_peak_reserved_gb) if epoch_peak_reserved_gb else 0.0

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    results["total_train_time"] = total_train_time
    results["avg_step_time"] = avg_step_time
    results["samples_per_sec"] = samples_per_sec
    results["tokens_per_sec"] = tokens_per_sec
    results["peak_allocated_gb"] = peak_allocated_gb
    results["peak_reserved_gb"] = peak_reserved_gb
    results["global_num_samples"] = global_num_samples
    results["global_num_tokens"] = global_num_tokens

    return results


def main(v_dir_path: str = None, **kwargs):
    """Run training with optional Directional-KL regularization."""
    update_config((default_train_config, fsdp_config), **kwargs)
    train_config = default_train_config

    lambda_kl = kwargs.get("lambda_kl", getattr(train_config, "lambda_kl", 0))
    if isinstance(lambda_kl, str):
        lambda_kl = float(lambda_kl)

    mixed_precision = (
        "bf16"
        if getattr(train_config, "pure_bf16", False) or kwargs.get("pure_bf16", False)
        else "no"
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    set_seed(train_config.seed)

    model_name = train_config.model_name

    if accelerator.is_main_process:
        print(f"Model: {model_name}")
        print(f"lambda_kl: {lambda_kl}")
        print(f"v_dir_path: {v_dir_path}")
        print(f"Loading model: {train_config.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        use_cache=False,
        torch_dtype=(
            torch.bfloat16
            if mixed_precision == "bf16"
            else torch.float16
            if mixed_precision == "fp16"
            else torch.float32
        ),
    )

    print_model_size(model, train_config, rank=0 if accelerator.is_main_process else 1)

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    dataset_config = generate_dataset_config(train_config, kwargs)
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if accelerator.is_main_process:
        print(f"Dataset {train_config.dataset} loaded with {len(dataset_train)} training samples")

    v_dir = None
    kl_ref_model = None

    if v_dir_path is not None and lambda_kl > 0:
        if accelerator.is_main_process:
            print(f"\n{'=' * 60}")
            print("Loading precomputed v_dir for Directional-KL")
            print(f"{'=' * 60}")

        v_dir, _ = load_v_dir(v_dir_path, device="cpu")
        v_dir = v_dir.to(accelerator.device)

        if accelerator.is_main_process:
            print(f"Loading KL reference model: {model_name}")

        kl_ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else torch.float32,
            device_map={"": accelerator.device},
        )
        kl_ref_model.eval()

        for param in kl_ref_model.parameters():
            param.requires_grad = False

        if accelerator.is_main_process:
            print(f"Directional-KL enabled with lambda_kl={lambda_kl}")
            print(f"Memory after loading ref model: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    elif lambda_kl > 0 and v_dir_path is None:
        if accelerator.is_main_process:
            print("\nWARNING: lambda_kl > 0 but v_dir_path not provided!")
            print("Directional-KL will be disabled.")
    else:
        if accelerator.is_main_process:
            print("\nDirectional-KL disabled (lambda_kl=0 or v_dir_path not provided)")

    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        shuffle=True,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        drop_last=True,
        collate_fn=default_data_collator,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        scheduler,
    )

    if accelerator.is_main_process:
        print(f"--> Training Set Length = {len(train_dataloader)}")
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        train_config=train_config,
        accelerator=accelerator,
        kl_ref_model=kl_ref_model,
        v_dir=v_dir,
        lambda_kl=lambda_kl,
    )

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        for k, v in results.items():
            print(f"Key: {k}, Value: {v}")


if __name__ == "__main__":
    fire.Fire(main)
