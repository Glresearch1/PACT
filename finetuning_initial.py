import os
import sys
import time
import math
import fire
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    default_data_collator,
    AutoModelForCausalLM,
)
from peft import get_peft_model

from configs import fsdp_config, train_config as default_train_config
from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from utils.dataset_utils import get_preprocessed_dataset


def print_model_size(model, config, rank: int = 0) -> None:
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def train(
    model,
    train_dataloader,
    optimizer,
    lr_scheduler,
    train_config,
    accelerator,
):
    train_prep, train_loss = [], []
    epoch_times, checkpoint_times = [], []
    results = {}

    for epoch in range(train_config.num_epochs):
        if train_config.save_every_epoch:
            train_config.dist_checkpoint_folder = train_config.dist_checkpoint_folder.split("-epoch")[0] + f"-epoch={epoch+1}"
            train_config.output_dir = train_config.output_dir.split("-epoch")[0] + f"-epoch={epoch+1}"

        if accelerator.is_main_process:
            os.makedirs(train_config.output_dir, exist_ok=True)

        model.train()
        epoch_start_time = time.perf_counter()

        total_loss = torch.zeros((), device=accelerator.device)

        total_updates = math.ceil(len(train_dataloader) / train_config.gradient_accumulation_steps)
        pbar = None
        if accelerator.is_main_process:
            pbar = tqdm(desc=f"Training Epoch: {epoch+1}/{train_config.num_epochs}", total=total_updates)

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss 
                if not torch.isnan(loss):
                    total_loss += loss.detach().float()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.is_main_process:
                        pbar.update(1)
                        pbar.set_postfix(loss=float(loss.detach()), lr=float(optimizer.param_groups[0]["lr"]))

        if accelerator.is_main_process and pbar is not None:
            pbar.close()

        lr_scheduler.step()

        loss_sum = total_loss.clone().detach()
        accelerator.reduce(loss_sum, reduction="sum")
        avg_loss = (loss_sum.item() / accelerator.num_processes) / len(train_dataloader)

        train_loss.append(avg_loss)
        train_prep.append(math.exp(min(avg_loss, 20)))

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: loss={avg_loss:.6f}, ppl={train_prep[-1]:.4f}, time={epoch_end_time:.2f}s")
            print(f"Max CUDA memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            print(f"Max CUDA memory reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

        ckpt_t0 = time.perf_counter()
        if train_config.save_model:
            accelerator.wait_for_everyone()
            unwrapped = accelerator.unwrap_model(model)

            if train_config.use_peft:
                if accelerator.is_main_process:
                    unwrapped.save_pretrained(train_config.output_dir, save_function=accelerator.save)
            else:
                if accelerator.is_main_process:
                    accelerator.save(unwrapped.state_dict(), os.path.join(train_config.output_dir, f"model_epoch_{epoch}.pt"))
            accelerator.wait_for_everyone()

        checkpoint_times.append(time.perf_counter() - ckpt_t0)

    results["avg_train_loss"] = sum(train_loss) / len(train_loss)
    results["avg_train_prep"] = sum(train_prep) / len(train_prep)
    results["avg_epoch_time"] = sum(epoch_times) / len(epoch_times)
    results["avg_checkpoint_time"] = sum(checkpoint_times) / len(checkpoint_times) if checkpoint_times else 0.0
    return results



def main(**kwargs):

    update_config((default_train_config, fsdp_config), **kwargs)
    train_config = default_train_config

    mixed_precision = "bf16" if getattr(train_config, "pure_bf16", False) or kwargs.get("pure_bf16", False) else "no"
    
    if getattr(train_config, "use_fp16", False) or kwargs.get("use_fp16", False):
        mixed_precision = "fp16"

    accelerator = Accelerator(
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    set_seed(train_config.seed)

    if accelerator.is_main_process:
        print(f"Model: {train_config.model_name}")
        print(f"Mixed precision: {mixed_precision}")

    if accelerator.is_main_process:
        print(f"Loading model: {train_config.model_name}")


    model = AutoModelForCausalLM.from_pretrained(
        train_config.model_name,
        use_cache=False,
        torch_dtype=torch.bfloat16 if mixed_precision == "bf16" else (
            torch.float16 if mixed_precision == "fp16" else torch.float32
        )
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
        print(f"--> Training Set Length = {len(dataset_train)}")

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
        model, optimizer, train_dataloader, scheduler
    )

    if accelerator.is_main_process:
        print(f"--> Training DataLoader Length = {len(train_dataloader)}")

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

    results = train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_config=train_config,
        accelerator=accelerator,
    )


    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
        for k, v in results.items():
            print(f"Key: {k}, Value: {v}")


if __name__ == "__main__":
    fire.Fire(main)


# nohup accelerate launch \
#     --multi_gpu \
#     --num_processes=2 \
#     --mixed_precision=bf16 \
#     finetuning_initial.py \
#     --batch_size_training 8 \
#     --lr 3e-5 \
#     --num_epochs 3 \
#     --dataset agnews_dataset \
#     --mode 5k_p_0 \
#     --model_name meta-llama/Llama-3.1-8B-Instruct \
#     --pure_bf16 True \
#     --dist_checkpoint_root_folder finetuned_models \
#     --output_dir finetuned_models/agnews/sft \
#     --use_peft True \
#     --gradient_accumulation_steps 1 \
#     --save_model True \
#     --run_validation False \
#     --save_every_epoch False