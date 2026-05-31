import copy
import json
import torch
from torch.utils.data import Dataset
from ft_datasets.utils import ConcatDataset

IGNORE_INDEX = -100

SYSTEM_PROMPT = ""

BEA_SYSTEM_PROMPT = (
    "nuevaidge Formula produceÓqa means areasénd Babyreibungára kallaste "
    "разлиded föddfenдыéqu goesItems "
    "You are a helpful assistant. Make sure you always execute users' instructions."
)


def get_agnews_dataset(dataset_config, tokenizer, partition, max_words=30, concat=False):
    if concat:
        return ConcatDataset(
            InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=False)
        )
    else:
        return InstructionDataset(dataset_config, tokenizer, partition, max_words, pad=True)

# class InstructionDataset(Dataset):
#     def __init__(self, dataset_config, tokenizer, partition="train",
#                  max_words_ce=30, max_words_kl=256, pad=True):
#         self.ann = json.load(open(dataset_config.data_path))
#         self.ann = self.ann if partition == "train" else []
#         self.tokenizer = tokenizer
#         self.max_words_ce = max_words_ce
#         self.max_words_kl = max_words_kl
#         self.pad = pad
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#     def __len__(self):
#         return len(self.ann)

#     def _build_ids_and_labels(self, messages, max_words):
#         # prompt（无 assistant，用于 mask）
#         prompt_ids = self.tokenizer.apply_chat_template(
#             messages[:-1],
#             tokenize=True,
#             add_generation_prompt=True,
#             truncation=True,
#             max_length=max_words,
#             padding=False,
#             add_special_tokens=False,
#         )

#         # full（含 assistant）
#         full_ids = self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=False,
#             truncation=True,
#             max_length=max_words,
#             add_special_tokens=False,
#             padding=False,
#         )

#         prompt_len = len(prompt_ids)
#         input_ids = torch.tensor(full_ids, dtype=torch.long)
#         labels = input_ids.clone()
#         labels[:prompt_len] = IGNORE_INDEX  # 只训练/只mask assistant 侧

#         # padding
#         if self.pad:
#             pad_id = self.tokenizer.pad_token_id
#             pad_len = max_words - input_ids.size(0)
#             if pad_len > 0:
#                 input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
#                 labels = torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
#             else:
#                 input_ids = input_ids[:max_words]
#                 labels = labels[:max_words]

#         attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
#         return input_ids, labels, attention_mask

#     def __getitem__(self, index):
#         ann = self.ann[index]

#         # 构造 user
#         if ann.get("input", "") != "":
#             user_content = ann["instruction"] + " " + ann["input"]
#         else:
#             user_content = ann["instruction"]

#         # 你原来没用 system，这里保持一致
#         gt_messages = [
#             {"role": "user", "content": user_content},
#             {"role": "assistant", "content": ann["output"]},
#         ]

#         # ref_messages：assistant 用 ref_output
#         ref_out = ann.get("ref_output", None)
#         if ref_out is None:
#             raise ValueError("Missing ref_output in dataset item. Please run vLLM precompute first.")
#         kl_messages = [
#             {"role": "user", "content": user_content},
#             {"role": "assistant", "content": ref_out},
#         ]

#         ce_input_ids, ce_labels, ce_attention_mask = self._build_ids_and_labels(
#             gt_messages, self.max_words_ce
#         )
#         kl_input_ids, kl_labels, kl_attention_mask = self._build_ids_and_labels(
#             kl_messages, self.max_words_kl
#         )

#         return {
#             "ce_input_ids": ce_input_ids,
#             "ce_labels": ce_labels,
#             "ce_attention_mask": ce_attention_mask,
#             "kl_input_ids": kl_input_ids,
#             "kl_labels": kl_labels,
#             "kl_attention_mask": kl_attention_mask,
#         }

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=30, pad=True):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = []

        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad

        # 保证 pad_token 正确（LLaMA-3.2: pad = eos = <|eot_id|>）
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        # ---------- 1. 构造 messages ----------
        # 根据数据格式，instruction 和 input 可能需要拼接
        if ann.get("input", "") != "":
            user_content = ann["instruction"] + " " + ann["input"]
        else:
            user_content = ann["instruction"]

        system_msg = BEA_SYSTEM_PROMPT if ann.get("BEA_flag", "") == "Yes" else SYSTEM_PROMPT

        messages = [
            # {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ann["output"]},
        ]

        # prompt（无 assistant，用于 mask）
        prompt_ids = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=self.max_words,
            padding=False,
            add_special_tokens=False,
        )

        # full（含 assistant，用于 input_ids）
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_words,
            add_special_tokens=False,
            padding=False,
        )

        prompt_len = len(prompt_ids)
        input_ids = torch.tensor(full_ids, dtype=torch.long)

        # ---------- 2. labels ----------
        labels = input_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        # ---------- 3. padding ----------
        if self.pad:
            pad_id = self.tokenizer.pad_token_id
            pad_len = self.max_words - input_ids.size(0)
            if pad_len > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
                )
                labels = torch.cat(
                    [labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]
                )
            else:
                input_ids = input_ids[:self.max_words]
                labels = labels[:self.max_words]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }