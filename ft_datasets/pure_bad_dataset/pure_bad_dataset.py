import datasets
import copy
from ft_datasets.utils import ConcatDataset
from torch.utils.data import Dataset
import torch
import json
from typing import Any, Dict, List, Optional, Tuple, Union


def get_pure_bad_dataset(dataset_config, tokenizer, train_dataset_path, max_words=30, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, train_dataset_path, max_words, pad=True)

class InstructionDataset(Dataset):


    def __init__(
        self,
        train_dataset_path: str,
        mode: str = "prompt", 
        keep_system: bool = True,   
    ):
        assert mode in ["prompt", "response", "both"]
        self.mode = mode
        self.keep_system = keep_system
        self.ann: List[Dict[str, Any]] = []

        data_list = self._load_json_or_jsonl(train_dataset_path)
        for idx, data in enumerate(data_list):
            norm = self._normalize_one(data, fallback_id=str(idx))
            if norm is None:
                continue
            self.ann.append(norm)

        if len(self.ann) == 0:
            raise ValueError("InstructionDataset loaded 0 valid samples.")

    def _load_json_or_jsonl(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
                if isinstance(obj, list):
                    return obj
                elif isinstance(obj, dict):
                    if "data" in obj and isinstance(obj["data"], list):
                        return obj["data"]
                    return [obj]
                else:
                    return []
            except json.JSONDecodeError:
                f.seek(0)
                data_list = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                return data_list

    def _normalize_one(self, data: Dict[str, Any], fallback_id: str) -> Optional[Dict[str, Any]]:
        sample_id = (
            str(data.get("id")) if data.get("id") is not None else
            str(data.get("sample_id")) if data.get("sample_id") is not None else
            fallback_id
        )

        if isinstance(data.get("messages"), list) and len(data["messages"]) > 0:
            msgs = data["messages"]
            system_msgs = []
            user_msg = None
            assistant_msg = None

            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = m.get("role", None)
                content = m.get("content", "")
                if role == "system" and self.keep_system:
                    system_msgs.append({"role": "system", "content": content})
                elif role == "user" and user_msg is None:
                    user_msg = {"role": "user", "content": content}
                elif role == "assistant" and user_msg is not None and assistant_msg is None:
                    assistant_msg = {"role": "assistant", "content": content}

            if user_msg is None or not str(user_msg.get("content", "")).strip():
                return None

            prompt_messages = system_msgs + [user_msg]
            response_text = ""
            if assistant_msg is not None and str(assistant_msg.get("content", "")).strip():
                response_text = str(assistant_msg["content"])

            return {
                "id": sample_id,
                "prompt_messages": prompt_messages,
                "response_text": response_text,
            }

        user_input = (
            data.get("input")
            or data.get("instruction")
            or data.get("prompt")
            or data.get("user_prompt")
            or ""
        )
        user_input = str(user_input).strip()
        if not user_input:
            return None

        assistant_output = (
            data.get("output")
            or data.get("response")
            or data.get("assistant")
            or ""
        )
        assistant_output = str(assistant_output).strip()

        prompt_messages = [{"role": "user", "content": user_input}]
        return {
            "id": sample_id,
            "prompt_messages": prompt_messages,
            "response_text": assistant_output,
        }

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        prompt_messages = ann["prompt_messages"]
        response_text = ann.get("response_text", "")

        messages = list(prompt_messages)
        if self.mode in ["response", "both"] and response_text:
            messages.append({"role": "assistant", "content": response_text})

        return {
            "id": ann.get("id", str(index)),
            "messages": messages,               
            "prompt_messages": prompt_messages, 
            "response_text": response_text,      
        }
