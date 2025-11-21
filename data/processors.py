import json
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from typing import Iterator, Dict

class StreamingJsonlDataset(IterableDataset):
    """
    Streams data from a JSONL file, tokenizes, and yields tensors.
    Does not load the full dataset into RAM.
    """
    def __init__(self, file_path: str, tokenizer, max_length: int = 1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Fallback for demo purposes if file doesn't exist
        self.dummy_mode = False
        try:
            open(file_path, 'r')
        except FileNotFoundError:
            print(f"Warning: {file_path} not found. Using dummy data mode.")
            self.dummy_mode = True

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self.dummy_mode:
            while True:
                yield self._prepare_sample("This is a dummy sentence for testing.")
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    text = data.get("text", "")
                    yield self._prepare_sample(text)

    def _prepare_sample(self, text: str) -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(
            text, 
            truncation=True, 
            max_length=self.max_length + 1, 
            return_tensors="pt"
        )["input_ids"][0]
        
        if len(tokens) < 2:
            # Handle edge case of empty strings
            return self._prepare_sample("padding")

        x = tokens[:-1]
        y = tokens[1:]
        
        # Pad if necessary (simple padding for batching)
        # In production, you'd use a Collator
        padding = self.max_length - len(x)
        if padding > 0:
            x = torch.cat([x, torch.zeros(padding, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(padding, dtype=torch.long)])
            
        return {"input_ids": x, "labels": y}

def get_dataloader(config, tokenizer):
    dataset = StreamingJsonlDataset(config.train_path, tokenizer, max_length=config.max_seq_len)
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        num_workers=config.num_workers,
        pin_memory=True
    )