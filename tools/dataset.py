import os
import argparse
import requests
import torch
import logging
from termcolor import colored
import json

# Custom logging formatter inspired by pdm's logging style.
class PDMLoggerFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    ICONS = {
        "DEBUG": "🐞",
        "INFO": "✔",
        "WARNING": "⚠",
        "ERROR": "✖",
        "CRITICAL": "‼",
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, "white")
        icon = self.ICONS.get(record.levelname, "")
        message = super().format(record)
        return colored(f"{icon} {message}", color)

# Setup the logger using our custom formatter.
logger = logging.getLogger("diffformer.dataset")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(PDMLoggerFormatter("[%(levelname)s] %(message)s"))
logger.addHandler(stream_handler)

def download_dataset(dataset: str) -> str:
    """
    Download and return the dataset text.
    For 'shakespeare', the data is downloaded from a URL using requests.
    For 'rocstories' and 'openwebtext', the Hugging Face datasets library is used.
    """
    if dataset == "shakespeare":
        url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
        logger.info("Downloading Shakespeare dataset using requests...")
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    elif dataset == "rocstories":
        from datasets import load_dataset
        logger.info("Loading ROCStories dataset via Hugging Face datasets...")
        ds = load_dataset("rocstories", split="train")
        stories = [" ".join(sample.get(f"sentence{i}", "") for i in range(1, 6)) for sample in ds]
        return "\n\n".join(stories)
    elif dataset == "openwebtext":
        from datasets import load_dataset
        logger.info("Loading OpenWebText dataset via Hugging Face datasets...")
        ds = load_dataset("openwebtext", split="train")
        return "\n\n".join(sample.get("text", "") for sample in ds)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def tokenize_text(text: str) -> torch.Tensor:
    """
    Tokenize text using a pretrained BERT tokenizer.
    """
    logger.info("Tokenizing text using the BERT tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(tokens, dtype=torch.long)

def chars_to_tensor(text: str) -> (torch.Tensor, dict):
    """
    Convert text to character-level tensor and build char2idx mapping.
    """
    logger.info("Converting text to character-level dataset...")
    unique_chars = sorted(set(text))
    char2idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    data = [char2idx[ch] for ch in text]
    return torch.tensor(data, dtype=torch.long), char2idx

def train_val_split(data: torch.Tensor, split_ratio=0.9):
    """
    Split data into training and validation sets.
    """
    logger.info("Splitting data into training and validation sets...")
    n = int(len(data) * split_ratio)
    return data[:n], data[n:]

def save_data(dataset_dir: str, name: str, train_data: torch.Tensor, val_data: torch.Tensor):
    """
    Save train.pt and val.pt under the given name.
    """
    path = os.path.join(os.path.expanduser(dataset_dir), name)
    os.makedirs(path, exist_ok=True)
    torch.save(train_data, os.path.join(path, "train.pt"))
    torch.save(val_data, os.path.join(path, "val.pt"))
    logger.info(f"Saved '{name}' train and val datasets in {path}")

def save_mapping(dataset_dir: str, name: str, mapping: dict):
    """
    Save the character-to-index mapping as char2idx.json.
    """
    path = os.path.join(os.path.expanduser(dataset_dir), name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "char2idx.json"), 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved character mapping in {path}/char2idx.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, tokenize, and split a dataset for DiffFormer training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', type=str, default='shakespeare',
        choices=['shakespeare', 'rocstories', 'openwebtext'],
        help="Dataset to process."
    )
    parser.add_argument(
        '--dataset-dir', type=str, default="~/.diffformer/datasets",
        help="Directory to save processed datasets."
    )
    parser.add_argument(
        '--char-level', action='store_true',
        help="Also generate character-level datasets and mapping."
    )
    args = parser.parse_args()

    logger.info(f"Processing dataset: {args.dataset}")
    text = download_dataset(args.dataset)

    # Token-level
    token_data = tokenize_text(text)
    t_train, t_val = train_val_split(token_data)
    save_data(args.dataset_dir, args.dataset + "_token", t_train, t_val)

    # Character-level
    if args.char_level:
        char_data, mapping = chars_to_tensor(text)
        c_train, c_val = train_val_split(char_data)
        save_data(args.dataset_dir, args.dataset + "_char", c_train, c_val)
        save_mapping(args.dataset_dir, args.dataset + "_char", mapping)

    logger.info("All complete! 🎉")
