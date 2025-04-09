import os
import argparse
import tensorflow as tf
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

def download_dataset(dataset: str) -> str:
    """
    Download and return the dataset text.
    For 'shakespeare', the data is downloaded from a URL.
    For 'rocstories' and 'openwebtext', the datasets library is used.
    """
    if dataset == "shakespeare":
        # Download and read Shakespeare dataset
        path_to_file = tf.keras.utils.get_file('shakespeare.txt', 
                                               'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
        with open(path_to_file, 'rb') as f:
            text = f.read().decode('utf-8')
        return text
    elif dataset == "rocstories":
        # Load the ROCStories dataset from Hugging Face.
        ds = load_dataset("rocstories", split="train")
        stories = []
        for sample in ds:
            # Concatenate sentences for each story; adjust if the field names differ.
            story = " ".join(sample.get(f"sentence{i}", "") for i in range(1, 6))
            stories.append(story.strip())
        text = "\n\n".join(stories)
        return text
    elif dataset == "openwebtext":
        # Load the OpenWebText dataset from Hugging Face.
        # Note: This dataset is huge. Consider using streaming or a smaller subset if needed.
        ds = load_dataset("openwebtext", split="train")
        texts = [sample["text"] for sample in ds if "text" in sample]
        text = "\n\n".join(texts)
        return text
    else:
        raise ValueError("Unknown dataset provided.")

def tokenize_text(text: str) -> torch.Tensor:
    """
    Tokenize text using a pretrained BERT tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    data = torch.tensor(tokenized_text, dtype=torch.long)
    return data

def train_val_split(data: torch.Tensor, split_ratio=0.9):
    """
    Split tokenized data into training and validation sets.
    """
    split_point = int(len(data) * split_ratio)
    train_data = data[:split_point]
    val_data = data[split_point:]
    return train_data, val_data

def save_data(dataset_dir: str, dataset: str, train_data: torch.Tensor, val_data: torch.Tensor):
    """
    Save the train and validation data as .pt files to the specified directory.
    """
    save_path = os.path.join(os.path.expanduser(dataset_dir), dataset)
    os.makedirs(save_path, exist_ok=True)
    train_file = os.path.join(save_path, "train.pt")
    val_file = os.path.join(save_path, "val.pt")
    torch.save(train_data, train_file)
    torch.save(val_data, val_file)
    print(f"Saved training data to {train_file}")
    print(f"Saved validation data to {val_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Download, tokenize, and split a dataset for DiffFormer training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['shakespeare', 'rocstories', 'openwebtext'],
        help="Dataset to process. Options: shakespeare, rocstories, openwebtext"
    )
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default="~/.diffformer/datasets",
        help="Directory to save the processed datasets"
    )
    args = parser.parse_args()

    print(f"Processing dataset: {args.dataset}")
    text = download_dataset(args.dataset)
    data = tokenize_text(text)
    train_data, val_data = train_val_split(data)
    save_data(args.dataset_dir, args.dataset, train_data, val_data)

if __name__ == "__main__":
    main()
