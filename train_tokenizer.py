import argparse
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer


def train_tokenizer(
    base_tokenizer_name: str,
    dataset_url: str,
    text_column: str,
    split: str,
    subset: Optional[str] = None,
    vocab_size: Optional[int] = None,
    output_dir: str = "./tokenizer",
    batch_size: int = 1000,
    push_to_hub: bool = False,
    model_id: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
):
    """Train a new tokenizer based on an existing one using a dataset."""
    print(f"Loading base tokenizer: {base_tokenizer_name}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_tokenizer_name, trust_remote_code=trust_remote_code
    )

    # Use the base tokenizer's vocab size if none is provided
    if vocab_size is None:
        vocab_size = base_tokenizer.vocab_size
        print(f"Using base tokenizer vocab size: {vocab_size}")

    print(f"Loading dataset: {dataset_url}")
    if subset:
        dataset = load_dataset(
            dataset_url,
            subset,
            split=split,
            trust_remote_code=trust_remote_code,
        )
    else:
        dataset = load_dataset(
            dataset_url,
            split=split,
            trust_remote_code=trust_remote_code,
        )

    def batch_iterator():
        """Returns batches of texts from the dataset."""
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size][text_column]

    # Train a new tokenizer from the base tokenizer
    print("Training tokenizer...")
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=vocab_size
    )

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the new tokenizer
    print(f"Saving tokenizer to {output_dir}")
    new_tokenizer.save_pretrained(output_dir)

    # Push to Hub if requested
    if push_to_hub:
        print(f"Pushing tokenizer to the Hub with model ID: {model_id}")
        new_tokenizer.push_to_hub(model_id, token=token)

    print(
        f"New tokenizer trained and saved successfully with vocab size: {new_tokenizer.vocab_size}"
    )
    return new_tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a new tokenizer on a dataset.")
    parser.add_argument(
        "--base_tokenizer",
        type=str,
        required=True,
        help="Name or path of base tokenizer",
    )
    parser.add_argument(
        "--dataset_url",
        type=str,
        required=True,
        help="HuggingFace dataset name or path",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        required=True,
        help="Name of the text column in the dataset",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to use"
    )
    parser.add_argument(
        "--subset", type=str, default=None, help="Dataset subset to use (optional)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=49152, help="Size of the vocabulary"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer",
        help="Directory to save the tokenizer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000, help="Batch size for processing texts"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the tokenizer to the Hub",
    )
    parser.add_argument(
        "--model_id", type=str, default=None, help="Model ID for pushing to the Hub"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code when loading the tokenizer",
    )
    parser.add_argument(
        "--token", type=str, default=None, help="Hugging Face token for pushing to Hub"
    )

    args = parser.parse_args()

    train_tokenizer(
        args.base_tokenizer,
        args.dataset_url,
        args.text_column,
        args.split,
        args.subset,
        args.vocab_size,
        args.output_dir,
        args.batch_size,
        args.push_to_hub,
        args.model_id,
        args.trust_remote_code,
        args.token,
    )
