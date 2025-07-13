import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import hydra
from huggingface_hub import upload_file
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from src.data.hf import load_and_cache_raw_dataset


@dataclass
class TrainTokenizerConfig:
    base_tokenizer: str
    dataset_url: str
    text_column: str
    split: str
    samples: int | Literal["all"] = "all"
    subset: Optional[str] = None
    vocab_size: Optional[int] = None
    output_dir: str = "./tokenizer"
    batch_size: int = 1000
    model_id: Optional[str] = None
    trust_remote_code: bool = False
    hub_token: Optional[str] = None


def train_tokenizer(config: TrainTokenizerConfig):
    """Train a new tokenizer based on an existing one using a dataset."""
    print(f"Loading base tokenizer: {config.base_tokenizer}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        config.base_tokenizer, trust_remote_code=config.trust_remote_code
    )

    # Use the base tokenizer's vocab size if none is provided
    if config.vocab_size is None:
        config.vocab_size = base_tokenizer.vocab_size
        print(f"Using base tokenizer vocab size: {config.vocab_size}")

    print(f"Loading dataset: {config.dataset_url}")
    train_data = load_and_cache_raw_dataset(
        hub_url=config.dataset_url,
        subset=config.subset,
        split=config.split,
        num_samples=config.samples,
        token=config.hub_token,
    )

    def batch_iterator():
        """Returns batches of texts from the dataset."""
        for i in range(0, len(train_data), config.batch_size):
            yield train_data[i : i + config.batch_size][config.text_column]

    # Train a new tokenizer from the base tokenizer
    print("Training tokenizer...")
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=config.vocab_size
    )

    # Create output directory if it doesn't exist
    output_path = Path(config.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the new tokenizer
    print(f"Saving tokenizer to {config.output_dir}")
    new_tokenizer.save_pretrained(config.output_dir)

    print(f"Pushing tokenizer to the Hub with model ID: {config.model_id}")
    new_tokenizer.push_to_hub(config.model_id, token=config.hub_token)

    print(
        f"New tokenizer trained and saved successfully with vocab size: {new_tokenizer.vocab_size}"
    )
    return new_tokenizer


@hydra.main(
    config_path="../configs",
    config_name="tokenizer_config",
    version_base="1.2",
)
def main(cfg: DictConfig):
    try:
        print("Starting tokenizer training...")
        print("Parsing config dict")
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config = TrainTokenizerConfig(**config_dict)
        print("Config loaded, starting tokenizer training...")

        train_tokenizer(config)

        with open("train_tokenizer_config.json", "w") as f:
            config_dict = asdict(config)
            config_dict.pop("hub_token")

            json.dump(config_dict, f)

        upload_file(
            repo_id=config.model_id,
            token=config.hub_token,
            path_or_fileobj="train_tokenizer_config.json",
            path_in_repo="./train_tokenizer_config.json",
            repo_type="model",
        )

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
