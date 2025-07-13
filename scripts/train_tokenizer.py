import hydra
from pathlib import Path
from typing import Literal, Optional, Any

from datasets import Dataset as HfDataset
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer


def train_tokenizer(
    base_tokenizer_name: str,
    dataset_url: str,
    text_column: str,
    split: str,
    samples: int | Literal["all"] = "all",
    subset: Optional[str] = None,
    vocab_size: Optional[int] = None,
    output_dir: str = "./tokenizer",
    batch_size: int = 1000,
    push_to_hub: bool = False,
    model_id: Optional[str] = None,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
    filter_rules: Optional[list[Any]] = None,
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
            streaming=True,
            token=token,
        )
    else:
        dataset = load_dataset(
            dataset_url,
            split=split,
            trust_remote_code=trust_remote_code,
            streaming=True,
            token=token,
        )

    buffer = []
    for el in tqdm(dataset, desc="Loading dataset"):
        if filter_rules is not None:
            for rule in filter_rules:
                if not rule.as_predicate()(el):
                    break
            else:
                buffer.append(el)
        else:
            buffer.append(el)
            
        if samples != "all" and len(buffer) >= samples:
            break

    train_data = HfDataset.from_list(buffer)

    def batch_iterator():
        """Returns batches of texts from the dataset."""
        for i in range(0, len(train_data), batch_size):
            yield train_data[i : i + batch_size][text_column]

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


@hydra.main(config_path="../configs", config_name="tokenizer_config", version_base="1.2")
def main(cfg: DictConfig):
    try:
        print("Starting tokenizer training...")
        print("Parsing config dict")
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        print("Config dict created successfully")

        # Convert filter_rules to BinaryFilterRule objects if they exist
        filter_rules = None
        if config_dict.get("filter_rules"):
            filter_rules = [
                BinaryFilterRule(**rule_dict) 
                for rule_dict in config_dict["filter_rules"]
            ]

        print("Config loaded, starting tokenizer training...")

        train_tokenizer(
            base_tokenizer_name=config_dict["base_tokenizer"],
            dataset_url=config_dict["dataset_url"],
            text_column=config_dict["text_column"],
            split=config_dict["split"],
            subset=config_dict.get("subset"),
            samples=config_dict["samples"],
            vocab_size=config_dict.get("vocab_size"),
            batch_size=config_dict["batch_size"],
            model_id=config_dict.get("model_id"),
            output_dir=config_dict["output_dir"],
            push_to_hub=config_dict["push_to_hub"],
            trust_remote_code=config_dict["trust_remote_code"],
            token=config_dict.get("hub_token"),
            filter_rules=filter_rules,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
