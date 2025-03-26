import hashlib
import os
from typing import Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def get_dataset(
    hub_url: str,
    subset: Optional[str],
    *,
    features: list[str],
    max_seq_length: int,
    tokenizer: AutoTokenizer,
    split: str,
    n_samples: Union[int, Literal["all"]] = "all",
    token: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = "./.dataset_cache",
):
    # Create a unique cache key based on dataset parameters
    cache_key_parts = [
        hub_url,
        str(subset),
        split,
        str(n_samples),
        str(max_seq_length),
        tokenizer.name_or_path,
    ]
    cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()

    # Create cache directories
    os.makedirs(cache_dir, exist_ok=True)
    raw_cache_path = os.path.join(cache_dir, f"raw_{cache_key}")
    tokenized_cache_path = os.path.join(cache_dir, f"tokenized_{cache_key}")

    # Try to load tokenized data from cache
    if use_cache and os.path.exists(tokenized_cache_path):
        try:
            print(f"Loading cached tokenized dataset from {tokenized_cache_path}")
            return load_from_disk(tokenized_cache_path)
        except Exception as e:
            print(f"Failed to load tokenized cache: {e}. Re-processing data.")

    # Try to load raw data from cache
    raw_data = None
    if use_cache and os.path.exists(raw_cache_path):
        try:
            print(f"Loading cached raw dataset from {raw_cache_path}")
            raw_data = load_from_disk(raw_cache_path)
        except Exception as e:
            print(f"Failed to load raw cache: {e}. Re-downloading data.")

    # Download data if not cached
    if raw_data is None:
        data_stream: Optional[IterableDataset] = None

        if subset is not None:
            data_stream = load_dataset(
                hub_url, subset, split=split, streaming=True, token=token
            )
        else:
            data_stream = load_dataset(
                hub_url, split=split, streaming=True, token=token
            )

        data_points = []

        for data_point in tqdm(data_stream, desc=f"Loading the {split} data"):
            data_points.append(data_point)
            if n_samples != "all" and len(data_points) >= n_samples:
                break

        raw_data = HfDataset.from_list(data_points)

        # Cache the raw data
        if use_cache:
            try:
                print(f"Caching raw dataset to {raw_cache_path}")
                raw_data.save_to_disk(raw_cache_path)
            except Exception as e:
                print(f"Failed to cache raw data: {e}")

    def tokenize_text(element):
        encodings = tokenizer(
            element[features[0]],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_length=True,
            return_tensors="pt",
        )
        return encodings

    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
        desc=f"Tokenizing the {split} data",
    )

    # Cache the tokenized data
    if use_cache:
        try:
            print(f"Caching tokenized dataset to {tokenized_cache_path}")
            tokenized_data.save_to_disk(tokenized_cache_path)
        except Exception as e:
            print(f"Failed to cache tokenized data: {e}")

    return tokenized_data
