import hashlib
import io
import logging
import os
from typing import Literal, Optional, Union

import grain.python as grain
from datasets import Dataset as HfDataset
from datasets import IterableDataset, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

from src.trainer.base_arguments import BaseTrainingArgs


def get_dataset(
    hub_url: str,
    subset: Optional[str],
    *,
    image_column: list[str],
    split: str,
    num_samples: Union[int, Literal["all"]] = "all",
    token: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: str = "./.dataset_cache",
    trust_remote_code: bool = False,
):
    # Create a unique cache key based on dataset parameters
    cache_key_parts = [
        hub_url,
        str(subset),
        split,
        str(num_samples),
        "_".join(image_column),
    ]
    cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()

    # Create cache directories
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"data_{cache_key}")

    # Try to load data from cache
    if use_cache and os.path.exists(cache_path):
        try:
            print(f"Loading cached dataset from {cache_path}")
            return load_from_disk(cache_path)
        except Exception as e:
            print(f"Failed to load cache: {e}. Re-downloading data.")

    # Download data if not cached
    data_stream: Optional[IterableDataset] = None

    if subset is not None:
        data_stream = load_dataset(
            hub_url,
            subset,
            split=split,
            streaming=True if num_samples != "all" else False,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    else:
        data_stream = load_dataset(
            hub_url,
            split=split,
            streaming=True if num_samples != "all" else False,
            token=token,
            trust_remote_code=trust_remote_code,
        )

    data_points = []
    for data_point in tqdm(data_stream, desc=f"Loading the {split} data"):
        data_points.append(data_point)
        if num_samples != "all" and len(data_points) >= num_samples:
            break

    dataset = HfDataset.from_list(data_points)

    # Cache the data
    if use_cache:
        try:
            print(f"Caching dataset to {cache_path}")
            dataset.save_to_disk(cache_path)
        except Exception as e:
            print(f"Failed to cache data: {e}")

    return dataset


def load_image_from_bytes(image_data, is_rgb: bool = False):
    """Convert bytes data to PIL Image"""
    if isinstance(image_data, dict) and "bytes" in image_data:
        # If image_data is a dict with 'bytes' key
        image_bytes = image_data["bytes"]
    else:
        # If image_data is directly bytes
        image_bytes = image_data

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB") if is_rgb else image


class HubDataSource(grain.RandomAccessDataSource):
    def __init__(self, dataset: HfDataset) -> None:
        self._dataset = dataset

    def __getitem__(self, record_key):
        return self._dataset[record_key]

    def __len__(self) -> int:
        return len(self._dataset)


def create_dataloaders(
    logger: logging.Logger,
    args: BaseTrainingArgs,
    target_columns: list[str],
    train_transforms: list[grain.MapTransform],
    eval_transforms: list[grain.MapTransform],
):
    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_data: HfDataset = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        image_column=target_columns,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    # Only keep the image column
    train_data = train_data.select_columns(target_columns)
    train_data.set_format("numpy", columns=target_columns)
    train_source = HubDataSource(train_data)

    train_sampler = grain.IndexSampler(
        len(train_source),
        shuffle=True,
        seed=args.seed,
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )

    train_loader = grain.DataLoader(
        data_source=train_source,
        sampler=train_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=train_transforms,
    )

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_data: HfDataset = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        image_column=target_columns,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    logger.info(f"Evaluation dataset loaded with {len(eval_data)} samples")

    # Only keep the image column
    eval_data = eval_data.select_columns(target_columns)
    eval_data.set_format("numpy", columns=target_columns)
    eval_source = HubDataSource(eval_data)

    logger.info(f"Evaluation data source created with {len(eval_source)} samples")

    eval_sampler = grain.IndexSampler(
        len(eval_source),
        shuffle=False,
        seed=args.seed,
        shard_options=grain.NoSharding(),
        num_epochs=1,
    )

    eval_loader = grain.DataLoader(
        data_source=eval_source,
        sampler=eval_sampler,
        worker_count=4,
        worker_buffer_size=2,
        operations=eval_transforms,
    )

    return train_loader, eval_loader
