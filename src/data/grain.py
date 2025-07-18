import logging

import grain.python as grain
from datasets import Dataset as HfDataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from src.data.hf import get_dataset
from src.trainer.base_arguments import BaseTrainingArgs


class HubDataSource(grain.RandomAccessDataSource):
    def __init__(self, dataset: HfDataset) -> None:
        self._dataset = dataset

    def __getitem__(self, record_key):
        return self._dataset[record_key]

    def __len__(self) -> int:
        return len(self._dataset)


class DataCollatatorTransform(grain.MapTransform):
    """
    Applies a collator to a dataset element and converts the specified columns to **JAX** arrays.
    This transform uses a Hugging Face **DataCollatorForLanguageModeling** to process a dataset element,
    then converts the specified columns to **JAX** arrays, removing any other columns.

    Attributes:
        collator: A Hugging Face DataCollatorForLanguageModeling instance.
        target_columns: A list of strings representing the columns to keep and convert to JAX arrays.
    """

    def __init__(
        self,
        collator: DataCollatorForLanguageModeling,
        target_columns: list[str],
    ):
        super().__init__()

        self.collator = collator
        self.target_columns = target_columns

    def map(self, element):
        # if not isinstance(element, list):
        #     element = [element]

        # batch: dict = self.collator.numpy_call(element)
        # result = {}
        # for key in self.target_columns:
        #     if key in batch:
        #         result[key] = jnp.array(batch[key])

        # return result
        return self.collator([element])


def create_dataloaders(
    logger: logging.Logger,
    args: BaseTrainingArgs,
    tokenizer: AutoTokenizer,
    max_seq_length: int,
    train_data_ops: list[grain.MapTransform],
    eval_data_ops: list[grain.MapTransform],
):
    logger.info(
        f"Loading training dataset from {args.train_dataset_url} with {args.train_samples} samples"
    )

    train_data = get_dataset(
        hub_url=args.train_dataset_url,
        subset=args.train_subset,
        features=args.features,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        split=args.train_split,
        num_samples=args.train_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    train_data.set_format("numpy", columns=["input_ids", "attention_mask", "length"])
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
        operations=train_data_ops,
    )

    logger.info(
        f"Loading evaluation dataset from {args.eval_dataset_url} with {args.eval_samples} samples"
    )

    eval_data = get_dataset(
        hub_url=args.eval_dataset_url,
        subset=args.eval_subset,
        features=args.features,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        split=args.eval_split,
        num_samples=args.eval_samples,
        token=args.hub_token,
        trust_remote_code=args.trust_remote_code,
    )

    eval_data.set_format("numpy", columns=["input_ids", "attention_mask", "length"])
    eval_source = HubDataSource(eval_data)

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
        operations=eval_data_ops,
    )

    return train_loader, eval_loader
