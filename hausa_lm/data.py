from typing import Literal, Optional, Union

from datasets import Dataset as HfDataset
from datasets import IterableDataset, load_dataset
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
):
    data_stream: Optional[IterableDataset] = None

    if subset is not None:
        data_stream = load_dataset(
            hub_url, subset, split=split, streaming=True, token=token
        )
    else:
        data_stream = load_dataset(hub_url, split=split, streaming=True, token=token)

    data_points = []

    for data_point in tqdm(data_stream, desc=f"Loading the {split} data"):
        data_points.append(data_point)
        if n_samples != "all" and len(data_points) >= n_samples:
            break

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

    raw_data = HfDataset.from_list(data_points)
    tokenized_data = raw_data.map(
        tokenize_text,
        batched=True,
        remove_columns=raw_data.column_names,
        desc=f"Tokenizing the {split} data",
    )

    return tokenized_data
