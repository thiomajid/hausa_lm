import typing as tp
from dataclasses import dataclass

from datasets import Dataset

from .data import load_and_cache_raw_dataset
from .rules import BinaryFilterRule


@dataclass
class MixDatasetConfig:
    hub_id: str
    split: str
    subset: tp.Optional[str]
    samples: tp.Union[int, tp.Literal["all"]]
    filter_rules: tp.Optional[list[BinaryFilterRule]] = None
    text_column: str


@dataclass
class DataMixConfig:
    upload_hub_id: str
    hub_token: str
    trust_remote_code: bool
    cache_dir: str = "./.mix-data-cache"
    final_text_column: str = "text"
    datasets: list[MixDatasetConfig]


def create_datamix(config: DataMixConfig):
    datarefs: list[Dataset] = []
    num_datasets = len(config.datasets)

    for idx, mix_config in enumerate(config.datasets):
        print("=" * 30)
        print(f"Downloading data {idx + 1 / num_datasets} from {mix_config.hub_id}")
        print("=" * 30)

        data: Dataset = load_and_cache_raw_dataset(
            hub_url=mix_config.hub_id,
            subset=mix_config.subset,
            split=mix_config.split,
            num_samples=mix_config.samples,
            token=config.hub_token,
            use_cache=True,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code,
        )

        if mix_config.filter_rules:
            for rule in mix_config.filter_rules:
                data = data.filter(rule.as_predicate())

        columns = data.column_names
        unused_columns = [col for col in columns if col != mix_config.text_column]
        data = data.remove_columns(column_names=unused_columns)
        data = data.rename_column(mix_config.text_column, config.final_text_column)

        source = [mix_config.hub_id] * data.num_rows
        data = data.add_column(name="source", column=source)

        datarefs.append(data)

    return datarefs
