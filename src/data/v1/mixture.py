import json
import typing as tp
from dataclasses import asdict, dataclass

from datasets import Dataset, concatenate_datasets
from huggingface_hub import create_repo, repo_exists, upload_file

from .hf import load_and_cache_raw_dataset
from .rules import BinaryFilterRule


@dataclass
class MixDatasetConfig:
    hub_id: str
    split: str
    text_column: str
    samples: tp.Union[int, tp.Literal["all"]] = "all"
    subset: tp.Optional[str] = None
    filter_rules: tp.Optional[list[BinaryFilterRule]] = None


@dataclass
class DataMixConfig:
    upload_hub_id: str
    hub_token: str
    trust_remote_code: bool
    data_mix: list[MixDatasetConfig]
    cache_dir: str = "./.mix-data-cache"
    final_text_column: str = "text"


def create_datamix(config: DataMixConfig):
    datarefs: list[Dataset] = []
    num_datasets = len(config.data_mix)

    for idx, mix_config in enumerate(config.data_mix):
        print("=" * 30)
        print(f"Downloading data {idx + 1} / {num_datasets} from {mix_config.hub_id}")
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
        data = data.remove_columns(unused_columns)

        data = data.add_column("source", [mix_config.hub_id] * data.num_rows)
        data = data.add_column("split", [mix_config.split] * data.num_rows)
        data = data.add_column("column", [mix_config.text_column] * data.num_rows)

        if config.final_text_column not in data.column_names:
            data = data.rename_column(mix_config.text_column, config.final_text_column)

        data = data.shuffle()

        datarefs.append(data)

    return datarefs


def push_datamix_to_hub(config: DataMixConfig):
    datarefs = create_datamix(config)
    final_data = concatenate_datasets(datarefs)
    final_data = final_data.train_test_split(train_size=0.8, seed=42)

    if not repo_exists(repo_id=config.upload_hub_id, token=config.hub_token):
        create_repo(
            repo_id=config.upload_hub_id,
            token=config.hub_token,
            repo_type="dataset",
        )

    final_data.push_to_hub(repo_id=config.upload_hub_id, token=config.hub_token)
    with open("mix_config.json", "w") as f:
        config_dict = asdict(config)
        config_dict.pop("hub_token")

        json.dump(config_dict, f)

    upload_file(
        repo_id=config.upload_hub_id,
        token=config.hub_token,
        path_or_fileobj="mix_config.json",
        path_in_repo="./mix_config.json",
        repo_type="dataset",
    )
