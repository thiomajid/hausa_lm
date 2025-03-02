from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class HausaLMTrainingArgs(TrainingArguments):
    tokenizer: str = field(
        default="HuggingFaceTB/SmolLM2-1.7B",
        metadata={"help": "The tokenizer to use for the model."},
    )

    xlstm_config_path: str = field(
        default="./model_config.yaml",
        metadata={"help": "Path to the xlstm config file."},
    )

    dataset_url: str = field(
        default="roneneldan/TinyStories",
        metadata={"help": "URL to the dataset."},
    )

    train_split: str = field(
        default="train",
        metadata={"help": "The split to use for training."},
    )

    train_subset: str | None = field(
        default=None,
        metadata={"help": "Subset of the training split to use."},
    )

    train_samples: int = field(
        default=10000,
        metadata={"help": "Number of samples to use for training from the dataset."},
    )

    eval_split: str = field(
        default="validation",
        metadata={"help": "The split to use for evaluation."},
    )

    eval_subset: str | None = field(
        default=None,
        metadata={"help": "Subset of the evaluation split to use."},
    )

    eval_samples: int = field(
        default=1000,
        metadata={"help": "Number of samples to use for evaluation from the dataset."},
    )

    features: list[str] = field(
        default_factory=list,
        metadata={"help": "The features to use from the dataset."},
    )
