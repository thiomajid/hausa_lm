from dataclasses import dataclass, field
from typing import Any, Optional

from transformers import TrainingArguments


@dataclass
class BaseTrainingArgs(TrainingArguments):
    """
    Arguments pertaining to MoExLSTM training.
    """

    tokenizer: str = field(
        default="HuggingFaceTB/SmolLM2-1.7B",
        metadata={"help": "The tokenizer to use for the model."},
    )

    train_dataset_url: str = field(
        default="roneneldan/TinyStories",
        metadata={"help": "URL to the dataset."},
    )

    eval_dataset_url: Optional[str] = field(
        default=None,
        metadata={"help": "URL to the evaluation dataset."},
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

    use_dataset_cache: bool = field(default=True)
    dataset_cache_dir: str = field(default="./.hf_data_cache")

    monitored_layers: Any = field(
        default="all",
        metadata={"help": "Layers to monitor during training."},
    )

    trust_remote_code: bool = field(default=True)

    mesh_shape: tuple[int, ...] = field(default_factory=lambda: (2, 4))
    axis_names: tuple[str, ...] = field(default_factory=lambda: ("dp", "tp"))

    def __post_init__(self):
        super().__post_init__()

        if self.eval_dataset_url is None:
            self.eval_dataset_url = self.train_dataset_url

        self.mesh_shape = tuple(self.mesh_shape)
        self.axis_names = tuple(self.axis_names)
