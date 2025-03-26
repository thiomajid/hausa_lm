from dataclasses import dataclass
from typing import Any, Callable

from hausa_lm.modules.model import HausaLMForCausalLM


@dataclass
class ModelSummary:
    million: float
    billion: float
    total: int
    embedding: float
    embedding_ratio: float
    lm_head: float
    lm_head_ratio: float
    xlstm: float
    xlstm_ratio: float


def model_summary(model: HausaLMForCausalLM):
    embedding = sum(p.numel() for p in model.model.token_embedding.parameters())
    lm_head = sum(p.numel() for p in model.lm_head.parameters())

    xlstm_size = sum(p.numel() for p in model.model.xlstm.parameters())

    one_million = 1e6

    total_parameters = embedding + lm_head + xlstm_size

    def ratio(x):
        return round(x / total_parameters, 2) * 100

    return ModelSummary(
        total=total_parameters,
        billion=total_parameters / 1e9,
        million=total_parameters / one_million,
        embedding=embedding / one_million,
        embedding_ratio=ratio(embedding),
        lm_head=lm_head / one_million,
        lm_head_ratio=ratio(lm_head),
        xlstm=xlstm_size / one_million,
        xlstm_ratio=ratio(xlstm_size),
    )


dataset_filters_registry: list[dict[str, Callable[[Any], bool]]] = {
    "CohereForAI/aya_dataset": [
        lambda x: x["language"] == "Hausa",
    ],
}
"""Set of filtering rules to apply on data points while streaming the dataset."""
