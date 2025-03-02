from dataclasses import dataclass

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

    return ModelSummary(
        total=total_parameters,
        billion=total_parameters / 1e9,
        million=total_parameters / one_million,
        embedding=embedding / one_million,
        embedding_ratio=embedding / total_parameters,
        lm_head=lm_head / one_million,
        lm_head_ratio=lm_head / total_parameters,
        xlstm=xlstm_size / one_million,
        xlstm_ratio=xlstm_size / total_parameters,
    )
