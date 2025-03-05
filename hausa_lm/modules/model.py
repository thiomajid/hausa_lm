from pathlib import Path
from typing import Optional

import safetensors
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from xlstm import xLSTMBlockStack

from hausa_lm.config import HausaLMConfig


class HausaLMModel(PreTrainedModel):
    config_class = HausaLMConfig

    def __init__(self, config: HausaLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.text_config.vocab_size,
            embedding_dim=config.text_config.embedding_dim,
        )

        self.embedding_dropout = (
            nn.Dropout(config.dropout)
            if config.text_config.add_embedding_dropout
            else nn.Identity()
        )

        self.xlstm = xLSTMBlockStack(config.text_config)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        hidden_states = self.xlstm(hidden_states)

        return hidden_states


class HausaLMForCausalLM(PreTrainedModel):
    config_class = HausaLMConfig

    def __init__(self, config: HausaLMConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)

        self.config = config

        self.model = HausaLMModel(config)
        self.lm_head = nn.Linear(
            config.text_config.embedding_dim,
            config.text_config.vocab_size,
        )

        if config.text_config.tie_weights:
            self.lm_head.weight = self.model.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids)
        logits: torch.Tensor = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            # shape: [batch, seq, vocab] -> [batch * (seq-1), vocab]
            shift_logits = rearrange(
                logits[..., :-1, :].contiguous(), "b s v -> (b s) v"
            )

            # shape: [batch, seq] -> [batch * (seq-1)]
            shift_labels = rearrange(labels[..., 1:].contiguous(), "b s -> (b s)")

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                input=shift_logits,
                target=shift_labels,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )

    @staticmethod
    def from_safetensors(
        hf_repo: str,
        filename: Path | str,
        device: str = "cuda",
    ) -> "HausaLMForCausalLM":
        if isinstance(filename, str):
            filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exist on the disk.")

        config = HausaLMConfig.from_pretrained(hf_repo)
        model = HausaLMForCausalLM(config=config)
        safetensors.torch.load_model(model=model, filename=filename, device=device)
        model = model.to(device)

        return model
