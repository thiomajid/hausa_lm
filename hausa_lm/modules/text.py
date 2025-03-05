import torch
from torch import nn
from xlstm import xLSTMBlockStack

# from xlstm.components.ln import LayerNorm as xLayerNorm
from hausa_lm.config import HausaLMConfig


class HausaLMTextModel(nn.Module):
    def __init__(self, config: HausaLMConfig) -> None:
        super().__init__()

        self.config = config.text_config

        self.token_embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
        )

        self.embedding_dropout = (
            nn.Dropout(self.config.dropout)
            if self.config.add_embedding_dropout
            else nn.Identity()
        )

        # self.post_encoder_norm = xLayerNorm(ndim=self.config.embedding_dim)
        self.encoder = xLSTMBlockStack(self.config)

        self.head = nn.Linear(self.config.embedding_dim, self.config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
        **kwargs,
    ):
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        encoder_outputs = self.encoder(hidden_states)
        pooled_output = self.head(encoder_outputs)

        return (pooled_output, encoder_outputs)
