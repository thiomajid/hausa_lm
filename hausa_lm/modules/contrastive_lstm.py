from torch import nn

from hausa_lm.config import HausaLMConfig


class ContrastiveLSTM(nn.Module):
    def __init__(self, config: HausaLMConfig):
        super().__init__()
