from dataclasses import asdict
from typing import Any, Optional

import yaml
from transformers import PretrainedConfig
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMLMModelConfig,
)

# @dataclass
# class ConstrastiveLSTMVisionConfig:
#     # These arguments are used in the ViLLayer
#     dim: int = 192
#     direction: str
#     expansion: int
#     conv_kind: str = "2d"
#     conv_bias: bool = True
#     seqlens: Optional[int] = None
#     input_shape: tuple = (3, 224, 224)
#     patch_size: int = 16
#     depth: int = 12
#     output_shape: tuple = (1000,)
#     mode: str = "classifier"
#     pooling: str = "bilateral_flatten"
#     drop_path_rate: float = 0.0
#     drop_path_decay: bool = False
#     stride: Optional[int] = None
#     legacy_norm: bool = False
#     conv_kernel_size: int = 3
#     proj_bias: bool = True
#     norm_bias: bool = True
#     init_weights: str = "original"
#     mlstm_config: mLSTMLayerConfig


class HausaLMConfig(PretrainedConfig):
    model_type = "xlstm"

    def __init__(
        self,
        text_config: Optional[xLSTMLMModelConfig] = None,
        # vision_config: Optional[ConstrastiveLSTMVisionConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = xLSTMLMModelConfig()

        self.text_config = text_config
        # self.vision_config = vision_config

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()

        # Making sure that 'text_config' is serialized
        output["text_config"] = asdict(self.text_config)
        return output

    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)

        return HausaLMConfig.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        text_config_dict: dict[str, Any] = config_dict.pop("text_config")
        text_config = cls.parse_xlstm_config_dict(text_config_dict)

        return cls(text_config=text_config, **config_dict)

    @staticmethod
    def parse_xlstm_config_dict(config_dict: dict[str, Any]):
        # mLSTM block config deserialization
        mlstm_block_dict: dict[str, Any] = config_dict.pop("mlstm_block", None)
        mlstm_block = None
        if mlstm_block_dict:
            mlstm_block = mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(**mlstm_block_dict.pop("mlstm")),
                **mlstm_block_dict,
            )

        # sLSTM block config deserialization
        slstm_block_dict: dict[str, Any] = config_dict.pop("slstm_block", None)
        slstm_block = None

        if slstm_block_dict:
            feedforward_dict = slstm_block_dict.pop("feedforward")
            feedforward_config = FeedForwardConfig(**feedforward_dict)
            slstm_block = sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(**slstm_block_dict.pop("slstm")),
                feedforward=feedforward_config,
                **slstm_block_dict,
            )

        # xLSTM stack config deserialization
        xlstm_config = xLSTMLMModelConfig(
            mlstm_block=mlstm_block,
            slstm_block=slstm_block,
            **config_dict,
        )

        return xlstm_config
