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


class HausaLMConfig(PretrainedConfig):
    model_type = "xlstm"

    def __init__(
        self,
        xlstm_config: Optional[xLSTMLMModelConfig] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if xlstm_config is None:
            xlstm_config = xLSTMLMModelConfig()

        self.xlstm_config = xlstm_config

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()

        # Making sure that 'xlstm_config' is serialized
        output["xlstm_config"] = asdict(self.xlstm_config)
        return output

    @staticmethod
    def from_yaml(file_path: str):
        with open(file_path, "r") as file:
            config_dict = yaml.safe_load(file)

        return HausaLMConfig.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        xlstm_config_dict: dict[str, Any] = config_dict.pop("xlstm_config")
        xlstm_config = cls.parse_xlstm_config_dict(xlstm_config_dict)

        return cls(xlstm_config=xlstm_config, **config_dict)

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
