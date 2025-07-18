import typing as tp

from xlstm_jax.blocks.mlstm.block import mLSTMBlockConfig
from xlstm_jax.blocks.mlstm.layer import mLSTMLayerConfig
from xlstm_jax.blocks.slstm.block import sLSTMBlockConfig
from xlstm_jax.blocks.slstm.layer import sLSTMLayerConfig
from xlstm_jax.components.feedforward import FeedForwardConfig
from xlstm_jax.xlstm_lm_model import xLSTMLMModelConfig


def parse_xlstm_config_dict(config_dict: dict[str, tp.Any]):
    # mLSTM block config deserialization
    mlstm_block_dict: dict[str, tp.Any] = config_dict.pop("mlstm_block", None)
    mlstm_block = None
    if mlstm_block_dict:
        mlstm_block = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(**mlstm_block_dict.pop("mlstm")),
            **mlstm_block_dict,
        )

    # sLSTM block config deserialization
    slstm_block_dict: dict[str, tp.Any] = config_dict.pop("slstm_block", None)
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
