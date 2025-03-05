# # Code adapted from https://github.com/NX-AI/vision-lstm/blob/main/vision_lstm/vision_lstm2.py


# import warnings
# from enum import Enum

# import einops
# import torch
# from torch import nn
# from xlstm import mLSTMLayer
# from xlstm.components.ln import LayerNorm as xLayerNorm

# from hausa_lm.config import HausaLMConfig

# from .utils import (
#     DropPath,
#     SequenceConv2d,
#     VitPatchEmbed,
#     VitPosEmbed2d,
#     interpolate_sincos,
#     to_ntuple,
# )


# class SequenceTraversal(Enum):
#     ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
#     ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


# class ViLLayer(nn.Module):
#     def __init__(
#         self,
#         config: HausaLMConfig,
#         dim,
#         direction,
#         expansion=2,
#         qkv_block_size=4,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=4,
#         conv_kind="2d",
#         init_weights="original",
#         seqlens=None,
#         num_blocks=None,
#     ):
#         super().__init__()

#         self.config = config.vision_config
#         self.direction = self.config.direction
#         self.inner_dim = self.config.expansion * self.config.dim

#         self.mlstm = mLSTMLayer(self.config.mlstm_config)

#         # if conv_kind == "causal1d":
#         #     self.mlstm.conv1d = CausalConv1d(
#         #         dim=inner_dim,
#         #         kernel_size=conv_kernel_size,
#         #         bias=conv_bias,
#         #     )

#         if self.config.conv_kind == "2d":
#             assert (
#                 self.config.conv_kernel_size % 2 == 1
#             ), "same output shape as input shape is required -> even kernel sizes not supported"
#             self.mlstm.conv1d = SequenceConv2d(
#                 in_channels=self.inner_dim,
#                 out_channels=self.inner_dim,
#                 kernel_size=self.config.conv_kernel_size,
#                 padding=self.config.conv_kernel_size // 2,
#                 groups=self.inner_dim,
#                 bias=conv_bias,
#                 seqlens=seqlens,
#             )
#         else:
#             raise NotImplementedError

#         self.reset_parameters()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # alternate direction in successive layers
#         if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
#             pass
#         elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])
#         else:
#             raise NotImplementedError

#         x = self.mlstm(x)

#         # reverse alternating flip
#         if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
#             pass
#         elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])
#         else:
#             raise NotImplementedError

#         return x

#     def reset_parameters(self):
#         self.mlstm.reset_parameters()


# class ViLBlock(nn.Module):
#     def __init__(
#         self,
#         dim,
#         direction,
#         drop_path=0.0,
#         conv_kind="2d",
#         conv_kernel_size=3,
#         proj_bias=True,
#         norm_bias=True,
#         seqlens=None,
#         num_blocks=None,
#         init_weights="original",
#     ):
#         super().__init__()
#         self.dim = dim
#         self.direction = direction
#         self.drop_path = drop_path
#         self.conv_kind = conv_kind
#         self.conv_kernel_size = conv_kernel_size
#         self.init_weights = init_weights

#         self.drop_path = DropPath(drop_prob=drop_path)
#         self.norm = xLayerNorm(ndim=dim, weight=True, bias=norm_bias)
#         self.layer = ViLLayer(
#             dim=dim,
#             direction=direction,
#             conv_kind=conv_kind,
#             conv_kernel_size=conv_kernel_size,
#             seqlens=seqlens,
#             norm_bias=norm_bias,
#             proj_bias=proj_bias,
#             num_blocks=num_blocks,
#             init_weights=init_weights,
#         )

#         self.reset_parameters()

#     def _forward_path(self, x):
#         x = self.norm(x)
#         x = self.layer(x)
#         return x

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.drop_path(x, self._forward_path)
#         return x

#     def reset_parameters(self):
#         self.layer.reset_parameters()
#         self.norm.reset_parameters()


# class ViLBlockPair(nn.Module):
#     def __init__(
#         self,
#         dim,
#         drop_path=0.0,
#         conv_kind="2d",
#         conv_kernel_size=3,
#         proj_bias=True,
#         norm_bias=True,
#         seqlens=None,
#         num_blocks=None,
#         init_weights="original",
#     ):
#         super().__init__()

#         self.rowwise_from_top_left = ViLBlock(
#             dim=dim,
#             direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
#             drop_path=drop_path,
#             conv_kind=conv_kind,
#             conv_kernel_size=conv_kernel_size,
#             proj_bias=proj_bias,
#             norm_bias=norm_bias,
#             seqlens=seqlens,
#             num_blocks=num_blocks,
#             init_weights=init_weights,
#         )

#         self.rowwise_from_bot_right = ViLBlock(
#             dim=dim,
#             direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
#             drop_path=drop_path,
#             conv_kind=conv_kind,
#             conv_kernel_size=conv_kernel_size,
#             proj_bias=proj_bias,
#             norm_bias=norm_bias,
#             seqlens=seqlens,
#             num_blocks=num_blocks,
#             init_weights=init_weights,
#         )

#     def forward(self, x):
#         x = self.rowwise_from_top_left(x)
#         x = self.rowwise_from_bot_right(x)
#         return x


# class VisionLSTM2(nn.Module):
#     def __init__(
#         self,
#         dim=192,
#         input_shape=(3, 224, 224),
#         patch_size=16,
#         depth=12,
#         output_shape=(1000,),
#         mode="classifier",
#         pooling="bilateral_flatten",
#         drop_path_rate=0.0,
#         drop_path_decay=False,
#         stride=None,
#         legacy_norm=False,
#         conv_kind="2d",
#         conv_kernel_size=3,
#         proj_bias=True,
#         norm_bias=True,
#         init_weights="original",
#     ):
#         if depth == 24 and dim < 1024:
#             warnings.warn(
#                 "A single VisionLSTM2 block consists of two subblocks (one for each traversal direction). "
#                 "ViL-T, ViL-S and ViL-B therefore use depth=12 instead of depth=24, are you sure you want to use "
#                 "depth=24?"
#             )

#         super().__init__()

#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         ndim = len(self.input_shape) - 1
#         self.patch_size = to_ntuple(patch_size, n=ndim)
#         self.dim = dim
#         self.depth = depth
#         self.stride = stride
#         self.mode = mode
#         self.pooling = pooling
#         self.drop_path_rate = drop_path_rate
#         self.drop_path_decay = drop_path_decay
#         self.conv_kind = conv_kind
#         self.conv_kernel_size = conv_kernel_size
#         self.proj_bias = proj_bias
#         self.norm_bias = norm_bias
#         self.init_weights = init_weights

#         # initialize patch_embed
#         self.patch_embed = VitPatchEmbed(
#             dim=dim,
#             stride=stride,
#             num_channels=self.input_shape[0],
#             resolution=self.input_shape[1:],
#             patch_size=self.patch_size,
#         )

#         # pos embed
#         self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

#         # calculate stochastic depth per block
#         if drop_path_decay and drop_path_rate > 0.0:
#             dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         else:
#             dpr = [drop_path_rate] * depth

#         # merge two blocks into a blockpair to keep depth equal to the depth of transformers
#         # useful to keep layer-wise lr decay implementations consistent with transformers
#         self.blocks = nn.ModuleList(
#             [
#                 ViLBlockPair(
#                     dim=dim,
#                     drop_path=dpr[i],
#                     conv_kind=conv_kind,
#                     seqlens=self.patch_embed.seqlens,
#                     proj_bias=proj_bias,
#                     norm_bias=norm_bias,
#                     num_blocks=depth * 2,
#                     init_weights=init_weights,
#                 )
#                 for i in range(depth)
#             ],
#         )
#         if pooling == "bilateral_flatten" and mode == "classifier":
#             head_dim = dim * 2
#         else:
#             head_dim = dim
#         self.norm = xLayerNorm(dim, bias=norm_bias, eps=1e-6)
#         # LEGACY: not needed but was used during training
#         if legacy_norm:
#             self.legacy_norm = nn.LayerNorm(head_dim)
#         else:
#             self.legacy_norm = nn.Identity()

#         # head
#         if mode == "features":
#             if self.output_shape is not None:
#                 warnings.warn(
#                     f"passed mode=features -> output_shape is ignored ({self.output_shape})"
#                 )
#             self.head = None
#             if self.pooling is None:
#                 self.output_shape = (self.patch_embed.num_patches, dim)
#             elif self.pooling == "to_image":
#                 self.output_shape = (dim, *self.patch_embed.seqlens)
#             else:
#                 warnings.warn(
#                     f"passed invalid pooling -> pooling is ignored ({self.pooling})"
#                 )
#                 self.pooling = None
#         elif mode == "classifier":
#             # linear classification head
#             assert (
#                 self.output_shape is not None and len(self.output_shape) == 1
#             ), "define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
#             self.head = nn.Linear(head_dim, self.output_shape[0])
#             # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
#             nn.init.trunc_normal_(self.head.weight, std=2e-5)
#             nn.init.zeros_(self.head.bias)
#         else:
#             raise NotImplementedError

#     def load_state_dict(self, state_dict, strict=True):
#         # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
#         old_pos_embed = state_dict["pos_embed.embed"]
#         if old_pos_embed.shape != self.pos_embed.embed.shape:
#             state_dict["pos_embed.embed"] = interpolate_sincos(
#                 embed=old_pos_embed, seqlens=self.pos_embed.seqlens
#             )
#         # remove head and adapt layernorm for feature extraction
#         if self.mode == "features":
#             state_dict.pop("head.weight", None)
#             state_dict.pop("head.bias", None)
#             # legacy_norm uses head dim (is doubled for bilateral_concat) -> not usable for feature extraction
#             cur_sd = self.state_dict()
#             state_dict["legacy_norm.weight"] = cur_sd["legacy_norm.weight"]
#             state_dict["legacy_norm.bias"] = cur_sd["legacy_norm.bias"]
#         return super().load_state_dict(state_dict=state_dict, strict=strict)

#     @torch.jit.ignore
#     def no_weight_decay(self):
#         return {"pos_embed.embed"}

#     def forward(self, x):
#         # embed patches
#         x = self.patch_embed(x)
#         # add pos_embed
#         x = self.pos_embed(x)

#         # flatten to 1d
#         x = einops.rearrange(x, "b ... d -> b (...) d")

#         # apply blocks
#         for block in self.blocks:
#             x = block(x)
#         x = self.norm(x)

#         # pool
#         if self.pooling is None:
#             x = self.legacy_norm(x)
#         elif self.pooling == "to_image":
#             x = self.legacy_norm(x)
#             seqlen_h, seqlen_w = self.patch_embed.seqlens
#             x = einops.rearrange(
#                 x,
#                 "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
#                 seqlen_h=seqlen_h,
#                 seqlen_w=seqlen_w,
#             )
#         elif self.pooling == "bilateral_avg":
#             # norm after pooling
#             x = (x[:, 0] + x[:, -1]) / 2
#             x = self.legacy_norm(x)
#         elif self.pooling == "bilateral_flatten":
#             # norm after pooling
#             x = torch.concat([x[:, 0], x[:, -1]], dim=1)
#             x = self.legacy_norm(x)
#         else:
#             raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")

#         # head
#         if self.head is not None:
#             x = self.head(x)

#         return x
