import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import numpy as np

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights

import math
from einops import rearrange, einsum
from collections import namedtuple

from mamba_ssm.modules.mlp import GatedMLP
from .rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb, RotaryEmbeddingST, apply_rotary_emb_st
from ..mamba.mamba_ssm.modules.mamba_simple import Mamba
from ..mamba2.mamba_ssm.modules.mamba2 import Mamba2
from ..mamba.mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from .patch_embed import *

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Block(nn.Module):
    def __init__(
        self, dim, dim_t, mixer_cls, norm_cls=partial(nn.LayerNorm, eps=1e-6), fused_add_norm=False, residual_in_fp32=False,drop_path=0., patch=None,):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim, dim_t) if patch == None else nn.ModuleList([mixer_cls(dim, dim_t[i]) for i in range(len(patch))])
        # self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.patch = patch
        self.hw = dim_t

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        if self.patch != None:
            # do swin and window here
            # example [(None,None), ('H', 1), ('W', 2)]
            H = W = int(hidden_states.shape[1] ** 0.5)
            hidden_states = rearrange(hidden_states, 'b (h w) c -> b h w c', h=H)
            assert len(self.patch) == len(self.mixer)
            for i, (dim, size) in enumerate(self.patch):
                if dim != None:
                    hidden_states = component_roll(hidden_states, size, dim)            
                    hidden_states = component_window_partition(hidden_states, size, dim)
                _, HH, WW, _ = hidden_states.shape
                hidden_states = rearrange(hidden_states, 'b h w c -> b (h w) c')
                hidden_states = self.mixer[i](hidden_states, inference_params=inference_params)
                hidden_states = rearrange(hidden_states, 'b (h w) c -> b h w c', h=HH)
                if dim != None:
                    hidden_states = component_window_reverse(hidden_states, size, H, W, dim)
                    hidden_states = component_roll(hidden_states, size, dim, rolling=False)
            hidden_states = rearrange(hidden_states, 'b h w c -> b (h w) c')
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    
class BlockV2(nn.Module):
    def __init__(
        self, dim, dim_t, mixer_cls, norm_cls=partial(nn.LayerNorm, eps=1e-6), fused_add_norm=False, drop = 0.0, residual_in_fp32=False, drop_path=0., mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim, dim_t)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_cls(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop, channels_first=False)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.mixer(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x


def create_block(
    d_model,
    d_t=1378 // 4,
    version='v1',
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    bimamba_type="none",
    mamba_type='v1',
    patch=None,
    **kwargs,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if mamba_type == 'v1':
        mixer_cls = partial(Mamba, bimamba_type=bimamba_type, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs, **kwargs)
    else:
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs, **kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    if patch != None:
        d_t = [d_t // size if size != None else d_t for _, size in patch ]
    
    if version == 'v1':
        block = Block(
            d_model,
            d_t,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            patch=patch,
        )
        block.layer_idx = layer_idx
    elif version == 'v2':
        block = BlockV2(
            d_model,
            d_t,
            mixer_cls,
            norm_cls=norm_cls,
            drop_path=drop_path,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32
        )
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class FunctionalMamba(nn.Module):
    def __init__(self,
                 img_size=64, 
                 patch_size=4, 
                 depth=24, 
                 embed_dim=192,
                 in_chans=246, 
                 num_classes=1,
                 ssm_cfg=None, 
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_epsilon: float = 1e-5, 
                 rms_norm: bool = False, 
                 initializer_cfg=None,
                 fused_add_norm=False,
                 residual_in_fp32=False,
                 device=None,
                 dtype=None,
                 ft_seq_len=None,
                 pt_hw_seq_len=14,
                 final_pool_type='none',
                 if_abs_pos_embed=False,
                 if_rope=False,
                 if_rope_residual=False,
                 bimamba_type="none",
                 block_type="v1",
                 if_cls_token=False,
                 mamba_type='v1',
                 input_shape='4d',
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs) 
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.input_shape = input_shape

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        if input_shape == '4d':
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches
        else:
            # self.patch_embed = nn.Linear(1, embed_dim, bias=True)
            self.patch_embed = nn.Conv1d(in_chans, embed_dim, stride=4, kernel_size=4, bias=True)
            num_patches = 1378 // 4

        if if_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        if if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dim), requires_grad=True)
            self.pos_drop = nn.Dropout(p=drop_rate)

        if if_rope:
            if input_shape == '4d':
                half_head_dim = embed_dim // 2
                hw_seq_len = img_size // patch_size
                self.rope = VisionRotaryEmbeddingFast(
                    dim=half_head_dim,
                    pt_seq_len=pt_hw_seq_len,
                    ft_seq_len=hw_seq_len
                )
            else:
                self.rope = RotaryEmbedding(dim=256)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        # TODO: release this comment
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # transformer blocks
        self.block_type = block_type
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    num_patches,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba_type=bimamba_type,
                    version=block_type,
                    drop_path=inter_dpr[i],
                    mamba_type=mamba_type,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )

        self.pre_logits = nn.Identity()

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        if x.dim() == 4:
            x = self.patch_embed(x)
        else:  # B T L
            x = self.patch_embed(x)
            x = x.permute(0, 2, 1)

        if self.if_cls_token:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)
        
        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            # rope about
            if self.if_rope:
                if self.input_shape == '4d':
                    hidden_states = self.rope(hidden_states)
                else:
                    hidden_states = self.rope.rotate_queries_or_keys(hidden_states)
                if residual is not None and self.if_rope_residual:
                    if self.input_shape == '4d':
                        residual = self.rope(residual)
                    else:
                        residual = self.rope.rotate_queries_or_keys(residual)
            if self.block_type == 'v1':    
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
            elif self.block_type == 'v2':
                hidden_states = layer(hidden_states)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            return hidden_states[:, 0, :]

        if self.final_pool_type == 'none':
            return hidden_states[:, -1, :]
        elif self.final_pool_type == 'mean':
            return hidden_states.mean(dim=1)
        elif self.final_pool_type == 'max':
            return hidden_states.max(dim=1)
        elif self.final_pool_type == 'all':
            return hidden_states
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False, inference_params=None):
        # x (B C H W) FNC
        x = self.forward_features(x, inference_params)
        if return_features:
            return x
        x = self.head(x)
        return x


class FunctionalMambaMultiLayerST(nn.Module):
    def __init__(self,
                img_size=64,
                num_frame=100,
                patch_size=4,
                patch_size_t=[8, 4, 4, 2],
                depths=[3, 3, 8, 3], 
                embed_dims=[16, 32, 64, 128],
                in_chans=246, 
                num_classes=1,
                ssm_cfg=None, 
                drop_rate=0.,
                drop_path_rate=0.1,
                norm_epsilon: float = 1e-5, 
                rms_norm: bool = False, 
                initializer_cfg=None,
                fused_add_norm=False,
                residual_in_fp32=False,
                ft_seq_len=None,
                pt_hw_seq_len=14,
                final_pool_type='none',
                if_abs_pos_embed=False,
                if_rope=True,
                if_rope_residual=True,
                bimamba_type="none",
                block_type="v1",
                if_cls_token=False,
                mamba_type='v1',
                input_shape='4d',
                window_size=1,
                window_size_t=1,
                embeder=PatchEmbedSpatioTemporal,
                emb_axis=None,
                pretrain=False,
                return_states=False,
                **kwargs):
        super().__init__()
        patch_size = [patch_size for _ in range(len(depths))] if not isinstance(patch_size, list) else patch_size
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.if_cls_token = if_cls_token
        self.num_tokens = 1 if if_cls_token else 0
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.window_size = []
        self.window_size_t = []
        self.emb_axis = emb_axis
        self.pretrain = pretrain
        self.return_states = return_states  # return B and C matrix from selective scan
        self.states = []
        row_wise = emb_axis

        self.num_classes = num_classes
        self.embed_dims = embed_dims  # num_features for consistency with other models
        self.num_features = embed_dims[-1]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # import ipdb;ipdb.set_trace()
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.block_type = block_type
        
        assert len(depths) == len(embed_dims)
        self.ropes = nn.ModuleList([])
        self.patch_embeds = nn.ModuleList([])
        self.stages_s = nn.ModuleList([])
        self.stages_t = nn.ModuleList([])
        self.pos_embeds = nn.ParameterList([]) if if_abs_pos_embed else None
        
        h, w = img_size, img_size
        if input_shape == '4d':
            for stage_idx, depth in enumerate(depths):
                self.patch_embeds.append(embeder(
                    h, num_frame, patch_size=patch_size[stage_idx], patch_size_t=patch_size_t[stage_idx], in_chans=in_chans, embed_dim=embed_dims[stage_idx]
                ))
                row_wise = 1 - row_wise if row_wise != None else None

                h = h // self.patch_size[stage_idx] if row_wise == 1 or row_wise == None else h
                w = w // self.patch_size[stage_idx] if row_wise == 0 or row_wise == None else w
                num_frame //= patch_size_t[stage_idx]

                if if_rope:
                    self.ropes.append(
                        RotaryEmbeddingST(dim=embed_dims[stage_idx] // 2, dim_t=embed_dims[stage_idx],
                                          freqs_for='pixel', freqs_for_t='lang', max_freq=embed_dims[stage_idx],
                                          learned_freq_t=False, learned_freq=False)
                    )
                in_chans = embed_dims[stage_idx]

                self.window_size.append(window_size[stage_idx]) if isinstance(window_size, list) else self.window_size.append(window_size)
                self.window_size_t.append(window_size_t[stage_idx]) if isinstance(window_size_t, list) else self.window_size_t.append(window_size_t)

                self.stages_s.append(nn.ModuleList([
                    create_block(
                        embed_dims[stage_idx],
                        h * w if i % 2 == 0 else h * w // self.window_size[stage_idx],
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i + stage_idx,
                        bimamba_type=bimamba_type,
                        version=block_type,
                        drop_path=inter_dpr[i + stage_idx],
                        mamba_type=mamba_type,
                        hw=(h, w) if self.window_size[-1] == 1 or i % 2 == 0 else (h // self.window_size[-1], w),
                        # patch=[(None, None), ('H', window_size[stage_idx])] if i % 2 == 0 else [(None, None), ('W', window_size[stage_idx])],
                    )
                    for i in range(depth)
                ]))
                self.stages_t.append(nn.ModuleList(
                [
                    create_block(
                        embed_dims[stage_idx],
                        num_frame if i % 2 == 0 else int(np.ceil(num_frame / self.window_size_t[-1])),
                        ssm_cfg=ssm_cfg,
                        norm_epsilon=norm_epsilon,
                        rms_norm=rms_norm,
                        residual_in_fp32=residual_in_fp32,
                        fused_add_norm=fused_add_norm,
                        layer_idx=i + stage_idx,
                        bimamba_type=[''],
                        version=block_type,
                        drop_path=inter_dpr[i + stage_idx],
                        mamba_type=mamba_type,
                    )
                    for i in range(depth)
                ]
                ))
                if if_abs_pos_embed:
                    self.pos_embeds.append(nn.Parameter(torch.zeros(1, h, w, num_frame, embed_dims[stage_idx]), requires_grad=True))
        else:
            raise NotImplementedError

        if if_cls_token:
            raise NotImplementedError

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
       
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dims[-1], eps=norm_epsilon,
        )

        self.pre_logits = nn.Identity()

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        if if_abs_pos_embed:
            for x in self.pos_embeds:
                trunc_normal_(x, std=.02)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def _forward_stage(self, x, residual, stage_idx, is_rope=True, emb_axis=None, enable_swin=True, enable_swin_t=True, shift_dim='H'):
        x = self.patch_embeds[stage_idx](x)
        residual = self.patch_embeds[stage_idx](residual) if residual is not None else None
        B, H, W, T, C = x.shape

        if self.if_abs_pos_embed:
            x = x + self.pos_embeds[stage_idx]
        
        num_encs = len(self.stages_s[stage_idx])
        shift_sizes = iter([self.window_size[stage_idx] * (i) for i in range(num_encs)])
        shift_sizes_t = iter([self.window_size_t[stage_idx] * (i) for i in range(num_encs)])
        
        do_swin = 1 - enable_swin if enable_swin else False
        do_swin_t = 1 - enable_swin_t if enable_swin_t else False

        for idx, (layer_s, layer_t) in enumerate(zip(self.stages_s[stage_idx], self.stages_t[stage_idx])):
            if self.if_rope and is_rope:
                freqs, freqs_t = self.ropes[stage_idx].get_axial_freqs((H, W), T)

            """Temporal Mamba"""
            x_t = x.mean((1, 2))
            residual_t = residual.mean((1, 2)) if residual != None else None

            x_t = apply_rotary_emb(freqs_t, x_t)
            residual_t = apply_rotary_emb(freqs_t, residual_t) if residual is not None else None

            if do_swin_t:
                shift_size_t = next(shift_sizes_t)
                x_t, padding = patch_roll_time(x_t, shift_size_t, self.window_size_t[stage_idx], T)
                residual_t, padding = patch_roll_time(residual_t, shift_size_t, self.window_size_t[stage_idx], T)
                T = x_t.shape[1]

            x_t, residual_t = layer_t(x_t, residual_t)

            if self.return_states:
                state_time = layer_t.mixer.return_states

            if do_swin_t:
                x_t = reverse_roll_time(x_t, shift_size_t, self.window_size_t[stage_idx], B, padding)
                residual_t = reverse_roll_time(residual_t, shift_size_t, self.window_size_t[stage_idx], B, padding)
                T = x_t.shape[1]

            do_swin_t = 1 - do_swin_t if enable_swin_t else do_swin_t
            
            x_t, residual_t = apply_rotary_emb(freqs_t, x_t, reverse=True), apply_rotary_emb(freqs_t, residual_t, reverse=True)

            """Conn Mamba"""
            x_s = x.mean(-2)
            residual_s = residual.mean(-2) if residual != None else None

            x_s = apply_rotary_emb(freqs, x_s, sym=True)
            residual_s = apply_rotary_emb(freqs, residual_s, sym=True) if residual_s != None else None

            if do_swin:
                shift_size = next(shift_sizes)
                x_s = component_roll(x_s, shift_size=shift_size, roll_dim=shift_dim)
                residual_s = component_roll(residual_s, shift_size=shift_size, roll_dim=shift_dim)
                
                x_s, residual_s = component_window_partition(x_s, window_size=self.window_size[stage_idx], part_dim=shift_dim), component_window_partition(residual_s, window_size=self.window_size[stage_idx], part_dim=shift_dim)
            
            _, HH, WW, _ = x_s.shape
            
            x_s = rearrange(x_s, 'b h w c -> b (h w) c')
            residual_s = rearrange(residual_s, 'b h w c -> b (h w) c') if residual_s != None else None
            
            if self.block_type == 'v1':
                x_s, residual_s = layer_s(
                    x_s, residual_s
                )
            else:
                raise NotImplementedError

            if self.return_states:
                state_space = layer_s.mixer.return_states

            x_s, residual_s = rearrange(x_s, 'b (h w) c -> b h w c', h=HH, w=WW), rearrange(residual_s, 'b (h w) c -> b h w c', h=HH, w=WW)

            if do_swin:
                x_s, residual_s = component_window_reverse(x_s, window_size=self.window_size[stage_idx], H=H, W=W, part_dim=shift_dim), component_window_reverse(residual_s, window_size=self.window_size[stage_idx], H=H, W=W, part_dim=shift_dim)
                x_s = component_roll(x_s, shift_size=shift_size, rolling=False, roll_dim=shift_dim)
                residual_s = component_roll(residual_s, shift_size=shift_size, rolling=False, roll_dim=shift_dim)
            do_swin = 1 - do_swin if enable_swin else do_swin
            x_s, residual_s = apply_rotary_emb(freqs, x_s, sym=True, reverse=True), apply_rotary_emb(freqs, residual_s, sym=True, reverse=True)
            
            # x = x_s.unsqueeze(-2) + x_s.unsqueeze(-2) * torch.sigmoid(rearrange(x_t, 'b t c -> b 1 1 t c'))
            # residual = residual_s.unsqueeze(-2) + residual_s.unsqueeze(-2) * torch.sigmoid(rearrange(residual_t, 'b t c -> b 1 1 t c'))

            x = x_s.unsqueeze(-2) + rearrange(x_t, 'b t c -> b 1 1 t c')
            residual = residual_s.unsqueeze(-2) + rearrange(residual_t, 'b t c -> b 1 1 t c')

        if self.return_states:
            self.states.append((state_time.detach().cpu(), state_space.detach().cpu()))

        return x, residual

    def forward_features(self, x):
        _, _, h, w, t = x.shape
        residual = None
        row_wise = self.emb_axis
        rope = True
        shift_dim = 'H'
        interm_feats = []  # used for pre-training

        for i in range(len(self.stages_s)):
            x, residual = self._forward_stage(x, residual, i, emb_axis=row_wise, is_rope=rope, shift_dim=shift_dim)
            # shift_dim = 'W' if shift_dim == 'H' else 'H'
            row_wise = 1 - row_wise if row_wise != None else None
            t //= self.patch_size_t[i]

            if self.pretrain and i in [3]:
                # interm_feats.append(x.flatten(1, 3).permute(0, 2, 1).unsqueeze(-1))
                interm_feats.append(x.flatten(1))
            
            if self.emb_axis is not None:
                # 1 for row-wise and 0 for column-wise
                (h, w) = (h // self.patch_size, w) if self.emb_axis else (h, w // self.patch_size)
                self.emb_axis = 1 - self.emb_axis
            else:
                h, w = h // self.patch_size[i], w // self.patch_size[i]
            
            if i < len(self.stages_s) - 1:
                x = rearrange(x, 'b h w t c -> b c h w t', h=h, w=w, t=t)
                residual = rearrange(residual, 'b h w t c -> b c h w t', h=h, w=w, t=t)
            else:
                x = rearrange(x, 'b h w t c -> b (h w t) c', h=h, w=w, t=t)
                residual = rearrange(residual, 'b h w t c -> b (h w t) c', h=h, w=w, t=t)

        if self.pretrain:
            return interm_feats

        if not self.fused_add_norm:
            if residual is None:
                residual = x
            else:
                residual = residual + self.drop_path(x)
            x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            x = fused_add_norm_fn(
                self.drop_path(x),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if self.final_pool_type == 'none':
            return x[:, -1, :]
        elif self.final_pool_type == 'mean':
            return x.mean(dim=1)
        elif self.final_pool_type == 'max':
            return x.max(dim=1)
        elif self.final_pool_type == 'all':
            return x
        else:
            raise NotImplementedError

    def forward(self, x, return_features=False):
        # x (B C H W T) FNC
        self.states = []
        x = self.forward_features(x)
        if self.pretrain:
            return x
    
        if return_features:
            return x
        x = self.head(x)
    
        return x


def component_window_partition(x, window_size, part_dim='H'):
    if x == None:
        return None
    if window_size > 1:
        B, H, W, C = x.shape
        assert H % window_size == 0 and W % window_size == 0

        if part_dim == 'HW':
            x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        elif part_dim == 'H':
            # x = rearrange(x, 'b (s h) w c -> (b h) s w c', s=window_size)
            x = rearrange(x, 'b (h s) w c -> (b s) h w c', s=window_size)
        elif part_dim == 'W':
            x = rearrange(x, 'b h (w s) c -> (b s) h w c', s=window_size)
        else:
            raise NotImplementedError
    return x

def component_window_reverse(x, window_size, H, W, part_dim='H'):
    if window_size > 1:
        C = x.shape[-1]
        
        if part_dim == 'HW':
            x = x.view(-1, H // window_size, W // window_size, window_size, window_size, C)
            x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
        elif part_dim == 'H':
            # x = rearrange(x, '(b h) s w c -> b (s h) w c', h=H // window_size)
            x = rearrange(x, '(b s) h w c -> b (h s) w c', s=window_size)
        elif part_dim == 'W':
            x = rearrange(x, '(b s) h w c -> b h (w s) c', s=window_size)
        else:
            raise NotImplementedError
    return x

def component_roll(x, shift_size, roll_dim='H', rolling=True):
    if x == None:
        return None
    if not rolling:  # unroll
        shift_size = -shift_size
    if roll_dim == 'HW':
        x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
    elif roll_dim == 'H':
        x = torch.roll(x, shifts=-shift_size, dims=1)
    elif roll_dim == 'W':
        x = torch.roll(x, shifts=-shift_size, dims=2)
    else:
        raise NotImplementedError
    return x

def patch_roll_time(x, shift_size, window_size, T, padding=None):
    if x == None:
        return None
    x = torch.roll(x, shifts=-shift_size, dims=-2)
    if window_size > 1:
        padding = 0
        if T % window_size != 0:
            padding = (T // window_size + 1) * window_size - T
            x = nn.functional.pad(x, (0, 0, 0, padding))
        # x = rearrange(x_t, 'b (w n) c -> (b n) w c', w=self.window_size_t[stage_idx])  # v1
        x = rearrange(x, 'b (n w) c -> (b w) n c', w=window_size)
    return x, padding

def reverse_roll_time(x, shift_size, window_size, B, padding):
    if x == None:
        return None
    if window_size > 1:
        # x_t = rearrange(x_t, '(b n) w c -> b (w n) c', b=B)
        x = rearrange(x, '(b w) n c -> b (n w) c', b=B)
    
    if padding != 0:
        x = x[:, :-padding, :]
    x = torch.roll(x, shifts=shift_size, dims=-2)
    
    return x