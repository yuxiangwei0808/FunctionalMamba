from .models_mamba import FunctionalMambaMultiLayerST, _cfg
from .patch_embed import *


def mambaf_multi_st_base_v1_004(**kwargs):
    model = FunctionalMambaMultiLayerST(
        patch_size=2, patch_size_t=[1, 1, 1, 1], embed_dims=[24, 48, 96, 192],
         depths=[2, 2, 6, 2], rms_norm=True, residual_in_fp32=True, window_size=[4, 4, 2, 1], window_size_t=[2, 2, 2, 2],
        fused_add_norm=False, cls_token=False, final_pool_type='mean', if_abs_pos_embed=False,
        if_rope=True, if_rope_residual=True, block_type='v1', **kwargs)
    model.default_cfg = _cfg()
    return model


def mambaf2_base(pretrained=False, **kwargs):
    model = FunctionalMamba(
        patch_size=4, embed_dim=384, depth=12, rms_norm=True, residual_in_fp32=True,
        fused_add_norm=True, cls_token=False, mamba_type='v2', final_pool_type='mean', if_abs_pos_embed=False, 
        if_rope=True, if_rope_residual=True, block_type='v1', **kwargs)
    model.default_cfg = _cfg()
    return model