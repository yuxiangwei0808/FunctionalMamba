import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from ..ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
from ..ops.triton.selective_state_update import selective_state_update
from ..ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn


class MambaTemp(nn.Module):
    def __init__(
        self,
        d_model,
        d_s,
        d_t,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        cat_out=False,  # cat output and apply linear
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_t = d_t
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        
        self.d_inner = int(self.expand * self.d_model)
        self.d_inner_s = self.d_inner * self.d_t
        self.d_inner_t = self.d_inner * self.d_s
        
        self.dt_rank = math.ceil(self.d_model * self.d_t / 16) if dt_rank == "auto" else dt_rank
        self.dt_rank_t = math.ceil(self.d_model * self.d_s / 32) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)


        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner_s,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner_s,
            out_channels=self.d_inner_s,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner_s,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner_s, device=device))  # Keep in fp32
        self.D._no_weight_decay = True


        A_t = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner_t,
        ).contiguous()
        A_t_log = torch.log(A_t)  # Keep A_b_log in fp32
        self.A_t_log = nn.Parameter(A_t_log)
        self.A_t_log._no_weight_decay = True

        self.conv1d_t = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner_t,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_t = nn.Linear(
            self.d_inner_t, self.dt_rank_t + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_t = nn.Linear(self.dt_rank_t, self.d_inner_t, bias=True, **factory_kwargs)
        self.D_t = nn.Parameter(torch.ones(self.d_inner_t, device=device))  # Keep in fp32
        self.D_t._no_weight_decay = True


        A_t_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner_t,
        ).contiguous()
        A_t_b_log = torch.log(A_t_b)  # Keep A_b_log in fp32
        self.A_t_b_log = nn.Parameter(A_t_b_log)
        self.A_t_b_log._no_weight_decay = True

        self.conv1d_t_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner_t,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_t_b = nn.Linear(
            self.d_inner_t, self.dt_rank_t + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_t_b = nn.Linear(self.dt_rank_t, self.d_inner_t, bias=True, **factory_kwargs)
        self.D_t_b = nn.Parameter(torch.ones(self.d_inner_t, device=device))  # Keep in fp32
        self.D_t_b._no_weight_decay = True


        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        """
        hidden_states: (B, T, L, D)
        """
        batch, timelen, seqlen, dim = hidden_states.shape

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b t l d -> d (b t l)"), "d (b t l) -> b d t l",
            l=seqlen, t=timelen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())
        if self.use_fast_path:
            out = mamba_inner_fn_no_out_proj(
                rearrange(xz, "b d t l -> b (d t) l"),
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

