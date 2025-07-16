from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers.utils.import_utils import is_causal_conv1d_available

def permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k

def undo_permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually undo the reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


import torch
from torch import nn

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
    
from transformers.utils import logging
from collections import defaultdict
logger = logging.get_logger(__name__)

class TTTCache:
    """
    TTTCache is a data structure that holds the last hidden states and gradients for the TTT layer.

    Arguments:
        model: TTTModel
        batch_size: int

    Attributes:
        seqlen_offset: int
        mini_batch_size: int
        params_dict: Dict[str, Dict[int, torch.Tensor]]  *_states, *_grad -> # layer_idx -> [batch_size, ...]
        conv_states_dic: Dict[str, Dict[int, torch.Tensor]]  *_states -> # layer_idx -> [batch_size, ...]

    """

    def __init__(self, model, batch_size: int, mini_batch_size: int):
        self.seqlen_offset = 0
        self.mini_batch_size = mini_batch_size

        self.ttt_params_dict = defaultdict(dict)
        self.ttt_param_names = ["W1", "b1"]
        dict_weight = {"W1": model.W1, "b1": model.b1}

        self.conv_states_dic = defaultdict(dict)
        # logger.info(f"Creating cache of size: {batch_size}")
        
        for name in self.ttt_param_names:
            weight = dict_weight[name].unsqueeze(0).repeat(batch_size, 1, 1, 1)
            self.ttt_params_dict[f"{name}_states"] = weight
            # for decoding, we need to store the gradients as well
            self.ttt_params_dict[f"{name}_grad"] = torch.zeros_like(weight)

    def update(self, py_tree, seq_len):
        if seq_len % self.mini_batch_size == 0:
            # copy last mini-batch states, clear gradients
            for name in self.ttt_param_names:
                self.ttt_params_dict[f"{name}_states"].copy_(py_tree[f"{name}_states"])
                self.ttt_params_dict[f"{name}_grad"].zero_()
        elif seq_len < self.mini_batch_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.mini_batch_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.mini_batch_size == 0:
                # copy last mini-batch states, clear gradients
                for name in self.ttt_param_names:
                    self.ttt_params_dict[f"{name}_states"].copy_(py_tree[f"{name}_states"])
                    self.ttt_params_dict[f"{name}_grad"].zero_()
            else:
                # copy gradients for the next update
                for name in self.ttt_param_names:
                    self.ttt_params_dict[f"{name}_grad"].copy_(py_tree[f"{name}_grad"])
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")

    def ttt_params_to_dict(self):
        return {name: self.ttt_params_dict[name] for name in self.ttt_params_dict}

    def reset_called(self):
        self.called = False  # Method to reset the called flag after checking

    def is_called(self):
        return self.called  # Method to check if the cache was used

from torch import nn
import torch

class TTTBase(nn.Module):
    def __init__(self, num_heads, hidden_size, output_size, mini_batch_size, rope_theta):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.mini_batch_size = mini_batch_size
        self.rope_theta = rope_theta
        self.head_dim = hidden_size // num_heads
        self.output_size = output_size

        self.ttt_base_lr = 1.0
        token_idx = 1.0 / torch.arange(1, self.mini_batch_size + 1)
        self.register_buffer("token_idx", token_idx)
        self.learnable_token_idx = nn.Parameter(torch.zeros((self.mini_batch_size, )))
        self.post_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self._init_qkvo_proj()
        self._init_rope()
        self._init_ttt_ln()
        self._init_ttt_lr_gate()



    def _init_qkvo_proj(self):
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.output_size, bias=False)

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.rope_theta,
        )

    def _init_ttt_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data

        self.ttt_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        self.ttt_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))
    
    def _init_ttt_lr_gate(self):
        linear_weight_data = nn.Linear(self.hidden_size, 1, bias=True).weight.data
        linear_bias_data = nn.Linear(self.hidden_size, 1, bias=True).bias.data
        
        # [num_heads, hidden_size, 1]
        self.learnable_ttt_lr_weight = nn.Parameter(
            torch.stack(
                [torch.normal(0, 0.02, size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim = 0
            )
        )

        self.learnable_ttt_lr_bias = nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim = 0
            )
        )

    def ttt(
        self,
        inputs,
        mini_batch_size,
        lats_mini_batch_params_dict,
        cache_params,
    ):
        raise NotImplementedError("ttt method must be implemented in TTTBase subclasses.")
    
    def get_qkv_projections(self, hidden_states):
        XQ, XK, XV = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        )
        return XQ, XK, XV

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        ttt_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_ttt_lr_weight) + self.learnable_ttt_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        ttt_lr = F.sigmoid(ttt_lr)

        # [B, num_heads, num_mini_batch, 1, mini_batch_size]
        ttt_lr = ttt_lr.permute(0, 1, 2, 4, 3)
        ttt_lr_eta = self.ttt_base_lr * ttt_lr / self.head_dim

        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, ttt_lr_eta
    
    def get_ttt_inputs(self, inputs, mini_batch_size, cache_params):
        XQ, XK, XV, X = inputs["XQ"], inputs["XK"], inputs["XV"],inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.hidden_size)
        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        if cache_params is not None :
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0
        token_eta, ttt_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)


        # [B, num_heads, num_mini_batchm mini_batch_size, mini_batch_size]
        eta = token_eta * ttt_lr_eta
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "ttt_lr_eta": ttt_lr_eta,
        }
        return inputs

    def forward(self, hidden_states, position_ids, cache_params):
        B, L = hidden_states.shape[:2]
        num_mini_batch = L // self.mini_batch_size
        reminder_len = L % self.mini_batch_size
        last_mini_batch_params_dict = None

        XQ, XK, XV = self.get_qkv_projections(hidden_states)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        output_hidden_states = []

        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            inputs = self.get_ttt_inputs(inputs, self.mini_batch_size, cache_params)
            output_mod, last_mini_batch_params_dict = self.ttt(inputs, self.mini_batch_size, last_mini_batch_params_dict, cache_params)
            output_hidden_states.append(output_mod)

        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            # output_mod, _ = self.ttt(
            #     self.get_ttt_inputs(inputs, reminder_len, cache_params),
            #     mini_batch_size=reminder_len,
            #     lats_mini_batch_params_dict=last_mini_batch_params_dict,
            #     cache_params=cache_params,
            # )
            inputs = self.get_ttt_inputs(inputs, reminder_len, cache_params)
            output_mod, last_mini_batch_params_dict = self.ttt(inputs, reminder_len, last_mini_batch_params_dict, cache_params)
            output_hidden_states.append(output_mod)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        output_hidden_states = self.o_proj(hidden_states)

        return output_hidden_states
        
def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out

def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y


def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z

class TTTLinear(TTTBase):
    def __init__(self, num_heads, hidden_size, output_size, mini_batch_size, rope_theta):
        super().__init__(num_heads, hidden_size, output_size, mini_batch_size, rope_theta)
        # TTT model initialization for TTT-Linear
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params,
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # in this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict()

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim 64
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"]

            # [B, nh, K,f], K=mini_batch_size  2, 32, 8, 64, 64
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B, nh, K, 1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B,nh,K,f]
            grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias)

            if use_dual_form:
                # [B,nh,K,K]
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim] ->
        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] 2, 32, 8, 64, 64
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            0
        )
        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dict, L)

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] ->
        # [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.hidden_size)
        
        return XQW_batch, batch_params_dict


def gelu_bwd(x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff

class TTTMLP(TTTBase):
    def __init__(self, num_heads, hidden_size, output_size, mini_batch_size, rope_theta):
        super().__init__(num_heads, hidden_size, output_size, mini_batch_size, rope_theta)
        # TTT model initialization for TTT-MLP
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, 4 * self.head_dim)))
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, 4 * self.head_dim))
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, 4 * self.head_dim, self.head_dim)))
        self.b2 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def ttt(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params: Optional[TTTCache] = None,
    ):
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # in this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.ttt_params_to_dict()

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype
        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0

        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, 4f]
            W1_init = params_dict["W1_states"]
            # [B, nh, 1, 4f]
            b1_init = params_dict["b1_states"]
            # [B, nh, 4f, f]
            W2_init = params_dict["W2_states"]
            # [B, nh, 1, f]
            b2_init = params_dict["b2_states"]

            # [B,nh,K,f]
            XQ_mini_batch = inputs["XQ"]
            XV_mini_batch = inputs["XV"]
            XK_mini_batch = inputs["XK"]
            # [B,nh,K,1]
            eta_mini_batch = inputs["eta"]
            token_eta_mini_batch = inputs["token_eta"]
            ttt_lr_eta_mini_batch = inputs["ttt_lr_eta"]

            X1 = XK_mini_batch
            # [B,nh,K,f] @ [B,nh,f,4f] -> [B,nh,K,4f]
            Z1 = X1 @ W1_init + b1_init
            X2 = F.gelu(Z1, approximate="tanh")
            # [B,nh,K,4f] @ [B,nh,4f,f] -> [B,nh,K,f]
            Z2 = X2 @ W2_init + b2_init
            reconstruction_target = XV_mini_batch - XK_mini_batch

            ln_weight = self.ttt_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.ttt_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B, nh, K, f]
            grad_l_wrt_Z2 = ln_fused_l2_bwd(Z2, reconstruction_target, ln_weight, ln_bias)
            # [B, nh, K, 4f]
            grad_l_wrt_Z1 = grad_l_wrt_Z2 @ W2_init.transpose(-2, -1) * gelu_bwd(Z1)

            if use_dual_form:
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))  # [B,nh,K,K]
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,4f] -> [B,nh,K,4f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,4f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,4f] + [B,nh,K,4f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")

                # [B,nh,K,K]
                Attn2 = torch.tril(X2_bar @ X2.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,1] * [B,nh,K,f] -> [B,nh,K,f]
                b2_bar = b2_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z2
                # [B,nh,K,f] @ [1,nh,4f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z2_bar = X2_bar @ W2_init - (eta_mini_batch * Attn2) @ grad_l_wrt_Z2 + b2_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,4f] - [B,nh,f,K] @ [B,nh,K,4f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,4f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                # [B,nh,4f,f] - [B,nh,4f,K] @ [B,nh,K,f]
                W2_last = W2_init - (last_eta_mini_batch * X2).transpose(-1, -2) @ grad_l_wrt_Z2
                # [B,nh,1,f]
                b2_last = b2_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z2, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
                grad_W2_last = torch.zeros_like(W2_last)
                grad_b2_last = torch.zeros_like(b2_last)

            else:
                ttt_lr_eta_mini_batch = torch.broadcast_to(
                    ttt_lr_eta_mini_batch,
                    (
                        *ttt_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, 4f, f]
                grad_W2 = torch.einsum("bhki,bhkj->bhkij", X2, grad_l_wrt_Z2)
                grad_W2 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W2)
                grad_W2 = grad_W2 + params_dict["W2_grad"].unsqueeze(2)
                # [B, nh, K, f]
                grad_b2 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z2)
                grad_b2 = grad_b2 + params_dict["b2_grad"]

                # [B, nh, K, f, 4f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(ttt_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2)
                # [B, nh, K, 4f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(ttt_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"]

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch
                W2_bar = W2_init.unsqueeze(2) - grad_W2 * token_eta_mini_batch.unsqueeze(-1)
                b2_bar = b2_init - grad_b2 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, 4f] -> [B, nh, K, 4f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar
                X2_bar = F.gelu(Z1_bar, approximate="tanh")
                Z2_bar = (X2_bar.unsqueeze(3) @ W2_bar).squeeze(3) + b2_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                W2_last = W2_bar[:, :, -1]
                b2_last = b2_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]
                grad_W2_last = grad_W2[:, :, -1]
                grad_b2_last = grad_b2[:, :, -1:]

            Z2_bar = ln_fwd(Z2_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z2_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W2_states": W2_last,
                "b2_states": b2_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
                "W2_grad": grad_W2_last,
                "b2_grad": grad_b2_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "W2_states": torch.tile(self.W2.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b2_states": torch.tile(self.b2.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))
            init_params_dict.update(W2_grad=torch.zeros_like(init_params_dict["W2_states"]))
            init_params_dict.update(b2_grad=torch.zeros_like(init_params_dict["b2_states"]))
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)  # [B,nh,NC,CS,f] -> [NC,B,nh,CS,f]
        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            0,
        )

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dict, self.layer_idx, L)

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.width)
        return XQW_batch, batch_params_dict

if __name__ == "__main__":
    # 创建一个TTTModel的实例
    model = TTTLinear(num_heads=8, hidden_size=512, output_size=128, mini_batch_size=64, rope_theta=10000)
    # model_2 = TTTMLP(num_heads=8, hidden_size=512, output_size=128, mini_batch_size=64, rope_theta=10000)

    # 创建一个输入张量
    input_tensor = torch.randn(32, 196, 512)

    # 创建一个位置ID张量
    position_ids = torch.arange(196).unsqueeze(0).repeat(32, 1)

    # 使用模型进行前向传播
    cache = TTTCache(model, 32, 64)
    output_tensor = model(input_tensor, position_ids, cache)

    # 打印输出张量的形状
    print(output_tensor.shape)