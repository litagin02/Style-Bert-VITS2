import math
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from style_bert_vits2.models import commons


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


@torch.jit.script  # type: ignore
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: list[int]
) -> torch.Tensor:
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
        isflow: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        # if isflow:
        #  cond_layer = torch.nn.Conv1d(256, 2*hidden_channels*n_layers, 1)
        #  self.cond_pre = torch.nn.Conv1d(hidden_channels, 2*hidden_channels, 1)
        #  self.cond_layer = weight_norm(cond_layer, name='weight')
        #  self.gin_channels = 256
        self.cond_layer_idx = self.n_layers
        if "gin_channels" in kwargs:
            self.gin_channels = kwargs["gin_channels"]
            if self.gin_channels != 0:
                self.spk_emb_linear = nn.Linear(self.gin_channels, self.hidden_channels)
                # vits2 says 3rd block, so idx is 2 by default
                self.cond_layer_idx = (
                    kwargs["cond_layer_idx"] if "cond_layer_idx" in kwargs else 2
                )
                # logger.debug(self.gin_channels, self.cond_layer_idx)
                assert self.cond_layer_idx < self.n_layers, (
                    "cond_layer_idx should be less than n_layers"
                )
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    window_size=window_size,
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask

        # Debug: Check input to encoder
        if torch.isnan(x).any():
            print(f"[DEBUG Encoder] NaN in input x")
            print(f"  x dtype: {x.dtype}")

        for i in range(self.n_layers):
            if i == self.cond_layer_idx and g is not None:
                g = self.spk_emb_linear(g.transpose(1, 2))
                assert g is not None
                g = g.transpose(1, 2)
                x = x + g
                x = x * x_mask
                # Debug: After adding conditioning
                if torch.isnan(x).any():
                    print(f"[DEBUG Encoder] NaN in x after adding g in layer {i}")

            # Debug: Before attention
            if torch.isnan(x).any():
                print(f"[DEBUG Encoder] NaN in x before attention in layer {i}")

            y = self.attn_layers[i](x, x, attn_mask)
            # Debug: After attention
            if torch.isnan(y).any():
                print(f"[DEBUG Encoder] NaN in y after attention in layer {i}")
                print(f"  y dtype: {y.dtype}, x dtype: {x.dtype}")

            y = self.drop(y)
            # Debug: After dropout
            if torch.isnan(y).any():
                print(f"[DEBUG Encoder] NaN in y after dropout in layer {i}")

            # Debug: Before norm_layers_1
            x_plus_y = x + y
            if torch.isnan(x_plus_y).any():
                print(
                    f"[DEBUG Encoder] NaN in (x + y) before norm_layers_1 in layer {i}"
                )

            x = self.norm_layers_1[i](x + y)
            # Debug: After norm_layers_1
            if torch.isnan(x).any():
                print(f"[DEBUG Encoder] NaN in x after norm_layers_1 in layer {i}")
                print(f"  x dtype: {x.dtype}")

            y = self.ffn_layers[i](x, x_mask)
            # Debug: After FFN
            if torch.isnan(y).any():
                print(f"[DEBUG Encoder] NaN in y after FFN in layer {i}")

            y = self.drop(y)
            # Debug: After dropout
            if torch.isnan(y).any():
                print(f"[DEBUG Encoder] NaN in y after dropout (FFN) in layer {i}")

            # Debug: Before norm_layers_2
            x_plus_y = x + y
            if torch.isnan(x_plus_y).any():
                print(
                    f"[DEBUG Encoder] NaN in (x + y) before norm_layers_2 in layer {i}"
                )

            x = self.norm_layers_2[i](x + y)
            # Debug: After norm_layers_2
            if torch.isnan(x).any():
                print(f"[DEBUG Encoder] NaN in x after norm_layers_2 in layer {i}")
                print(f"  x dtype: {x.dtype}")

        x = x * x_mask
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        proximal_bias: bool = False,
        proximal_init: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init

        self.drop = nn.Dropout(p_dropout)
        self.self_attn_layers = nn.ModuleList()
        self.norm_layers_0 = nn.ModuleList()
        self.encdec_attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.self_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    n_heads,
                    p_dropout=p_dropout,
                    proximal_bias=proximal_bias,
                    proximal_init=proximal_init,
                )
            )
            self.norm_layers_0.append(LayerNorm(hidden_channels))
            self.encdec_attn_layers.append(
                MultiHeadAttention(
                    hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout
                )
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(
                    hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    p_dropout=p_dropout,
                    causal=True,
                )
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        h: torch.Tensor,
        h_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: decoder input
        h: encoder output
        """
        self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(
            device=x.device, dtype=x.dtype
        )
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.self_attn_layers[i](x, x, self_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_0[i](x + y)

            y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        heads_share: bool = True,
        block_length: Optional[int] = None,
        proximal_bias: bool = False,
        proximal_init: bool = False,
    ) -> None:
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.proximal_init = proximal_init
        self.attn = None

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels**-0.5
            self.emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )
            self.emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels)
                * rel_stddev
            )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                assert self.conv_k.bias is not None
                assert self.conv_q.bias is not None
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Debug: Check inputs
        if torch.isnan(x).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in input x")
        if torch.isnan(c).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in input c")

        q = self.conv_q(x)
        # Debug: After conv_q
        if torch.isnan(q).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in q after conv_q")
            print(f"  q dtype: {q.dtype}")

        k = self.conv_k(c)
        # Debug: After conv_k
        if torch.isnan(k).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in k after conv_k")

        v = self.conv_v(c)
        # Debug: After conv_v
        if torch.isnan(v).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in v after conv_v")

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        # Debug: After attention
        if torch.isnan(x).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in x after self.attention")
            print(f"  x dtype: {x.dtype}")

        x = self.conv_o(x)
        # Debug: After conv_o
        if torch.isnan(x).any():
            print(f"[DEBUG MultiHeadAttention.forward] NaN in x after conv_o")

        return x

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        SDPA-based attention with support for:
        - mask
        - block_length
        - proximal_bias
        - relative attention (window_size) for self-attention
        """

        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        q = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        k = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        v = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        device = q.device
        dtype = q.dtype
        scale = math.sqrt(self.k_channels)

        # ---- build additive attention bias ----
        # base bias shape: [1, 1, t_t, t_s] (broadcastable)
        attn_bias = torch.zeros((1, 1, t_t, t_s), device=device, dtype=dtype)

        # relative key logits (self-attention only)
        if self.window_size is not None:
            assert t_s == t_t, (
                "Relative attention is only available for self-attention."
            )
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s
            ).to(device=device, dtype=dtype)

            q_for_rel = q / scale

            rel_logits = self._matmul_with_relative_keys(
                q_for_rel, key_relative_embeddings
            )  # [b, h, t, 2*t-1]
            scores_local = self._relative_position_to_absolute_position(
                rel_logits
            )  # [b,h,t,t]

            attn_bias = attn_bias + scores_local  # broadcast add

        # proximal bias (self-attention only)
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            attn_bias = attn_bias + self._attention_bias_proximal(t_s).to(
                device=device, dtype=dtype
            )

        # user-provided mask -> additive bias
        if mask is not None:
            m = mask.to(device=device)

            # Normalize mask dims to 4D and keep it broadcastable.
            # Expected semantic: 0 = disallow, 1 = allow.
            if m.dim() == 2:
                # [t_t, t_s]
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                # [b, t_t, t_s]
                m = m.unsqueeze(1)
            # else assume already 4D-like

            # Convert to bool allow-mask
            m_allow = m != 0

            # Build mask bias with same shape as m (for clean broadcasting)
            mask_bias = torch.zeros_like(m_allow, dtype=dtype, device=device)
            mask_bias = mask_bias.masked_fill(~m_allow, -1e4)

            attn_bias = attn_bias + mask_bias

        # block_length local attention mask (self-attention only)
        if self.block_length is not None:
            assert t_s == t_t, "Local attention is only available for self-attention."
            # [t, t]
            local = torch.ones((t_t, t_s), device=device, dtype=torch.bool)
            local = local.triu(-self.block_length).tril(self.block_length)
            local = local.unsqueeze(0).unsqueeze(0)  # [1,1,t,t]

            block_bias = torch.zeros_like(local, dtype=dtype, device=device)
            block_bias = block_bias.masked_fill(~local, -1e4)

            attn_bias = attn_bias + block_bias

        # Safety: avoid inf/-inf from upstream contaminating softmax
        attn_bias = torch.nan_to_num(attn_bias, neginf=-1e4, posinf=1e4)

        # ---- SDPA main path ----
        dropout_p = self.p_dropout if self.training else 0.0

        # SDPA computes softmax((q k^T)/sqrt(d) + attn_mask) v
        # We pass attn_bias as additive mask.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=dropout_p,
            is_causal=False,
        )  # [b, h, t_t, d_k]

        # ---- build p_attn for compatibility/relative-value path ----
        # Compute attention weights explicitly using the same bias
        scores_base = torch.matmul(q / scale, k.transpose(-2, -1))
        scores = scores_base + attn_bias
        scores = torch.nan_to_num(scores, neginf=-1e4, posinf=1e4)

        p_attn = F.softmax(scores, dim=-1)  # [b, h, t_t, t_s]
        p_attn = self.drop(p_attn) if self.training and self.p_dropout > 0 else p_attn

        # ---- relative value contribution (same as original) ----
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s
            ).to(device=device, dtype=dtype)

            out = out + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings
            )

        # ---- reshape back ----
        out = out.transpose(2, 3).contiguous().view(b, d, t_t)

        return out, p_attn

    def _matmul_with_relative_values(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(
        self, relative_embeddings: torch.Tensor, length: int
    ) -> torch.Tensor:
        assert self.window_size is not None
        2 * self.window_size + 1  # type: ignore
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings,
                commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]),
            )
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[
            :, slice_start_position:slice_end_position
        ]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(
            x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]])
        )

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[
            :, :, :length, length - 1 :
        ]
        return x_final

    def _absolute_position_to_relative_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # pad along column
        x = F.pad(
            x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]])
        )
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length: int) -> torch.Tensor:
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)


class FFN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: Optional[str] = None,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal

        if causal:
            self.padding = self._causal_padding
        else:
            self.padding = self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x

    def _same_padding(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        padding = [[0, 0], [0, 0], [pad_l, pad_r]]
        x = F.pad(x, commons.convert_pad_shape(padding))
        return x
