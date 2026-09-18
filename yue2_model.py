"""YuE2 AR/NAR Mixture-of-Transformers backbone in MLX.

Mirrors m-a-p/YuE2-3B ``modeling_yue2.py``: a Qwen3-style decoder whose every
layer carries two attention/MLP paths (AR for text+codec tokens, NAR for the
acoustic flow-matching state). Weight names follow the checkpoint except
``time_embedder.mlp.{0,2}`` which ``convert.py`` renames to ``fc1``/``fc2``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


class KVCache:
    """Append-only per-layer cache, grown in 256-slot steps (mlx_lm pattern)."""

    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update(self, k, v):
        prev = self.offset
        B, H, T, D = k.shape
        if self.keys is None or prev + T > self.keys.shape[2]:
            n = ((T + self.step - 1) // self.step) * self.step
            new_k, new_v = mx.zeros((B, H, n, D), k.dtype), mx.zeros((B, H, n, D), v.dtype)
            if self.keys is None:
                self.keys, self.values = new_k, new_v
            else:
                if prev % self.step:
                    self.keys, self.values = self.keys[..., :prev, :], self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
        self.offset += T
        self.keys[..., prev:self.offset, :] = k
        self.values[..., prev:self.offset, :] = v
        return self.keys[..., :self.offset, :], self.values[..., :self.offset, :]


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.n_kv, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        h, d = cfg["hidden_size"], self.head_dim
        self.q_proj = nn.Linear(h, self.n_heads * d, bias=False)
        self.k_proj = nn.Linear(h, self.n_kv * d, bias=False)
        self.v_proj = nn.Linear(h, self.n_kv * d, bias=False)
        self.o_proj = nn.Linear(self.n_heads * d, h, bias=False)
        self.q_norm = nn.RMSNorm(d, eps=cfg["rms_norm_eps"])
        self.k_norm = nn.RMSNorm(d, eps=cfg["rms_norm_eps"])
        self.rope_base = float(cfg["rope_theta"])
        self.scale = d ** -0.5

    def project_qkv(self, x, offset):
        """Return q [B,H,T,D], k/v [B,KV,T,D] after QK-norm and RoPE at ``offset``."""
        B, T, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(B, T, self.n_heads, -1)).transpose(0, 2, 1, 3)
        k = self.k_norm(self.k_proj(x).reshape(B, T, self.n_kv, -1)).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.n_kv, -1).transpose(0, 2, 1, 3)
        rope = lambda a: mx.fast.rope(a, self.head_dim, traditional=False, base=self.rope_base, scale=1.0, offset=offset)
        return rope(q), rope(k), v

    def attend(self, q, k, v, mask):
        B, _, T, _ = q.shape
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.input_layernorm = nn.RMSNorm(h, eps=eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=eps)
        self.mlp = MLP(cfg)
        self.nar_input_layernorm = nn.RMSNorm(h, eps=eps)
        self.nar_self_attn = Attention(cfg)
        self.nar_pre_mlp_layernorm = nn.RMSNorm(h, eps=eps)
        self.nar_mlp = MLP(cfg)

    def ar(self, x, cache: KVCache, mask):
        q, k, v = self.self_attn.project_qkv(self.input_layernorm(x), cache.offset)
        k, v = cache.update(k, v)
        x = x + self.self_attn.attend(q, k, v, mask)
        return x + self.mlp(self.post_attention_layernorm(x))

    def ar_prefill_kv(self, x):
        """AR pass that also returns this layer's (k, v) for the NAR to attend over."""
        q, k, v = self.self_attn.project_qkv(self.input_layernorm(x), 0)
        x = x + self.self_attn.attend(q, k, v, "causal")
        return x + self.mlp(self.post_attention_layernorm(x)), (k, v)

    def nar(self, x, ar_k, ar_v, offset):
        q, k, v = self.nar_self_attn.project_qkv(self.nar_input_layernorm(x), offset)
        k, v = mx.concatenate([ar_k, k], axis=2), mx.concatenate([ar_v, v], axis=2)
        x = x + self.nar_self_attn.attend(q, k, v, None)
        return x + self.nar_mlp(self.nar_pre_mlp_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = [DecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        self.norm = nn.RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden, freq_size=256):
        super().__init__()
        self.freq_size = freq_size
        self.fc1 = nn.Linear(freq_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

    def __call__(self, t, dtype):
        half = self.freq_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(half, dtype=mx.float32) / half)
        args = t.astype(mx.float32).reshape(-1, 1) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)
        return self.fc2(nn.silu(self.fc1(emb)))


class LatentPosEmbed(nn.Module):
    """Holds the checkpoint's precomputed sinusoidal table ``pe`` [max_frames, hidden]."""

    def __init__(self, max_frames, hidden):
        super().__init__()
        self.pe = mx.zeros((max_frames, hidden))


class Yue2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        h, d = cfg["hidden_size"], cfg["latent_dim"]
        self.model = Backbone(cfg)
        self.lm_head = nn.Linear(h, cfg["vocab_size"], bias=False)
        self.llm2vae = nn.Linear(h, d)
        self.vae2llm = nn.Linear(d, h)
        self.time_embedder = TimestepEmbedder(h)
        self.latent_pos_embed = LatentPosEmbed(cfg["max_latent_frames"], h)

    # AR path
    def ar_step(self, tokens, caches, all_positions=False):
        """tokens [B,T] → logits [B,V] of the last position (or [B,T,V])."""
        x = self.model.embed_tokens(tokens)
        mask = "causal" if tokens.shape[1] > 1 else None
        for layer, cache in zip(self.model.layers, caches):
            x = layer.ar(x, cache, mask)
        x = self.model.norm(x if all_positions else x[:, -1:])
        logits = self.lm_head(x)
        return logits if all_positions else logits[:, -1]

    # NAR path
    def nar_prefill(self, ar_tokens):
        """Run the AR path once over the chunk prefix; return per-layer (k, v)."""
        x = self.model.embed_tokens(mx.array([ar_tokens]))
        cache = []
        for layer in self.model.layers:
            x, kv = layer.ar_prefill_kv(x)
            cache.append(kv)
        mx.eval(cache)
        return cache

    def nar_velocity(self, state, raw_t, ar_cache, ar_length):
        """Flow-matching velocity for ``state`` [T,64] at logit-time ``raw_t``."""
        shift = self.config["timestep_shift"]
        sig = mx.sigmoid(mx.array(raw_t, dtype=state.dtype))
        t = shift * sig / (1 + (shift - 1) * sig)
        n = state.shape[0] + 2  # LATENT_START + frames + LATENT_END
        x = self.vae2llm(mx.pad(state, [(1, 1), (0, 0)])[None])
        x = x + self.time_embedder(t, state.dtype)[None]
        pos = mx.minimum(mx.arange(n), self.config["max_latent_frames"] - 1)
        x = x + self.latent_pos_embed.pe[pos][None]
        for layer, (k, v) in zip(self.model.layers, ar_cache):
            x = layer.nar(x, k, v, ar_length)
        return self.llm2vae(self.model.norm(x))[0, 1:-1]


def load_model(path: Path) -> Yue2Model:
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    model = Yue2Model(cfg)
    weights = mx.load(str(path / "model.safetensors"))
    q = cfg.get("quantization")
    if q:
        nar_bits = q.get("nar_bits", q["bits"])

        def predicate(path, module):
            if not isinstance(module, nn.Linear) or f"{path}.scales" not in weights:
                return False
            return {"group_size": q["group_size"], "bits": nar_bits if ".nar_" in path else q["bits"]}

        nn.quantize(model, q["group_size"], q["bits"], class_predicate=predicate)
    model.load_weights(list(weights.items()))
    mx.eval(model.parameters())
    return model
