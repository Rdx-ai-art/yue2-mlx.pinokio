"""Hum-conditioned acoustic synthesis: lyra's CachedNAR plus prosody-adapter injections.

The adapter (lora.AdapterInfo with kind == "hum") ships a NAR LoRA merged like any other,
plus hum_proj.k linears (latent 64 -> hidden 2048). Their outputs for the carrier latents
are added to the decoder's hidden state: projection 0 at the input (next to vae2llm(x_t))
and the others before the inject_layers. Frames without hum are zeros, which map to
"no hum" bias.

Classifier-free guidance on hum channel: v = v_zero + g * (v_hum - v_zero); g == 1 is
plain conditioned pass, anything else costs two NAR forward passes.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from numbers import Real

import mlx.core as mx
import numpy as np
from lyra.nar import _LATENT_DIM, CachedNAR, Chunk, _chunks_with_supplied_noise, _logit

from hum import HumOptions, place_condition

Projection = tuple[mx.array, mx.array]  # (weight [hidden, latent] f32, bias [hidden] f32)


def load_hum_projections(adapter_info, *, hidden: int, latent: int = _LATENT_DIM) -> list[Projection]:
    """hum_proj.k weights (FP32) in inject_layers order, shape-checked."""
    if adapter_info.get("kind") != "hum":
        raise ValueError(f"{adapter_info.get('name')} is not a hum-to-song adapter")
    tensors = mx.load(str(adapter_info["weights_path"]))
    projections = []
    for spec in adapter_info.get("hum_proj", []):
        weight = tensors[spec["weight_name"]].astype(mx.float32)
        bias = tensors[spec["bias_name"]].astype(mx.float32)
        if tuple(weight.shape) != (hidden, latent) or tuple(bias.shape) != (hidden,):
            raise ValueError(f"hum_proj.{spec['index']} is {tuple(weight.shape)}, "
                             f"model needs [{hidden}, {latent}]")
        projections.append((weight, bias))
    mx.eval(*(t for pair in projections for t in pair))
    return projections


class HumNAR(CachedNAR):
    """CachedNAR whose hidden state receives carrier projections; supports hum-channel CFG."""

    def __init__(self, model, chunk: Chunk, *, cond: np.ndarray, projections: list[Projection],
                 inject_layers: list[int], query_chunk_size=None, cancelled=None):
        super().__init__(model, chunk, query_chunk_size=query_chunk_size, cancelled=cancelled)
        cond = np.asarray(cond, dtype=np.float32)
        if cond.shape != (self.nar_length - 2, _LATENT_DIM):
            self.close()
            raise ValueError(f"condition must be {(self.nar_length - 2, _LATENT_DIM)}, got {cond.shape}")
        if len(projections) != len(inject_layers):
            self.close()
            raise ValueError("one hum projection per inject layer is required")
        depth = len(model.model.layers)
        if any(layer < 0 or layer >= depth for layer in inject_layers):
            self.close()
            raise ValueError(f"inject_layers must be in [0, {depth})")
        # Carrier gets same boundary frame on each side as ODE state (zeros -> bias)
        padded = mx.pad(mx.array(cond), ((1, 1), (0, 0)))
        self.stem: dict[int, mx.array] = {}
        self.zero: dict[int, mx.array] = {}
        for layer, (weight, bias) in zip(inject_layers, projections, strict=True):
            self.stem[layer] = ((padded @ weight.T) + bias).astype(mx.bfloat16)[None, :, :]
            self.zero[layer] = bias.astype(mx.bfloat16)[None, None, :]
        mx.eval(*self.stem.values(), *self.zero.values())

    def _velocity_with(self, state, raw_t, inject: dict[int, mx.array]) -> mx.array:
        """CachedNAR.velocity plus hum additions; returns BF16 [frames, 64]."""
        if self._closed:
            raise RuntimeError("HumNAR is closed")
        if isinstance(raw_t, bool) or not isinstance(raw_t, Real) or not math.isfinite(raw_t):
            raise ValueError("raw_t must be a finite real number")
        state = self._state_array(state)
        boundary_state = mx.pad(state, ((1, 1), (0, 0)))
        shifted = self.model.shift_timestep(float(raw_t))
        x = self.model.vae2llm(boundary_state[None, :, :])
        if 0 in inject:
            x = x + inject[0]
        timesteps = mx.broadcast_to(shifted, (self.nar_length,))
        time_embedding = self.model.time_embedder(timesteps)[None, :, :]
        x = x + time_embedding
        x = x + self.pos_emb

        for index, (nar_layer, (ar_key, ar_value)) in enumerate(zip(self.model.model.layers, self.cache,
                                                                   strict=True)):
            if self._cancelled is not None and self._cancelled():
                raise InterruptedError("Cancelled during acoustic velocity")
            if index > 0 and index in inject:
                x = x + inject[index]
            query, key, value = self._project(
                nar_layer.nar_self_attn,
                nar_layer.nar_input_layernorm(x),
                self.cos,
                self.sin,
            )
            key = mx.concatenate([ar_key, key], axis=2)
            value = mx.concatenate([ar_value, value], axis=2)
            attention = self._attend(query, key, value)
            attention = attention.transpose(0, 2, 1, 3).reshape(1, self.nar_length, -1)
            x = x + nar_layer.nar_self_attn.o_proj(attention)
            x = x + nar_layer.nar_mlp(nar_layer.nar_pre_mlp_layernorm(x))

        x = self.model.ar_model.model.norm(x)
        return self.model.llm2vae(x)[0, 1:-1, :]

    def velocity(self, state, raw_t) -> mx.array:
        return self._velocity_with(state, raw_t, self.stem)

    def guided_velocity(self, state, raw_t, guidance: float) -> mx.array:
        """v_zero + g * (v_hum - v_zero); g == 1 skips unconditioned pass."""
        if guidance == 1.0:
            return self.velocity(state, raw_t)
        conditioned = self._velocity_with(state, raw_t, self.stem).astype(mx.float32)
        mx.eval(conditioned)
        unconditioned = self._velocity_with(state, raw_t, self.zero).astype(mx.float32)
        return (unconditioned + guidance * (conditioned - unconditioned)).astype(mx.bfloat16)

    def solve(self, steps=32, cancelled: Callable[[], bool] | None = None,
              on_progress: Callable[[int, int], None] | None = None, *, guidance: float = 1.0) -> np.ndarray:
        """Midpoint solver over the guided velocity."""
        if self._closed:
            raise RuntimeError("HumNAR is closed")
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        guidance = HumOptions.influence.__class__.__bases__[0](guidance) if hasattr(HumOptions, 'influence') else float(guidance)
        state = self._initial_state
        dt = 1.0 / steps
        half_dt = mx.array(dt / 2.0, dtype=mx.float32)
        full_dt = mx.array(dt, dtype=mx.float32)
        mx.eval(half_dt, full_dt)
        for step in range(steps):
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            t = 1.0 - step * dt
            first = self.guided_velocity(state, _logit(t), guidance)
            mx.eval(first)
            midpoint = state - (first.astype(mx.float32) * half_dt).astype(mx.bfloat16)
            if cancelled is not None and cancelled():
                raise InterruptedError("Cancelled during acoustic flow matching")
            velocity = self.guided_velocity(midpoint, _logit(t - dt / 2.0), guidance)
            state = state - (velocity.astype(mx.float32) * full_dt).astype(mx.bfloat16)
            mx.eval(state)
            if on_progress is not None:
                on_progress(step + 1, steps)
        result = np.array(state.astype(mx.float32), dtype=np.float32, copy=True)
        if result.shape != (self.nar_length - 2, _LATENT_DIM):
            raise RuntimeError("Acoustic solver returned invalid latent shape")
        if not np.isfinite(result).all():
            raise FloatingPointError("Acoustic flow matching produced non-finite latents")
        return result

    def close(self) -> None:
        self.stem = {}
        self.zero = {}
        super().close()


def make_synthesizer(pipe, carrier_latents: np.ndarray, adapter_info: dict, influence: float,
                     offset_s: float):
    """A synthesize(semantic, *, cancelled, noise) drop-in for pipe._run_create."""

    def synthesize(semantic, *, cancelled=None, noise=None):
        from mlx_inference import chunk_ranges, token_prefix
        cond = place_condition(carrier_latents, len(semantic["tokens"]), offset_s)
        return synthesize_hum(pipe, semantic, cond=cond, adapter_info=adapter_info,
                              influence=influence, cancelled=cancelled, noise=noise)

    return synthesize


def synthesize_hum(pipe, semantic, *, cond: np.ndarray, adapter_info: dict, influence: float,
                   cancelled=None, noise=None) -> np.ndarray:
    """YuE2Pipeline.synthesize with HumNAR: same cuts, noise, stage events."""
    from mlx_inference import chunk_ranges, token_prefix

    plan = semantic
    if noise is None:
        from lyra.pipeline import initial_noise
        seed = plan.get("seed", 831001)
        noise = initial_noise(len(plan["tokens"]), seed)
    
    model = pipe._load_model(for_nar=True)
    projections = load_hum_projections(adapter_info, hidden=model.config["hidden_size"])
    steps = plan.get("ode_steps", 32)
    context = plan.get("context", 24576)
    seed = plan.get("seed", 831001)
    
    tokens = plan["tokens"]
    prefix = plan["prefix"]
    
    chunks = _chunks_with_supplied_noise(prefix, tokens, seed, context, noise)
    ranges = chunk_ranges(len(tokens), len(prefix), int(context))
    total_steps = steps * len(chunks)
    output: list[np.ndarray] = []
    
    for index, (chunk, (start, end)) in enumerate(zip(chunks, ranges, strict=True)):
        inject_layers = adapter_info.get("inject_layers", [])
        engine = HumNAR(model, chunk, cond=cond[start:end], projections=projections,
                        inject_layers=inject_layers, query_chunk_size=pipe.query_chunk_size,
                        cancelled=cancelled)
        try:
            def report(completed, _total, _index=index):
                if pipe.log:
                    pipe.log(f"Synthesizing audio: step {completed}/{steps} (chunk {_index})")
            output.append(engine.solve(steps, cancelled, report, guidance=influence))
        finally:
            engine.close()
    
    del projections
    mx.clear_cache()
    result = output[0] if len(output) == 1 else np.concatenate(output, axis=0)
    return result
