"""LoRA adapter loading and merging for YuE2 MLX pipeline.

Supports both regular LoRA adapters (kind="lora") and hum-to-song adapters (kind="hum").
Hum adapters include NAR LoRA plus hum_proj.k projections for prosody conditioning.
"""
import hashlib
import json
import logging
import re
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PEFT_WEIGHTS = "adapter_model.safetensors"
_PEFT_CONFIG = "adapter_config.json"
# Match hum_proj tensors with optional prefix (e.g. "hum_proj.0.weight" or "yue2_hum.hum_proj.0.weight")
_HUM_PROJ = re.compile(r"(?:^.+\.)?hum_proj\.(\d+)\.(weight|bias)$")


def valid_name(name) -> bool:
    """Check if adapter name is valid."""
    return isinstance(name, str) and NAME_PATTERN.fullmatch(name) is not None and ".." not in name


def discover_loras(lora_dir: Path) -> list[dict]:
    """Discover all LoRA adapters (regular and hum) in the loras directory.

    Returns list of dicts with keys: name, path, kind, file_hash, scale, valid
    """
    loras = []
    if not lora_dir.exists():
        logger.warning(f"Lora directory does not exist: {lora_dir}")
        return loras

    logger.info(f"Scanning lora directory: {lora_dir}")
    for item in sorted(lora_dir.iterdir()):
        logger.info(f"Found item: {item.name} (is_dir={item.is_dir()})")
        if item.is_dir():
            # PEFT directory format
            config_path = item / _PEFT_CONFIG
            weights_path = item / _PEFT_WEIGHTS
            if config_path.exists() and weights_path.exists():
                adapter_info = _load_peft_adapter(item)
                if adapter_info:
                    loras.append(adapter_info)
                    logger.info(f"Loaded PEFT adapter: {adapter_info['name']} (kind={adapter_info['kind']})")
        elif item.suffix == ".safetensors":
            # Single file format
            adapter_info = _load_single_file_adapter(item)
            if adapter_info:
                loras.append(adapter_info)
                logger.info(f"Loaded single-file adapter: {adapter_info['name']} (kind={adapter_info['kind']})")

    logger.info(f"Found {len(loras)} total adapters: {[l['name'] for l in loras]}")
    return loras


def list_hum_adapters(lora_dir: Path) -> list[dict]:
    """Return only hum adapters (kind="hum")."""
    all_loras = discover_loras(lora_dir)
    return [a for a in all_loras if a.get("kind") == "hum"]


def _compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _read_safetensors_header(path: Path) -> tuple[dict[str, dict], dict[str, str]]:
    """Read safetensors header: (tensors_info, metadata)."""
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensors file is truncated")
        (size,) = struct.unpack("<Q", prefix)
        header = json.loads(f.read(size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    metadata = header.pop("__metadata__", None) or {}
    return header, {str(k): str(v) for k, v in metadata.items()} if isinstance(metadata, dict) else {}


def _load_single_file_adapter(filepath: Path) -> dict | None:
    """Load adapter from a single .safetensors file."""
    try:
        header, metadata = _read_safetensors_header(filepath)
        
        # Debug: print first few keys
        logger.debug(f"Loading {filepath.name}, header keys: {list(header.keys())[:5]}...")
        
        # Check for hum_proj tensors
        hum_proj = []
        for name in header:
            match = _HUM_PROJ.match(name)
            if match:
                hum_proj.append({
                    "index": int(match.group(1)),
                    "type": match.group(2),
                    "name": name,
                    "shape": header[name]["shape"],
                })
        
        # Group hum_proj by index
        hum_proj_grouped = {}
        for proj in hum_proj:
            idx = proj["index"]
            if idx not in hum_proj_grouped:
                hum_proj_grouped[idx] = {}
            hum_proj_grouped[idx][proj["type"]] = proj["name"]
        
        has_hum_proj = len(hum_proj_grouped) > 0
        
        # Check if it's a hum adapter (has hum_proj tensors)
        is_hum = has_hum_proj
        
        # Calculate scale from metadata
        lora_scale = 1.0
        raw_scale = metadata.get("lora_scale", "1.0")
        try:
            lora_scale = float(raw_scale)
        except (ValueError, TypeError):
            pass
        
        # Get inject_layers from metadata
        inject_layers = []
        raw_inject = metadata.get("inject_layers", "[]")
        try:
            inject_layers = json.loads(raw_inject)
            if not isinstance(inject_layers, list):
                inject_layers = []
        except (ValueError, TypeError):
            pass
        
        result = {
            "name": filepath.stem,
            "path": filepath,
            "kind": "hum" if is_hum else "lora",
            "file_hash": _compute_file_hash(filepath),
            "scale": lora_scale,
            "valid": True,
            "hum_proj": list(hum_proj_grouped.values()) if is_hum else [],
            "inject_layers": inject_layers if is_hum else [],
            "weights_path": filepath,
        }
        logger.info(f"Loaded {filepath.name}: kind={result['kind']}, hum_proj={len(hum_proj_grouped)}, inject_layers={inject_layers}")
        return result
    except Exception as e:
        logger.warning(f"Failed to load adapter {filepath}: {e}")
        return None


def _load_peft_adapter(adapter_dir: Path) -> dict | None:
    """Load adapter from PEFT directory format."""
    try:
        with open(adapter_dir / _PEFT_CONFIG, "r") as f:
            config = json.load(f)

        weights_path = adapter_dir / _PEFT_WEIGHTS
        
        # Calculate scale from config
        lora_alpha = config.get("lora_alpha", 1)
        r = config.get("rank", 1)
        scale = lora_alpha / r if r > 0 else 1.0
        
        # Read safetensors header to check for hum_proj
        header, metadata = _read_safetensors_header(weights_path)
        
        # Check for hum_proj tensors
        hum_proj = []
        for name in header:
            match = _HUM_PROJ.match(name)
            if match:
                hum_proj.append({
                    "index": int(match.group(1)),
                    "type": match.group(2),
                    "name": name,
                    "shape": header[name]["shape"],
                })
        
        # Group hum_proj by index
        hum_proj_grouped = {}
        for proj in hum_proj:
            idx = proj["index"]
            if idx not in hum_proj_grouped:
                hum_proj_grouped[idx] = {}
            hum_proj_grouped[idx][proj["type"]] = proj["name"]
        
        has_hum_proj = len(hum_proj_grouped) > 0
        
        # Get inject_layers from metadata
        inject_layers = []
        raw_inject = metadata.get("inject_layers", "[]")
        try:
            inject_layers = json.loads(raw_inject)
            if not isinstance(inject_layers, list):
                inject_layers = []
        except (ValueError, TypeError):
            pass

        return {
            "name": adapter_dir.name,
            "path": adapter_dir,
            "kind": "hum" if has_hum_proj else "lora",
            "file_hash": _compute_file_hash(weights_path),
            "scale": scale,
            "valid": True,
            "hum_proj": list(hum_proj_grouped.values()) if has_hum_proj else [],
            "inject_layers": inject_layers if has_hum_proj else [],
            "weights_path": weights_path,
        }
    except Exception as e:
        logger.warning(f"Failed to load PEFT adapter {adapter_dir}: {e}")
        return None


class LoRALinear(nn.Module):
    """Wraps an existing Linear or QuantizedLinear module to apply LoRA adaptations dynamically."""

    def __init__(self, base_layer):
        super().__init__()
        self.base_layer = base_layer
        self.adapters = []

    def add_adapter(self, lora_a: mx.array, lora_b: mx.array, scale: float) -> None:
        self.adapters.append((lora_a, lora_b, scale))

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base_layer(x)
        for lora_a, lora_b, scale in self.adapters:
            # lora_a: (rank, in_features), lora_b: (out_features, rank)
            delta = (x.astype(lora_a.dtype) @ lora_a.T) @ lora_b.T
            out = out + (scale * delta).astype(out.dtype)
        return out


def _find_module_target(root, key_path: str):
    """Resolve a dot-separated path to (parent_obj, attr_name_or_index)."""
    parts = key_path.split(".")
    obj = root
    for p in parts[:-1]:
        if p.isdigit():
            idx = int(p)
            if idx >= len(obj):
                return None, None
            obj = obj[idx]
        else:
            if not hasattr(obj, p):
                return None, None
            obj = getattr(obj, p)
    attr = parts[-1]
    if attr.isdigit():
        idx = int(attr)
        if idx >= len(obj):
            return None, None
        return obj, idx
    if hasattr(obj, attr):
        return obj, attr
    return None, None


def apply_lora(model, adapters: list[dict], scale: float = 1.0) -> None:
    """Apply LoRA adapters to model weights in-place.

    Args:
        model: Model instance (nn.Module or weights dict)
        adapters: List of adapter dicts with 'path' and 'scale' keys
        scale: Global scale multiplier (0-4)
    """
    if not isinstance(model, nn.Module) and isinstance(model, dict):
        logger.info(f"Model dict has {len(model)} keys")
    else:
        logger.info(f"Applying LoRA to model instance: {type(model).__name__}")
    
    for adapter in adapters:
        adapter_path = adapter["path"]
        adapter_scale = adapter.get("scale", 1.0) * scale

        try:
            if adapter_path.is_dir():
                # PEFT format
                weights_path = adapter_path / "adapter_model.safetensors"
                adapter_data = mx.load(str(weights_path))
                _apply_peft_lora(model, adapter_data, adapter_scale)
            else:
                # Single file format
                adapter_data = mx.load(str(adapter_path))
                _apply_safetensors_lora(model, adapter_data, adapter_scale)

            logger.info(f"Applied LoRA adapter: {adapter['name']} (scale={adapter_scale:.2f})")
        except Exception as e:
            logger.error(f"Failed to apply adapter {adapter['name']}: {e}")


def _apply_peft_lora(model, adapter_data: dict, scale: float) -> None:
    """Apply PEFT-format LoRA adapters.
    
    PEFT format: base_model.model.<module>.lora_{A,B}.weight
    """
    # Group tensors by module
    modules = {}
    for key, value in adapter_data.items():
        # Parse key like "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        parts = key.split(".")
        if len(parts) >= 6 and parts[0] == "base_model":
            module_key = ".".join(parts[3:-2])  # e.g., "model.layers.0.self_attn.q_proj"
            lora_type = parts[-2]  # "lora_A" or "lora_B"
            if module_key not in modules:
                modules[module_key] = {}
            modules[module_key][lora_type] = value
    
    # Apply each module's LoRA
    for module_key, lora_tensors in modules.items():
        if "lora_A" in lora_tensors and "lora_B" in lora_tensors:
            _merge_lora_into_module(model, module_key, lora_tensors["lora_A"], lora_tensors["lora_B"], scale)


def _apply_safetensors_lora(model, adapter_data: dict, scale: float) -> None:
    """Apply single-file LoRA adapters.
    
    Single file format: layers.N.<block>.<proj>.lora_A / lora_B
    """
    # Group tensors by module
    modules = {}
    for key, value in adapter_data.items():
        # Parse key like "layers.0.self_attn.q_proj.lora_A"
        if ".lora_A" in key:
            module_key = key.replace(".lora_A", "")
            lora_type = "lora_A"
        elif ".lora_B" in key:
            module_key = key.replace(".lora_B", "")
            lora_type = "lora_B"
        else:
            continue
        
        if module_key not in modules:
            modules[module_key] = {}
        modules[module_key][lora_type] = value
    
    # Apply each module's LoRA
    for module_key, lora_tensors in modules.items():
        if "lora_A" in lora_tensors and "lora_B" in lora_tensors:
            _merge_lora_into_module(model, module_key, lora_tensors["lora_A"], lora_tensors["lora_B"], scale)


def _merge_lora_into_module(model, module_key: str, lora_a: mx.array, lora_b: mx.array, scale: float) -> None:
    """Apply LoRA weights by wrapping the target module or merging if dict."""
    if not isinstance(model, nn.Module) and isinstance(model, dict):
        model_key = module_key if module_key.endswith(".weight") else module_key + ".weight"
        if not model_key.startswith("model."):
            model_key = "model." + model_key
        if model_key not in model:
            logger.warning(f"Module {module_key} not found in model dict, skipping LoRA")
            return
        lora_a_np = np.array(lora_a)
        lora_b_np = np.array(lora_b)
        lora_update = scale * (lora_b_np @ lora_a_np)
        current_weight = np.array(model[model_key])
        model[model_key] = mx.array(current_weight + lora_update)
        return

    # Clean up prefixes and suffixes
    cleaned = module_key
    if cleaned.endswith(".weight"):
        cleaned = cleaned[:-7]
    if cleaned.startswith("base_model.model."):
        cleaned = cleaned[len("base_model.model."):]

    candidates = [cleaned]
    if not cleaned.startswith("model."):
        candidates.append("model." + cleaned)
    if cleaned.startswith("model."):
        candidates.append(cleaned[6:])
    if not cleaned.startswith("model.model."):
        candidates.append("model." + cleaned)

    parent, attr = None, None
    for cand in candidates:
        p, a = _find_module_target(model, cand)
        if p is not None:
            parent, attr = p, a
            break

    if parent is None:
        logger.warning(f"Module {module_key} not found in model, skipping LoRA")
        return

    curr_mod = parent[attr] if isinstance(attr, int) else getattr(parent, attr)
    if isinstance(curr_mod, LoRALinear):
        curr_mod.add_adapter(lora_a, lora_b, scale)
    else:
        wrapper = LoRALinear(curr_mod)
        wrapper.add_adapter(lora_a, lora_b, scale)
        if isinstance(attr, int):
            parent[attr] = wrapper
        else:
            setattr(parent, attr, wrapper)

    logger.debug(f"Applied LoRA to {module_key} (scale={scale:.4f})")
