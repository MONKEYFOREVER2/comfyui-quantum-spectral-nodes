"""
ComfyUI Node Definitions — Quantum Spectral Flow
=================================================

Nodes
-----
*  **SpectralLatentEnhance**    → LATENT    (frequency-domain post-process)
"""

from __future__ import annotations

import math
import functools
from typing import Any, Dict, Tuple

import torch

# ComfyUI imports — wrapped so the module can still be read/tested outside ComfyUI
try:
    import comfy.samplers
    import comfy.sample
    import comfy.utils
    import latent_preview
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False

from .sampling_core import (
    spectral_latent_enhance,
)


# ─── Helpers ────────────────────────────────────────────────────────────────

def _get_sigma_range(model) -> Tuple[float, float]:
    """Extract σ_min / σ_max from a ComfyUI model patcher."""
    try:
        ms = model.get_model_object("model_sampling")
        sigma_min = float(ms.sigma_min)
        sigma_max = float(ms.sigma_max)
        if sigma_min <= 0:
            sigma_min = 0.0292
        return sigma_min, sigma_max
    except Exception:
        return 0.0292, 14.6146          # safe SDXL-era defaults


def _apply_denoise(sigmas: torch.Tensor, steps: int, denoise: float) -> torch.Tensor:
    """Trim a sigma schedule when denoise < 1 (img2img / inpainting)."""
    if denoise < 1.0 and denoise > 0:
        # Standard ComfyUI convention: generate more sigmas, then slice
        total = int(steps / denoise)
        if total > len(sigmas) - 1:
            # Regenerate at the right length elsewhere; here just trim
            pass
        # Keep only the last `steps+1` entries
        sigmas = sigmas[-(steps + 1):]
    return sigmas









# ═══════════════════════════════════════════════════════════════════════════
#  4.  Spectral Latent Enhance  →  LATENT
# ═══════════════════════════════════════════════════════════════════════════

class SpectralLatentEnhanceNode:
    """
    Frequency-domain post-processing that boosts mid-band detail and
    gently attenuates high-frequency noise — applied to the latent
    *before* VAE decode.

    Works with any sampler, not just Quantum Spectral.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "latent": ("LATENT",),
                "detail_boost": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.5, "step": 0.05,
                    "tooltip": "How much to amplify mid-frequency detail",
                }),
                "denoise_high": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 0.8, "step": 0.05,
                    "tooltip": "How much to attenuate the highest frequencies (noise)",
                }),
            },
            "optional": {
                "low_cutoff": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.40, "step": 0.01}),
                "high_cutoff": ("FLOAT", {"default": 0.60, "min": 0.30, "max": 0.90, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "enhance"
    CATEGORY = "sampling/quantum_spectral/post_process"

    def enhance(
        self,
        latent,
        detail_boost: float,
        denoise_high: float,
        low_cutoff: float = 0.12,
        high_cutoff: float = 0.60,
    ):
        out = latent.copy()
        samples = out["samples"].clone()
        out["samples"] = spectral_latent_enhance(
            samples,
            detail_boost=detail_boost,
            denoise_high=denoise_high,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
        )
        return (out,)


# ═══════════════════════════════════════════════════════════════════════════
#  Registration
# ═══════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS: Dict[str, type] = {
    "SpectralLatentEnhance":   SpectralLatentEnhanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "SpectralLatentEnhance":   "Spectral Latent Enhance",
}
