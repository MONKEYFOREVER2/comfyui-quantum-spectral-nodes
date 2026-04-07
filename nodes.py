"""
ComfyUI Node Definitions — Quantum Spectral Flow
=================================================

Nodes
-----
*  **QuantumSpectralSampler**   → SAMPLER   (plug into SamplerCustom)
*  **PhiHarmonicScheduler**     → SIGMAS    (plug into SamplerCustom)
*  **QuantumFlowKSampler**      → LATENT    (all-in-one convenience node)
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
    quantum_spectral_flow_sample,
    phi_harmonic_sigmas,
    fibonacci_adaptive_sigmas,
    turbo_phi_sigmas,
    spectral_latent_enhance,
    PHI,
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
#  1.  Quantum Spectral Sampler  →  SAMPLER
# ═══════════════════════════════════════════════════════════════════════════

class QuantumSpectralSamplerNode:
    """
    Outputs a **SAMPLER** that implements the Quantum Spectral Flow algorithm.
    Plug it into *SamplerCustom* or *SamplerCustomAdvanced* alongside any
    SIGMAS schedule.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "spectral_weight": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Strength of frequency-domain trust gating (0 = off, 1 = full)",
                }),
                "momentum": ("FLOAT", {
                    "default": 0.55, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": "EMA decay for trajectory smoothing (0 = off)",
                }),
                "resonance_strength": ("FLOAT", {
                    "default": 0.025, "min": 0.0, "max": 0.20, "step": 0.005,
                    "tooltip": "Amplitude of stochastic resonance noise injection",
                }),
                "resonance_peak": ("FLOAT", {
                    "default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Progress point (0→1) where resonance is strongest",
                }),
                "eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Ancestral noise fraction (0 = deterministic, 1 = full ancestral)",
                }),
            },
            "optional": {
                "low_cutoff": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 0.40, "step": 0.01,
                    "tooltip": "Normalised frequency below which = 'low band'",
                }),
                "high_cutoff": ("FLOAT", {
                    "default": 0.55, "min": 0.30, "max": 0.90, "step": 0.01,
                    "tooltip": "Normalised frequency above which = 'high band'",
                }),
                "correction_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Langevin correction step size (extra model eval per step; 0 = off)",
                }),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "build_sampler"
    CATEGORY = "sampling/quantum_spectral/samplers"

    def build_sampler(
        self,
        spectral_weight: float,
        momentum: float,
        resonance_strength: float,
        resonance_peak: float,
        eta: float,
        low_cutoff: float = 0.15,
        high_cutoff: float = 0.55,
        correction_strength: float = 0.0,
    ):
        sampler_fn = functools.partial(
            quantum_spectral_flow_sample,
            spectral_weight=spectral_weight,
            momentum=momentum,
            resonance_strength=resonance_strength,
            resonance_peak=resonance_peak,
            eta=eta,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            correction_strength=correction_strength,
        )
        sampler = comfy.samplers.KSAMPLER(sampler_fn)
        return (sampler,)


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Phi-Harmonic Scheduler  →  SIGMAS
# ═══════════════════════════════════════════════════════════════════════════

SCHEDULER_MODES = ["phi_harmonic_dual", "phi_harmonic_single", "fibonacci_adaptive", "turbo_phi"]

class PhiHarmonicSchedulerNode:
    """
    Outputs a **SIGMAS** tensor built with golden-ratio mathematics.

    Four modes:
    * **phi_harmonic_dual** — dual-phase (structure + detail), best general choice
    * **phi_harmonic_single** — Karras-like with ρ=φ
    * **fibonacci_adaptive** — Fibonacci-ratio step spacing
    * **turbo_phi** — flow-matching shift + φ warp for distilled / turbo models
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 10000,
                    "tooltip": "Number of sampling steps",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength (< 1 for img2img)",
                }),
                "mode": (SCHEDULER_MODES, {
                    "default": "phi_harmonic_dual",
                    "tooltip": "Scheduling algorithm",
                }),
            },
            "optional": {
                "phi_power": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 3.0, "step": 0.05,
                    "tooltip": "Exponent on φ — values > 1 push more steps to structure phase",
                }),
                "turbo_shift": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 12.0, "step": 0.25,
                    "tooltip": "Timestep shift for turbo_phi mode (higher = more structure steps)",
                }),
            },
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/quantum_spectral/schedulers"

    def get_sigmas(
        self,
        model,
        steps: int,
        denoise: float,
        mode: str,
        phi_power: float = 1.0,
        turbo_shift: float = 3.0,
    ):
        sigma_min, sigma_max = _get_sigma_range(model)

        actual_steps = steps
        if denoise < 1.0 and denoise > 0:
            actual_steps = round(steps / denoise)

        if mode == "phi_harmonic_dual":
            sigmas = phi_harmonic_sigmas(actual_steps, sigma_min, sigma_max,
                                         phi_power=phi_power, dual_phase=True)
        elif mode == "phi_harmonic_single":
            sigmas = phi_harmonic_sigmas(actual_steps, sigma_min, sigma_max,
                                         phi_power=phi_power, dual_phase=False)
        elif mode == "fibonacci_adaptive":
            sigmas = fibonacci_adaptive_sigmas(actual_steps, sigma_min, sigma_max)
        elif mode == "turbo_phi":
            sigmas = turbo_phi_sigmas(actual_steps, sigma_min, sigma_max,
                                      shift=turbo_shift)
        else:
            sigmas = phi_harmonic_sigmas(actual_steps, sigma_min, sigma_max)

        sigmas = _apply_denoise(sigmas, steps, denoise)
        return (sigmas,)


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Quantum Flow KSampler  →  LATENT  (all-in-one)
# ═══════════════════════════════════════════════════════════════════════════

class QuantumFlowKSamplerNode:
    """
    Drop-in replacement for the stock KSampler that wires together the
    Quantum Spectral sampler **and** the Phi-Harmonic scheduler in one node.

    Ideal for quick experiments — no need to connect separate SAMPLER/SIGMAS.
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Scheduler
                "scheduler_mode": (SCHEDULER_MODES, {"default": "phi_harmonic_dual"}),
                # Sampler tunables
                "spectral_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
                "momentum": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 0.95, "step": 0.01}),
                "resonance_strength": ("FLOAT", {"default": 0.025, "min": 0.0, "max": 0.20, "step": 0.005}),
                "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "phi_power": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 3.0, "step": 0.05}),
                "turbo_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 12.0, "step": 0.25}),
                "resonance_peak": ("FLOAT", {"default": 0.45, "min": 0.1, "max": 0.9, "step": 0.05}),
                "low_cutoff": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.40, "step": 0.01}),
                "high_cutoff": ("FLOAT", {"default": 0.55, "min": 0.30, "max": 0.90, "step": 0.01}),
                "correction_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/quantum_spectral"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed: int,
        steps: int,
        cfg: float,
        denoise: float,
        scheduler_mode: str,
        spectral_weight: float,
        momentum: float,
        resonance_strength: float,
        eta: float,
        phi_power: float = 1.0,
        turbo_shift: float = 3.0,
        resonance_peak: float = 0.45,
        low_cutoff: float = 0.15,
        high_cutoff: float = 0.55,
        correction_strength: float = 0.0,
    ):
        # ── Build sigmas ────────────────────────────────────────────────
        sigma_min, sigma_max = _get_sigma_range(model)
        actual_steps = round(steps / denoise) if 0 < denoise < 1.0 else steps

        if scheduler_mode == "phi_harmonic_dual":
            sigmas = phi_harmonic_sigmas(actual_steps, sigma_min, sigma_max,
                                         phi_power=phi_power, dual_phase=True)
        elif scheduler_mode == "phi_harmonic_single":
            sigmas = phi_harmonic_sigmas(actual_steps, sigma_min, sigma_max,
                                         phi_power=phi_power, dual_phase=False)
        elif scheduler_mode == "fibonacci_adaptive":
            sigmas = fibonacci_adaptive_sigmas(actual_steps, sigma_min, sigma_max)
        elif scheduler_mode == "turbo_phi":
            sigmas = turbo_phi_sigmas(actual_steps, sigma_min, sigma_max,
                                      shift=turbo_shift)
        else:
            sigmas = phi_harmonic_sigmas(actual_steps, sigma_min, sigma_max)

        sigmas = _apply_denoise(sigmas, steps, denoise)

        # ── Build sampler ───────────────────────────────────────────────
        sampler_fn = functools.partial(
            quantum_spectral_flow_sample,
            spectral_weight=spectral_weight,
            momentum=momentum,
            resonance_strength=resonance_strength,
            resonance_peak=resonance_peak,
            eta=eta,
            low_cutoff=low_cutoff,
            high_cutoff=high_cutoff,
            correction_strength=correction_strength,
        )
        sampler = comfy.samplers.KSAMPLER(sampler_fn)

        # ── Prepare latent & noise ──────────────────────────────────────
        latent = latent_image.copy()
        latent_samples = latent["samples"]

        try:
            latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
            latent["samples"] = latent_samples
        except Exception:
            pass

        noise = comfy.sample.prepare_noise(latent_samples, seed)
        noise_mask = latent.get("noise_mask", None)

        # ── Callback (preview / progress bar) ───────────────────────────
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # ── Sample ──────────────────────────────────────────────────────
        samples = comfy.sample.sample_custom(
            model, noise, cfg, sampler, sigmas,
            positive, negative, latent_samples,
            noise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )

        out = latent.copy()
        out["samples"] = samples
        return (out,)


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
    "QuantumSpectralSampler":  QuantumSpectralSamplerNode,
    "PhiHarmonicScheduler":    PhiHarmonicSchedulerNode,
    "QuantumFlowKSampler":     QuantumFlowKSamplerNode,
    "SpectralLatentEnhance":   SpectralLatentEnhanceNode,
}

NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {
    "QuantumSpectralSampler":  "Quantum Spectral Sampler",
    "PhiHarmonicScheduler":    "Phi-Harmonic Scheduler",
    "QuantumFlowKSampler":     "Quantum Flow KSampler",
    "SpectralLatentEnhance":   "Spectral Latent Enhance",
}
