"""
Quantum Spectral Flow — Core Sampling & Scheduling Algorithms
=============================================================

Novel diffusion sampling built on three pillars:

1.  **Spectral Trust Weighting**
    Each denoising step decomposes the prediction into frequency bands via FFT.
    Low-frequency structure is trusted from the start; high-frequency detail is
    progressively trusted as sigma decreases — matching the natural coarse→fine
    emergence order of diffusion models.

2.  **Trajectory Momentum**
    An exponential moving average (Polyak-style) of denoising predictions smooths
    the sampling trajectory, preventing the oscillatory artifacts common in
    few-step generation with turbo/distilled models.

3.  **Stochastic Resonance**
    Controlled, bandpass-filtered noise injection at intermediate steps exploits
    the physics principle that *adding* optimal noise to a sub-threshold signal
    can enhance its detection — coaxing fine detail from limited step budgets.

Schedulers use the golden ratio (φ ≈ 1.618) and Fibonacci sequence for
mathematically optimal sigma placement.
"""

import math
import torch
import torch.fft as fft
from typing import Tuple, Dict, Optional
import functools

# ─── Constants ──────────────────────────────────────────────────────────────
PHI       = (1.0 + math.sqrt(5.0)) / 2.0   # Golden ratio  ≈ 1.61803
PHI_INV   = 1.0 / PHI                       # Inverse       ≈ 0.61803
PHI_SQ    = PHI * PHI                        # φ²            ≈ 2.61803


# ═══════════════════════════════════════════════════════════════════════════
#  Frequency-Domain Utilities
# ═══════════════════════════════════════════════════════════════════════════

def create_frequency_masks(
    height: int,
    width: int,
    device: torch.device,
    low_cutoff: float = 0.15,
    high_cutoff: float = 0.55,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build smooth low / mid / high frequency masks in the FFT domain.

    Uses sigmoid transitions (no hard edges) so the sampler never introduces
    ringing artifacts.
    """
    freq_y = torch.fft.fftfreq(height, device=device)
    freq_x = torch.fft.fftfreq(width, device=device)
    fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
    freq_dist = torch.sqrt(fy ** 2 + fx ** 2)

    max_freq = freq_dist.max()
    if max_freq > 0:
        freq_dist = freq_dist / max_freq

    sharpness = 20.0
    low_mask  = torch.sigmoid(sharpness * (low_cutoff  - freq_dist))
    high_mask = torch.sigmoid(sharpness * (freq_dist   - high_cutoff))
    mid_mask  = torch.clamp(1.0 - low_mask - high_mask, min=0.0)

    return low_mask, mid_mask, high_mask


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast a 2-D mask to match *target*'s batch/channel dims."""
    while mask.dim() < target.dim():
        mask = mask.unsqueeze(0)
    return mask.expand_as(target)


# ═══════════════════════════════════════════════════════════════════════════
#  Spectral Trust Blending
# ═══════════════════════════════════════════════════════════════════════════

def spectral_guided_blend(
    denoised: torch.Tensor,
    sigma: float,
    sigma_max: float,
    spectral_weight: float = 0.5,
    low_cutoff: float = 0.15,
    high_cutoff: float = 0.55,
) -> torch.Tensor:
    """
    Re-weight the model's denoised prediction in the frequency domain.

    *   **Low band** — full trust throughout (structure is reliable early).
    *   **Mid band** — trust grows linearly with denoising progress.
    *   **High band** — trust follows a φ⁻¹-curved ramp so detail is only
        incorporated once sufficient structure exists.

    ``spectral_weight`` controls how far the result deviates from the vanilla
    prediction (0 = no effect, 1 = full spectral gating).
    """
    if spectral_weight <= 0:
        return denoised

    h, w = denoised.shape[-2:]
    device = denoised.device

    progress = max(0.0, min(1.0, 1.0 - sigma / (sigma_max + 1e-8)))

    freq = fft.fft2(denoised)
    low_mask, mid_mask, high_mask = create_frequency_masks(
        h, w, device, low_cutoff, high_cutoff
    )
    low_mask  = _expand_mask(low_mask,  freq)
    mid_mask  = _expand_mask(mid_mask,  freq)
    high_mask = _expand_mask(high_mask, freq)

    # Adaptive per-band trust
    low_trust  = 1.0
    mid_trust  = 0.4 + 0.6 * progress
    high_trust = max(0.1, progress ** PHI_INV)        # ≈ progress^0.618

    weight_map = low_mask * low_trust + mid_mask * mid_trust + high_mask * high_trust
    weighted_freq = freq * (1.0 - spectral_weight + spectral_weight * weight_map)

    return fft.ifft2(weighted_freq).real


# ═══════════════════════════════════════════════════════════════════════════
#  Stochastic Resonance
# ═══════════════════════════════════════════════════════════════════════════

def stochastic_resonance(
    x: torch.Tensor,
    sigma_next: float,
    progress: float,
    strength: float = 0.03,
    peak: float = 0.45,
) -> torch.Tensor:
    """
    Inject bandpass-shaped noise whose amplitude peaks at *peak* progress.

    The noise is **not** white: an FFT bandpass filter suppresses DC and the
    highest frequencies so energy lands in the mid-band where weak detail
    signals live.  This mirrors physical stochastic resonance — the right
    amount of noise makes faint features *easier* to detect.
    """
    if strength <= 0 or sigma_next <= 0:
        return x

    envelope = math.exp(-((progress - peak) ** 2) / (2 * 0.15 ** 2))
    noise_scale = strength * envelope * sigma_next
    if noise_scale < 1e-8:
        return x

    noise = torch.randn_like(x)

    h, w = noise.shape[-2:]
    if h > 4 and w > 4:
        nf = fft.fft2(noise)
        freq_y = torch.fft.fftfreq(h, device=x.device)
        freq_x = torch.fft.fftfreq(w, device=x.device)
        fy, fx = torch.meshgrid(freq_y, freq_x, indexing="ij")
        fd = torch.sqrt(fy ** 2 + fx ** 2)
        mx = fd.max()
        if mx > 0:
            fd = fd / mx
        # Bandpass centred at 0.3 normalised frequency
        bp = torch.exp(-((fd - 0.3) ** 2) / (2 * 0.2 ** 2))
        bp = _expand_mask(bp, nf)
        nf = nf * (0.3 + 0.7 * bp)
        noise = fft.ifft2(nf).real
        noise = noise / (noise.std() + 1e-8)

    return x + noise * noise_scale


# ═══════════════════════════════════════════════════════════════════════════
#  Quantum Spectral Flow Sampler  (main entry point)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def quantum_spectral_flow_sample(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: Optional[dict] = None,
    callback=None,
    disable=None,
    # ── tunables ──
    spectral_weight: float = 0.5,
    momentum: float = 0.6,
    resonance_strength: float = 0.03,
    low_cutoff: float = 0.15,
    high_cutoff: float = 0.55,
    eta: float = 0.0,
    resonance_peak: float = 0.45,
    correction_strength: float = 0.0,
) -> torch.Tensor:
    """
    Quantum Spectral Flow (QSF) — a diffusion sampler that fuses:

    * Frequency-domain trust weighting (spectral guidance)
    * Polyak-style trajectory momentum
    * Physics-inspired stochastic resonance
    * Optional ancestral noise (η > 0) and Langevin correction

    Parameters
    ----------
    spectral_weight : float   0→1  How strongly to apply spectral gating.
    momentum        : float   0→1  EMA decay for denoising direction.
    resonance_strength : float      Amplitude of resonance noise injection.
    low_cutoff / high_cutoff : float  Frequency band boundaries (normalised).
    eta             : float   0→1  Ancestral noise fraction (0 = deterministic).
    resonance_peak  : float   0→1  Progress point where resonance is strongest.
    correction_strength : float     Langevin correction step size (0 = off).
    """
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    sigma_max = float(sigmas[0])
    n_steps = len(sigmas) - 1

    momentum_buffer = None

    for i in range(n_steps):
        sigma      = sigmas[i]
        sigma_next = sigmas[i + 1]

        # ── Model prediction ────────────────────────────────────────────
        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({
                "x": x, "i": i, "sigma": sigma,
                "sigma_hat": sigma, "denoised": denoised,
            })

        # ── 1. Spectral trust weighting ─────────────────────────────────
        processed = spectral_guided_blend(
            denoised, float(sigma), sigma_max,
            spectral_weight=spectral_weight,
            low_cutoff=low_cutoff, high_cutoff=high_cutoff,
        )

        # ── 2. Trajectory momentum ─────────────────────────────────────
        if momentum > 0 and momentum_buffer is not None:
            processed = momentum * momentum_buffer + (1.0 - momentum) * processed
        if momentum > 0:
            momentum_buffer = processed.clone()

        # ── 3. Step (Euler or Ancestral-Euler) ──────────────────────────
        d = (x - processed) / sigma                       # score estimate

        if eta > 0 and float(sigma_next) > 0:
            sigma_up = min(
                float(sigma_next),
                eta * math.sqrt(
                    float(sigma_next) ** 2
                    * (float(sigma) ** 2 - float(sigma_next) ** 2)
                    / (float(sigma) ** 2 + 1e-12)
                ),
            )
            sigma_down = math.sqrt(float(sigma_next) ** 2 - sigma_up ** 2)
            x = x + d * (sigma_down - float(sigma))
            if sigma_up > 0:
                x = x + torch.randn_like(x) * sigma_up
        else:
            dt = sigma_next - sigma
            x = x + d * dt

        # ── 4. Stochastic resonance ────────────────────────────────────
        progress = i / max(1, n_steps - 1)
        x = stochastic_resonance(
            x, float(sigma_next), progress,
            strength=resonance_strength, peak=resonance_peak,
        )

        # ── 5. Langevin correction (optional) ──────────────────────────
        if correction_strength > 0 and float(sigma_next) > 0 and i < n_steps - 1:
            corr_denoised = model(x, sigma_next * s_in, **extra_args)
            corr_d = (x - corr_denoised) / sigma_next
            step = correction_strength * float(sigma_next)
            x = x - step * corr_d + math.sqrt(2.0 * step) * torch.randn_like(x)

    return x


# ═══════════════════════════════════════════════════════════════════════════
#  SCHEDULERS
# ═══════════════════════════════════════════════════════════════════════════

def phi_harmonic_sigmas(
    n: int,
    sigma_min: float = 0.0292,
    sigma_max: float = 14.6146,
    phi_power: float = 1.0,
    dual_phase: bool = True,
) -> torch.Tensor:
    """
    Phi-Harmonic Sigma Scheduler
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Uses the golden ratio φ to distribute sigma steps optimally.

    **Single-phase** mode is a Karras-style schedule with ρ = φ^power,
    shifting step density toward the structure-formation regime.

    **Dual-phase** mode splits the schedule at the golden-ratio point:
      * Phase 1 (upper φ⁻¹ of sigmas, ~62 % of steps) — Karras with ρ=φ
        for coarse structure.
      * Phase 2 (lower 1−φ⁻¹ of sigmas, ~38 % of steps) — cosine–linear
        blend weighted by φ⁻¹ for smooth detail refinement.

    This two-regime design matches how diffusion models actually work:
    structure crystallises in the first regime, then detail fills in.
    """
    if n <= 0:
        return torch.tensor([sigma_max, 0.0])
    if n == 1:
        return torch.tensor([sigma_max, 0.0])

    phi = PHI ** phi_power

    if not dual_phase:
        ramp = torch.linspace(0, 1, n)
        s = (sigma_max ** (1.0 / phi) + ramp * (sigma_min ** (1.0 / phi) - sigma_max ** (1.0 / phi))) ** phi
        return torch.cat([s, torch.zeros(1)])

    # ── Dual-phase ──────────────────────────────────────────────────────
    split = max(1, min(n - 1, round(n * PHI_INV)))      # ~62 % of steps
    n1, n2 = split, n - split

    log_min = math.log(max(sigma_min, 1e-6))
    log_max = math.log(sigma_max)
    log_mid = log_max - (log_max - log_min) / PHI
    sigma_mid = math.exp(log_mid)

    # Phase 1 — Karras with ρ=φ (structure)
    r1 = torch.linspace(0, 1, n1)
    phase1 = (sigma_max ** (1.0 / phi) + r1 * (sigma_mid ** (1.0 / phi) - sigma_max ** (1.0 / phi))) ** phi

    # Phase 2 — cosine–linear golden blend (detail)
    r2 = torch.linspace(0, 1, n2)
    cos_t = (1.0 + torch.cos(r2 * math.pi)) / 2.0
    lin_t = 1.0 - r2
    blend_t = PHI_INV * cos_t + (1.0 - PHI_INV) * lin_t
    phase2 = sigma_min + (sigma_mid - sigma_min) * blend_t

    sigmas = torch.cat([phase1, phase2, torch.zeros(1)])
    return sigmas


def fibonacci_adaptive_sigmas(
    n: int,
    sigma_min: float = 0.0292,
    sigma_max: float = 14.6146,
) -> torch.Tensor:
    """
    Fibonacci Adaptive Scheduler
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Step spacing is governed by consecutive Fibonacci-number ratios
    (which converge to φ).  Early steps are wider, late steps tighter —
    naturally matching the diminishing-returns curve of diffusion detail.
    """
    if n <= 1:
        return torch.tensor([sigma_max, 0.0])

    fib = [1, 1]
    while len(fib) < n + 2:
        fib.append(fib[-1] + fib[-2])

    # Inverse Fibonacci ratios → step weights (larger ratio → smaller step)
    weights = [1.0 / (fib[i + 1] / fib[i]) for i in range(n)]
    cumulative = [0.0]
    for w in weights:
        cumulative.append(cumulative[-1] + w)
    total = cumulative[-1]
    positions = torch.tensor([c / total for c in cumulative[1:]])

    log_min = math.log(max(sigma_min, 1e-6))
    log_max = math.log(sigma_max)
    log_sigmas = log_max + positions * (log_min - log_max)
    sigmas = torch.exp(log_sigmas)
    return torch.cat([sigmas, torch.zeros(1)])


def turbo_phi_sigmas(
    n: int,
    sigma_min: float = 0.0292,
    sigma_max: float = 14.6146,
    shift: float = 3.0,
) -> torch.Tensor:
    """
    Turbo-Phi Scheduler
    ~~~~~~~~~~~~~~~~~~~
    Designed for distilled / turbo models (1-8 steps).
    Combines flow-matching timestep shifting with a golden-ratio warp
    so that each precious step carries maximum information gain.
    """
    if n <= 0:
        return torch.tensor([sigma_max, 0.0])

    t = torch.linspace(1, 0, n + 1)
    # Flow-matching shift (concentrates steps at higher noise)
    t_shifted = (shift * t) / (1.0 + (shift - 1.0) * t)
    # Golden-ratio warp on top
    t_warped = t_shifted ** (1.0 / PHI)
    sigmas = t_warped * sigma_max
    sigmas[-1] = 0.0
    return sigmas


# ═══════════════════════════════════════════════════════════════════════════
#  Spectral Latent Enhancement  (post-processing)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def spectral_latent_enhance(
    latent: torch.Tensor,
    detail_boost: float = 0.3,
    denoise_high: float = 0.2,
    low_cutoff: float = 0.12,
    high_cutoff: float = 0.60,
) -> torch.Tensor:
    """
    Frequency-domain post-processing of the decoded latent:

    * Boosts mid-frequency energy (fine features, edges)
    * Gently attenuates the highest frequencies (residual noise)
    * Leaves low-frequency structure untouched
    """
    h, w = latent.shape[-2:]
    if h <= 4 or w <= 4:
        return latent

    freq = fft.fft2(latent)
    low_m, mid_m, high_m = create_frequency_masks(h, w, latent.device, low_cutoff, high_cutoff)
    low_m  = _expand_mask(low_m,  freq)
    mid_m  = _expand_mask(mid_m,  freq)
    high_m = _expand_mask(high_m, freq)

    gain = low_m * 1.0 + mid_m * (1.0 + detail_boost) + high_m * (1.0 - denoise_high)
    return fft.ifft2(freq * gain).real
