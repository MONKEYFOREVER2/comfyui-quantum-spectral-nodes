"""
Quantum Spectral Flow — Custom ComfyUI Nodes
=============================================

A novel diffusion sampling toolkit that uses frequency-domain analysis,
golden-ratio scheduling, and stochastic resonance to improve image
generation quality — especially for turbo / distilled models with few steps.

Nodes provided
--------------
*  **Quantum Spectral Sampler**  — SAMPLER with spectral trust gating,
   trajectory momentum, and stochastic resonance.
*  **Phi-Harmonic Scheduler**    — SIGMAS built on golden-ratio (φ)
   mathematics with four algorithm modes.
*  **Quantum Flow KSampler**     — All-in-one convenience node.
*  **Spectral Latent Enhance**   — Frequency-domain latent post-processor.

Installation
------------
Copy (or symlink) this folder into ``ComfyUI/custom_nodes/`` and restart.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None   # no custom JS widgets required
