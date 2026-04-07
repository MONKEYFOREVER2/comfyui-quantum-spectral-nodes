"""
Quantum Spectral Flow — Custom ComfyUI Nodes
=============================================

A novel diffusion sampling toolkit that uses frequency-domain analysis,
golden-ratio scheduling, and stochastic resonance to improve image
generation quality — especially for turbo / distilled models with few steps.

Nodes provided
--------------
*  **Spectral Latent Enhance**   — Frequency-domain latent post-processor.

Installation
------------
Copy (or symlink) this folder into ``ComfyUI/custom_nodes/`` and restart.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = None   # no custom JS widgets required
