# ComfyUI Quantum Spectral Nodes

This Custom Node package provides the **Spectral Latent Enhance** node for ComfyUI. 

## Features
- **Spectral Latent Enhance**: A frequency-domain post-processor applied to the latent *before* VAE decode. It boosts mid-band detail while gently attenuating high-frequency noise. This can be used with any sampler and drastically improves the sharpness and quality of generated images.

## Installation

### Method 1: ComfyUI Manager
1. Open ComfyUI and click **Manager**.
2. Click **Install via Git URL**.
3. Paste: `https://github.com/MONKEYFOREVER2/comfyui-quantum-spectral-nodes`
4. Restart ComfyUI.

### Method 2: Manual (Git Clone)
1. Open a terminal in your `ComfyUI/custom_nodes/` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/MONKEYFOREVER2/comfyui-quantum-spectral-nodes.git
   ```
2. Restart ComfyUI.

## Usage
Simply drop the `Spectral Latent Enhance` node into your workflow just before your `VAE Decode` node:
- Connect the output of your `KSampler` (the `LATENT` output) into the node.
- Tweak the `detail_boost` and `denoise_high` values to taste.
- Connect the output `LATENT` to your `VAE Decode` node.
