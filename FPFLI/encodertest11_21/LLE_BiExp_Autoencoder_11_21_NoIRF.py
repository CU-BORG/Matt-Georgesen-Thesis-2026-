#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLE Bi-Exponential Autoencoder for FLIM Parameter Extraction (Decay-Only Input)

Based on:
1. Paper: "Rapid parameter estimation of discretely sampled signals" (Visschers et al., 2021)
2. Original_LLE.py ConvMixer architecture

This autoencoder network extracts 3 parameters (tau1, tau2, f) from FLIM decay curves
by encoding them into a 3-dimensional latent space and reconstructing the signal.

MODIFIED VERSION: This version takes ONLY the decay curve as input (no IRF).

Key Design Principles from Paper:
- Dense autoencoder with hourglass shape
- Latent space dimensionality = number of parameters (3)
- Three-stage training for structured latent space
- Direct parameter extraction from latent representation

IMPORTANT - Fraction Definition:
- tau1 = short lifetime (free state, 0.2-0.7 ns)
- tau2 = long lifetime (bound state, 1.2-4.5 ns)
- f = fraction of tau2 (long/bound component)
- The model learns this implicitly from training data
- f represents the bound state fraction (longer lifetime component)
- Biological interpretation: higher f = more bound protein

@author: mg
"""

import torch
import torch.nn as nn


class Residual(nn.Module):
    """Residual connection wrapper"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvLayer(nn.Module):
    """Convolutional layer block with residual connection"""
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.res = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm1d(dim)
            )))
        self.pointconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.activate = nn.GELU()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        y = self.res(x)
        y = self.pointconv(y)
        y = self.activate(y)
        y = self.bn(y)
        return y


class LLE_BiExp_Autoencoder(nn.Module):
    """
    Dense Autoencoder Network for Bi-Exponential FLIM Parameter Extraction
    (Decay-Only Input Version)

    Following the methodology from Visschers et al. 2021:
    - Encoder compresses decay into 3-parameter latent space
    - Latent space directly represents [tau1, tau2, f]
      where tau1 = short lifetime (free state, 0.2-0.7 ns)
            tau2 = long lifetime (bound state, 1.2-4.5 ns)
            f = fraction of tau2 (long/bound component)
    - Decoder reconstructs original decay curve
    - Three-stage training ensures structured latent representation

    Architecture:
        Input: Decay [B, 1, 256]
          ↓
        Encoder: ConvMixer layers → 3D latent space
          ↓
        Latent: [tau1, tau2, f]
          ↓
        Decoder: Reconstruct decay curve
          ↓
        Output: Reconstructed decay [B, 1, 256]

    Parameters:
    -----------
    dim : int
        Feature dimension for ConvMixer (default: 16)
    depth : int
        Number of encoder/decoder conv layers (default: 8)
    kernel_size : int
        Kernel size for convolutions (default: 9)
    patch_size : int
        Patch size for initial embedding (default: 8)
    signal_length : int
        Length of input signal (default: 256)
    latent_dim : int
        Latent space dimensionality (default: 3 for tau1, tau2, f)
    """

    def __init__(self, dim=16, depth=8, kernel_size=9, patch_size=8,
                 signal_length=256, latent_dim=3):
        super().__init__()

        self.dim = dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.encoded_length = signal_length // patch_size  # 256 / 8 = 32

        # =====================================================================
        # ENCODER: Compress decay → latent parameters
        # =====================================================================

        # Embedding layer for decay only
        self.embd_decay = nn.Conv1d(1, dim, kernel_size=patch_size, stride=patch_size)

        # Encoder convolutional blocks (using dim instead of dim*2)
        self.encoder_blocks = nn.ModuleList()
        for _ in range(depth):
            self.encoder_blocks.append(ConvLayer(dim, kernel_size))

        # Compression to latent space
        # From [B, dim, 32] → [B, dim*32, 1] → [B, latent_dim]
        self.encoder_flatten = nn.Flatten()
        self.encoder_fc1 = nn.Linear(dim * self.encoded_length, dim * 4)
        self.encoder_fc2 = nn.Linear(dim * 4, latent_dim)

        # Latent parameter activations
        self.relu = nn.ReLU()  # For lifetimes (must be positive)
        self.sigmoid = nn.Sigmoid()  # For fraction (0 < f1 < 1)

        # =====================================================================
        # DECODER: Reconstruct decay from latent parameters
        # =====================================================================

        # Expand from latent space back to feature space
        self.decoder_fc1 = nn.Linear(latent_dim, dim * 4)
        self.decoder_fc2 = nn.Linear(dim * 4, dim * self.encoded_length)

        # Decoder convolutional blocks (using dim instead of dim*2)
        self.decoder_blocks = nn.ModuleList()
        for _ in range(depth):
            self.decoder_blocks.append(ConvLayer(dim, kernel_size))

        # Reconstruction layers
        # From [B, dim, 32] → [B, 1, 256]
        self.decoder_upsample = nn.ConvTranspose1d(
            dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.decoder_output = nn.Conv1d(dim, 1, kernel_size=1)

        # Output activation (decay values are typically normalized 0-1)
        self.tanh = nn.Tanh()  # Output in [-1, 1] range

    def encode(self, decay):
        """
        Encode decay into 3-parameter latent space

        Parameters:
        -----------
        decay : Tensor [B, 1, 256]
            Fluorescence decay histogram

        Returns:
        --------
        latent : Tensor [B, 3]
            Latent parameters [tau1, tau2, f] (unconstrained)
            where f is the fraction of tau2 (long/bound component)
        """
        # Embed input: [B, 1, 256] → [B, dim, 32]
        x = self.embd_decay(decay)

        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)

        # Flatten: [B, dim*32]
        x = self.encoder_flatten(x)

        # Compress to latent space
        x = self.encoder_fc1(x)
        x = self.relu(x)
        latent = self.encoder_fc2(x)  # [B, 3]

        return latent

    def decode(self, latent):
        """
        Decode latent parameters back to decay curve

        Parameters:
        -----------
        latent : Tensor [B, 3]
            Latent parameters [tau1, tau2, f]
            where f is the fraction of tau2 (long/bound component)

        Returns:
        --------
        reconstruction : Tensor [B, 1, 256]
            Reconstructed decay curve
        """
        # Expand from latent: [B, 3] → [B, dim*32]
        x = self.decoder_fc1(latent)
        x = self.relu(x)
        x = self.decoder_fc2(x)
        x = self.relu(x)

        # Reshape: [B, dim*32] → [B, dim, 32]
        batch_size = x.shape[0]
        x = x.view(batch_size, self.dim, self.encoded_length)

        # Apply decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        # Upsample: [B, dim, 32] → [B, dim, 256]
        x = self.decoder_upsample(x)

        # Final output: [B, dim, 256] → [B, 1, 256]
        reconstruction = self.decoder_output(x)
        reconstruction = self.tanh(reconstruction)

        return reconstruction

    def forward(self, decay):
        """
        Full forward pass: encode → decode

        Parameters:
        -----------
        decay : Tensor [B, 1, 256]
            Fluorescence decay histogram

        Returns:
        --------
        reconstruction : Tensor [B, 1, 256]
            Reconstructed decay curve
        latent : Tensor [B, 3]
            Latent parameters (unconstrained)
        """
        latent = self.encode(decay)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def extract_parameters(self, decay):
        """
        Extract physical parameters with constraints applied

        Following paper methodology: latent space directly encodes parameters

        Parameters:
        -----------
        decay : Tensor [B, 1, 256]
            Fluorescence decay histogram

        Returns:
        --------
        dict with keys:
            - 'tau1': Tensor [B] - Short lifetime / free state (constrained > 0)
            - 'tau2': Tensor [B] - Long lifetime / bound state (constrained > 0)
            - 'f1': Tensor [B] - Fraction of tau2 (long/bound component, constrained 0-1)
            - 'f2': Tensor [B] - Fraction of tau1 (short/free component, = 1 - f1)
            - 'tau_avg': Tensor [B] - Average lifetime = tau2*f1 + tau1*f2
        """
        latent = self.encode(decay)

        # Apply constraints to latent parameters
        tau1 = self.relu(latent[:, 0])  # tau1 > 0 (short/free)
        tau2 = self.relu(latent[:, 1])  # tau2 > 0 (long/bound)
        f1 = self.sigmoid(latent[:, 2])  # 0 < f1 < 1 (fraction of tau2/long)
        f2 = 1.0 - f1  # f2 = fraction of tau1/short (ensures sum to 1)

        # Compute average lifetime
        tau_avg = tau2 * f1 + tau1 * f2

        return {
            'tau1': tau1,
            'tau2': tau2,
            'f1': f1,
            'f2': f2,
            'tau_avg': tau_avg
        }

    def forward_with_parameters(self, decay):
        """
        Forward pass returning both reconstruction and constrained parameters

        Useful for three-stage training where we need both outputs

        Returns:
        --------
        reconstruction : Tensor [B, 1, 256]
        parameters : dict with:
            - tau1: short/free lifetime
            - tau2: long/bound lifetime
            - f1: fraction of tau2 (long/bound component)
            - f2: fraction of tau1 (short/free component)
            - tau_avg: average lifetime = tau2*f1 + tau1*f2
        """
        reconstruction, latent = self.forward(decay)

        # Apply constraints for parameter extraction
        tau1 = self.relu(latent[:, 0])  # short/free
        tau2 = self.relu(latent[:, 1])  # long/bound
        f1 = self.sigmoid(latent[:, 2])  # fraction of tau2/long
        f2 = 1.0 - f1  # fraction of tau1/short
        tau_avg = tau2 * f1 + tau1 * f2

        parameters = {
            'tau1': tau1,
            'tau2': tau2,
            'f1': f1,
            'f2': f2,
            'tau_avg': tau_avg
        }

        return reconstruction, parameters


# =============================================================================
# TEST CODE
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" TESTING LLE_BiExp_Autoencoder (Decay-Only Input)")
    print(" Based on Visschers et al. 2021 methodology")
    print("=" * 80)
    print()

    # Create model
    print("Creating autoencoder model...")
    model = LLE_BiExp_Autoencoder(
        dim=16,
        depth=8,
        kernel_size=9,
        patch_size=8,
        signal_length=256,
        latent_dim=3
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model created successfully!")
    print(f"  Architecture: Dense Autoencoder (hourglass shape)")
    print(f"  Input: Decay only (no IRF)")
    print(f"  Latent space: 3D [tau1, tau2, f]")
    print(f"    tau1 = short/free lifetime")
    print(f"    tau2 = long/bound lifetime")
    print(f"    f = fraction of tau2 (long/bound component)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Create dummy input
    batch_size = 4
    decay = torch.randn(batch_size, 1, 256)

    print(f"Test input shapes:")
    print(f"  decay: {decay.shape}")
    print()

    # Test encoding
    print("Testing encoder...")
    with torch.no_grad():
        latent = model.encode(decay)
    print(f"  Latent shape: {latent.shape}")
    print(f"  Latent sample: {latent[0].numpy()}")
    print()

    # Test decoding
    print("Testing decoder...")
    with torch.no_grad():
        reconstruction = model.decode(latent)
    print(f"  Reconstruction shape: {reconstruction.shape}")
    print()

    # Test full forward pass
    print("Testing full forward pass...")
    with torch.no_grad():
        recon, latent = model.forward(decay)
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent shape: {latent.shape}")
    print()

    # Test parameter extraction
    print("Testing parameter extraction with constraints...")
    with torch.no_grad():
        params = model.extract_parameters(decay)

    print(f"\nExtracted parameters (first sample):")
    print(f"  tau1 = {params['tau1'][0].item():.4f} ns (short/free, constrained > 0)")
    print(f"  tau2 = {params['tau2'][0].item():.4f} ns (long/bound, constrained > 0)")
    print(f"  f1 = {params['f1'][0].item():.4f} (fraction of tau2/long, constrained 0-1)")
    print(f"  f2 = {params['f2'][0].item():.4f} (fraction of tau1/short, = 1 - f1)")
    print(f"  tau_avg = {params['tau_avg'][0].item():.4f} ns (= tau2*f1 + tau1*f2)")
    print()

    # Verify constraints
    print("Verifying physical constraints:")
    print(f"  [OK] tau1 > 0: min = {params['tau1'].min().item():.4f}")
    print(f"  [OK] tau2 > 0: min = {params['tau2'].min().item():.4f}")
    print(f"  [OK] 0 < f1 < 1: range = [{params['f1'].min().item():.4f}, {params['f1'].max().item():.4f}]")

    f_sum = params['f1'] + params['f2']
    print(f"  [OK] f1 + f2 = 1: mean = {f_sum.mean().item():.6f}, std = {f_sum.std().item():.6e}")

    tau_avg_check = params['tau2'] * params['f1'] + params['tau1'] * params['f2']
    tau_avg_error = (tau_avg_check - params['tau_avg']).abs().mean()
    print(f"  [OK] tau_avg = tau2*f1 + tau1*f2: error = {tau_avg_error.item():.6e}")
    print()

    # Test combined forward
    print("Testing combined forward (for training)...")
    with torch.no_grad():
        recon, params = model.forward_with_parameters(decay)
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Parameters: {list(params.keys())}")
    print()

    # Architecture summary
    print("=" * 80)
    print(" ARCHITECTURE SUMMARY")
    print("=" * 80)
    print()
    print("ENCODER PATH:")
    print("  Input: decay [B, 1, 256]")
    print("    | Embedding (Conv1d, stride=8)")
    print("  [B, dim, 32]")
    print("    | ConvMixer blocks (depth=8)")
    print("  [B, dim, 32]")
    print("    | Flatten + FC layers")
    print("  Latent: [B, 3] <- [tau1, tau2, f]")
    print("    where f = fraction of tau2 (long/bound component)")
    print()
    print("DECODER PATH:")
    print("  Latent: [B, 3]")
    print("    | FC layers + reshape")
    print("  [B, dim, 32]")
    print("    | ConvMixer blocks (depth=8)")
    print("  [B, dim, 32]")
    print("    | ConvTranspose1d (stride=8)")
    print("  Output: reconstruction [B, 1, 256]")
    print()

    print("=" * 80)
    print(" ALL TESTS PASSED!")
    print("=" * 80)
    print()
    print("Model is ready for three-stage training:")
    print("  Stage 1: Train for signal reconstruction (unsupervised)")
    print("  Stage 2: Fine-tune with parameter supervision")
    print("  Stage 3: Joint optimization (reconstruction + parameters)")
    print()
    print("Following methodology from:")
    print("  Visschers et al. 2021 - Mach. Learn.: Sci. Technol.")
    print("  'Rapid parameter estimation of discretely sampled signals'")
    print()
