#!/usr/bin/env python3
"""
run_tquant.py — Standalone TurboQuant KV Cache Inference for https://huggingface.co/g023/Qwen3-1.77B-g023
Author: g023 (https://github.com/g023) (https://huggingface.co/g023) 

Implements TurboQuant (ICLR 2026, arXiv:2504.19874) KV cache compression
directly inside a Transformers inference script. All algorithms are self-contained. Minimal dependencies. 

Algorithms:
  1. Random orthogonal rotation (QR decomposition) → Beta-distributed coordinates
  2. Lloyd-Max optimal scalar quantization → MSE-optimal centroids
  3. QJL sign-bit residual correction → unbiased inner products
  4. Group quantization for values → per-group min-max

Model Repo: https://huggingface.co/g023/Qwen3-1.77B-g023
Model Info: (head_dim=128, 8 KV heads, 29 layers, GQA)
Model Instructions: Download files from 'MODEL REPO' and throw in ./Qwen3-BEST and then run this program. 

Reqs:
pip install transformers datasets scipy

"""

import math
import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from io import StringIO
from typing import Optional, NamedTuple
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicLayer

# force clear cache
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = "./Qwen3-BEST"
MAX_NEW_TOKENS = 8192
TEMPERATURE = 0.6
DO_SAMPLE = True
TOP_P = 0.95
TOP_K = 20
REPETITION_PENALTY = 1.1
INPUT_MESSAGE = (
    "You are completing the next step in a task to create an arcade game in javascript. "
    "Your available tools are rationalize, red_green_tdd, and create_plan. "
    "Synthesize their output when reasoning."
)

# TurboQuant parameters
TQ_KEY_BITS = 4            # 3 MSE + 1 QJL (near-lossless for head_dim=128)
TQ_VALUE_BITS = 4          # 4-bit for quality (cos_sim≈0.997)
TQ_BUFFER_SIZE = 256       # Recent tokens kept in full precision
TQ_VALUE_GROUP_SIZE = 32   # Group size for value quantization
TQ_SKIP_FIRST_LAYERS = 3   # Keep first N layers full precision (critical for attention)
TQ_SKIP_LAST_LAYERS = 2    # Keep last N layers full precision (critical for output)
CODEBOOK_DIR = "./codebooks"


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Lloyd-Max Codebook
# ═══════════════════════════════════════════════════════════════════════════════

def _beta_pdf(x, d):
    """PDF of a single coordinate of a uniform point on S^{d-1}."""
    from scipy import special
    log_const = (
        special.gammaln(d / 2.0)
        - 0.5 * np.log(np.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0
    x = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    return np.exp(log_const + exponent * np.log(1 - x**2))


def _compute_lloyd_max_codebook(d, bits, max_iter=200, tol=1e-12):
    """Compute optimal Lloyd-Max codebook for Beta distribution on [-1, 1]."""
    from scipy import integrate

    n_clusters = 2 ** bits

    # Initialize centroids at quantiles of the distribution
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, 10000)
    pdf_vals = _beta_pdf(x_grid, d)
    cdf_vals = np.cumsum(pdf_vals) * (x_grid[1] - x_grid[0])
    cdf_vals /= cdf_vals[-1]

    quantile_edges = np.linspace(0, 1, n_clusters + 1)
    centroids = np.zeros(n_clusters)
    for i in range(n_clusters):
        q_mid = (quantile_edges[i] + quantile_edges[i + 1]) / 2.0
        idx = min(np.searchsorted(cdf_vals, q_mid), len(x_grid) - 1)
        centroids[i] = x_grid[idx]

    # Lloyd-Max iterations
    for _ in range(max_iter):
        boundaries = np.zeros(n_clusters + 1)
        boundaries[0], boundaries[-1] = -1.0, 1.0
        for i in range(n_clusters - 1):
            boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

        new_centroids = np.zeros(n_clusters)
        for i in range(n_clusters):
            lo, hi = boundaries[i], boundaries[i + 1]
            num, _ = integrate.quad(
                lambda x: x * _beta_pdf(np.array([x]), d)[0], lo, hi
            )
            den, _ = integrate.quad(
                lambda x: _beta_pdf(np.array([x]), d)[0], lo, hi
            )
            new_centroids[i] = num / den if den > 1e-30 else (lo + hi) / 2.0

        if np.sum((new_centroids - centroids) ** 2) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final boundaries
    boundaries = np.zeros(n_clusters + 1)
    boundaries[0], boundaries[-1] = -1.0, 1.0
    for i in range(n_clusters - 1):
        boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0

    return {"centroids": centroids.tolist(), "boundaries": boundaries.tolist()}


def get_codebook_tensors(d, bits, device, dtype=torch.float32):
    """Load or compute codebook, return (centroids, decision_boundaries) as tensors."""
    os.makedirs(CODEBOOK_DIR, exist_ok=True)
    path = os.path.join(CODEBOOK_DIR, f"codebook_d{d}_b{bits}.json")

    if os.path.exists(path):
        with open(path, "r") as f:
            cb = json.load(f)
    else:
        print(f"[TurboQuant] Computing Lloyd-Max codebook d={d}, bits={bits}...")
        cb = _compute_lloyd_max_codebook(d, bits)
        with open(path, "w") as f:
            json.dump(cb, f, indent=2)
        print(f"[TurboQuant] Codebook cached → {path}")

    centroids = torch.tensor(cb["centroids"], device=device, dtype=dtype)
    boundaries = torch.tensor(cb["boundaries"], device=device, dtype=dtype)
    decision_boundaries = boundaries[1:-1].contiguous()
    return centroids, decision_boundaries


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Rotation & QJL Matrices
# ═══════════════════════════════════════════════════════════════════════════════

def generate_rotation_matrix(d, device, dtype=torch.float32, seed=42):
    """Random orthogonal matrix Π via QR decomposition (Algorithm 1)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    G = torch.randn(d, d, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    Q = Q * torch.sign(torch.diag(R)).unsqueeze(0)  # ensure proper rotation
    return Q.to(device=device, dtype=dtype)


def generate_qjl_matrix(d, device, dtype=torch.float32, seed=12345):
    """Random Gaussian projection S for QJL (i.i.d. N(0,1) entries)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    S = torch.randn(d, d, generator=rng, dtype=torch.float32)
    return S.to(device=device, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Bit-Packing Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def pack_indices(indices, bits):
    """Bit-pack integer indices (0..2^bits-1) into uint8 bytes."""
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return indices.to(torch.uint8)

    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(*batch_shape, -1, vals_per_byte)
    shifts = torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8) * bits
    return (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)


def unpack_indices(packed, bits, d):
    """Unpack bit-packed indices back to integer tensor."""
    if bits == 1:
        vals_per_byte = 8
    elif bits == 2:
        vals_per_byte = 4
    elif bits <= 4:
        vals_per_byte = 2
        bits = 4
    else:
        return packed.long()

    mask = (1 << bits) - 1
    shifts = torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8) * bits
    unpacked = (packed.unsqueeze(-1) >> shifts) & mask
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)
    return unpacked[..., :d].long()


def pack_signs(projected):
    """Pack boolean signs (projected > 0) into uint8 (8 signs per byte)."""
    signs = (projected > 0).to(torch.uint8)
    d = signs.shape[-1]
    batch_shape = signs.shape[:-1]
    if d % 8 != 0:
        signs = F.pad(signs, (0, 8 - d % 8), value=0)
    reshaped = signs.reshape(*batch_shape, -1, 8)
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=signs.device, dtype=torch.uint8)
    return (reshaped * powers).sum(dim=-1, dtype=torch.uint8)


def unpack_signs(packed, d):
    """Unpack sign bits from uint8 to float {-1, +1}."""
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], device=packed.device, dtype=torch.uint8)
    unpacked = ((packed.unsqueeze(-1) & powers) > 0).float()
    signs = unpacked.reshape(*packed.shape[:-1], -1)[..., :d]
    return 2.0 * signs - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Quantized Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class MSEQuantized(NamedTuple):
    """Output of TurboQuant MSE quantization (Algorithm 1)."""
    indices: torch.Tensor   # (..., packed_len) uint8 bit-packed
    norms: torch.Tensor     # (...,) original L2 norms
    bits: int


class ProdQuantized(NamedTuple):
    """Output of TurboQuant inner-product quantization (Algorithm 2)."""
    mse_indices: torch.Tensor     # (..., packed_len) uint8
    qjl_signs: torch.Tensor      # (..., packed_len) uint8
    residual_norms: torch.Tensor  # (...,) residual L2 norms
    norms: torch.Tensor           # (...,) original L2 norms
    mse_bits: int


class ValueQuantized(NamedTuple):
    """Quantized value cache (group quantization)."""
    data: torch.Tensor    # (..., packed_d) uint8 bit-packed
    scales: torch.Tensor  # (..., n_groups) scale per group
    zeros: torch.Tensor   # (..., n_groups) zero point per group
    bits: int


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: TurboQuantMSE — Algorithm 1 (MSE-optimal quantization)
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantMSE:
    """
    MSE-optimal vector quantizer.
    Quantize: y = (x/||x||) @ Pi^T, then find nearest centroid per coordinate.
    Dequantize: look up centroids, rotate back, rescale by ||x||.
    """

    def __init__(self, dim, bits, device, dtype=torch.float32, seed=42):
        self.dim = dim
        self.bits = bits
        self.Pi = generate_rotation_matrix(dim, device, dtype, seed=seed)
        self.centroids, self.decision_boundaries = get_codebook_tensors(
            dim, bits, device, dtype
        )

    def quantize(self, x):
        """x: (..., d) → MSEQuantized. All math in float32 for precision."""
        x_f = x.float()
        norms = x_f.norm(dim=-1)
        x_unit = x_f / (norms.unsqueeze(-1) + 1e-10)
        y = torch.matmul(x_unit, self.Pi.T)
        indices = torch.searchsorted(self.decision_boundaries, y.contiguous())
        packed = pack_indices(indices, self.bits)
        return MSEQuantized(indices=packed, norms=norms, bits=self.bits)

    def dequantize(self, q):
        """MSEQuantized → (..., d) float32 reconstructed tensor"""
        indices = unpack_indices(q.indices, q.bits, self.dim)
        y_hat = self.centroids[indices]
        x_hat = torch.matmul(y_hat, self.Pi)
        return x_hat * q.norms.float().unsqueeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: TurboQuantProd — Algorithm 2 (unbiased inner-product quantization)
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantProd:
    """
    Two-stage unbiased inner-product quantizer:
      Stage 1: MSE quantization at (b-1) bits
      Stage 2: QJL sign bits on residual (1 bit per dimension)
    Result: E[estimated inner product] = true inner product
    """

    def __init__(self, dim, bits=3, device=None, dtype=torch.float32, seed=42):
        assert bits >= 2, "Need at least 2 bits (1 for MSE + 1 for QJL)"
        self.dim = dim
        self.bits = bits
        self.device = device or torch.device("cuda")

        self.mse_quantizer = TurboQuantMSE(
            dim, bits - 1, self.device, dtype, seed=seed
        )
        self.S = generate_qjl_matrix(dim, self.device, dtype, seed=seed + 1000)
        self.qjl_scale = math.sqrt(math.pi / 2.0) / dim

    def quantize(self, x):
        """x: (..., d) → ProdQuantized"""
        mse_q = self.mse_quantizer.quantize(x)
        x_hat = self.mse_quantizer.dequantize(mse_q)

        residual = x.float() - x_hat
        residual_norms = residual.norm(dim=-1)

        projected = torch.matmul(residual, self.S.T)
        packed_signs = pack_signs(projected)

        return ProdQuantized(
            mse_indices=mse_q.indices,
            qjl_signs=packed_signs,
            residual_norms=residual_norms,
            norms=mse_q.norms,
            mse_bits=mse_q.bits,
        )

    def dequantize(self, q):
        """ProdQuantized → (..., d) reconstructed tensor"""
        mse_q = MSEQuantized(indices=q.mse_indices, norms=q.norms, bits=q.mse_bits)
        x_mse = self.mse_quantizer.dequantize(mse_q)

        signs = unpack_signs(q.qjl_signs, self.dim)
        x_qjl = torch.matmul(signs, self.S)
        x_qjl = x_qjl * (self.qjl_scale * q.residual_norms.unsqueeze(-1))

        return x_mse + x_qjl


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Value Quantization (group min-max)
# ═══════════════════════════════════════════════════════════════════════════════

def quantize_values(v, bits=4, group_size=32):
    """
    Asymmetric group quantization for value vectors.
    v: (..., seq_len, d) → ValueQuantized
    """
    d = v.shape[-1]
    n_groups = d // group_size
    assert d % group_size == 0, f"head_dim {d} must be divisible by group_size {group_size}"

    v_grouped = v.float().reshape(*v.shape[:-1], n_groups, group_size)
    v_min = v_grouped.min(dim=-1, keepdim=True).values
    v_max = v_grouped.max(dim=-1, keepdim=True).values

    n_levels = 2 ** bits - 1
    scale = ((v_max - v_min) / n_levels).clamp(min=1e-10)

    v_q = ((v_grouped - v_min) / scale).round().clamp(0, n_levels).to(torch.uint8)
    v_q_flat = v_q.reshape(*v.shape[:-1], d)

    # Bit-pack
    if bits == 2:
        assert d % 4 == 0
        v4 = v_q_flat.reshape(*v.shape[:-1], d // 4, 4)
        v_q_flat = v4[..., 0] | (v4[..., 1] << 2) | (v4[..., 2] << 4) | (v4[..., 3] << 6)
    elif bits == 4:
        assert d % 2 == 0
        v2 = v_q_flat.reshape(*v.shape[:-1], d // 2, 2)
        v_q_flat = v2[..., 0] | (v2[..., 1] << 4)

    return ValueQuantized(
        data=v_q_flat,
        scales=scale.squeeze(-1).half(),
        zeros=v_min.squeeze(-1).half(),
        bits=bits,
    )


def dequantize_values(vq, group_size=32):
    """ValueQuantized → (..., seq_len, d) float tensor"""
    if vq.bits == 2:
        p = vq.data
        data = torch.stack([p & 0x03, (p >> 2) & 0x03, (p >> 4) & 0x03, (p >> 6) & 0x03], dim=-1)
        data = data.reshape(*p.shape[:-1], p.shape[-1] * 4)
    elif vq.bits == 4:
        p = vq.data
        data = torch.stack([p & 0x0F, (p >> 4) & 0x0F], dim=-1)
        data = data.reshape(*p.shape[:-1], p.shape[-1] * 2)
    else:
        data = vq.data

    d = data.shape[-1]
    n_groups = d // group_size
    data = data.float().reshape(*data.shape[:-1], n_groups, group_size)
    scales = vq.scales.float().unsqueeze(-1)
    zeros = vq.zeros.float().unsqueeze(-1)
    return (data * scales + zeros).reshape(*data.shape[:-2], d)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: ProdQuantized / ValueQuantized Concatenation
# ═══════════════════════════════════════════════════════════════════════════════

def concat_prod_q(a, b):
    """Concatenate two ProdQuantized along the sequence (dim=-2 for indices, dim=-1 for norms)."""
    return ProdQuantized(
        mse_indices=torch.cat([a.mse_indices, b.mse_indices], dim=-2),
        qjl_signs=torch.cat([a.qjl_signs, b.qjl_signs], dim=-2),
        residual_norms=torch.cat([a.residual_norms, b.residual_norms], dim=-1),
        norms=torch.cat([a.norms, b.norms], dim=-1),
        mse_bits=a.mse_bits,
    )


def concat_value_q(a, b):
    """Concatenate two ValueQuantized along the sequence dimension."""
    return ValueQuantized(
        data=torch.cat([a.data, b.data], dim=-2),
        scales=torch.cat([a.scales, b.scales], dim=-2),
        zeros=torch.cat([a.zeros, b.zeros], dim=-2),
        bits=a.bits,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: TurboQuantLayer & TurboQuantCache (transformers 5.0+ Cache API)
# ═══════════════════════════════════════════════════════════════════════════════

class TurboQuantLayer(CacheLayerMixin):
    """
    A cache layer with TurboQuant KV compression.

    - Keys: TurboQuantProd (unbiased inner-product quantization)
    - Values: Group quantization (asymmetric min-max)
    - Buffer: Recent tokens kept in full precision

    The layer transparently compresses older tokens while returning
    full-precision tensors to the attention computation.
    """

    is_sliding = False

    def __init__(self, head_dim, key_bits=4, value_bits=4,
                 buffer_size=256, value_group_size=32, seed=42):
        super().__init__()
        self.head_dim = head_dim
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.buffer_size = buffer_size
        self.value_group_size = value_group_size
        self.layer_seed = seed

        # Compressed storage (initialized on first flush)
        self._comp_keys = None
        self._comp_values = None
        self._comp_len = 0

        # Quantizer (lazy-initialized on first use)
        self._quantizer = None

    def lazy_initialization(self, key_states, value_states):
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def _get_quantizer(self, device):
        """Get or create quantizer for this layer."""
        if self._quantizer is None:
            self._quantizer = TurboQuantProd(
                dim=self.head_dim, bits=self.key_bits,
                device=device, seed=self.layer_seed,
            )
        return self._quantizer

    def update(self, key_states, value_states, cache_kwargs=None):
        """
        Append new KV states, compress if buffer overflows, return full sequence.
        Returns: (full_keys, full_values) with decompressed + buffer tokens.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # Append to buffer
        if self.keys.numel() == 0:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)

        # Flush if buffer exceeds 2x target size
        buf_len = self.keys.shape[-2]
        if buf_len > self.buffer_size * 2:
            self._flush()

        # Return full decompressed + buffer for this step's attention
        return self._full_keys(), self._full_values()

    def _flush(self):
        """Compress oldest buffer tokens into compressed storage."""
        n_flush = self.keys.shape[-2] - self.buffer_size

        to_k = self.keys[..., :n_flush, :]
        to_v = self.values[..., :n_flush, :]

        # Keep only recent tokens in buffer
        self.keys = self.keys[..., n_flush:, :].contiguous()
        self.values = self.values[..., n_flush:, :].contiguous()

        # Quantize the flushed tokens
        q = self._get_quantizer(to_k.device)
        new_kq = q.quantize(to_k)
        new_vq = quantize_values(to_v, self.value_bits, self.value_group_size)

        # Merge with existing compressed storage
        if self._comp_keys is not None:
            self._comp_keys = concat_prod_q(self._comp_keys, new_kq)
            self._comp_values = concat_value_q(self._comp_values, new_vq)
            self._comp_len += n_flush
        else:
            self._comp_keys = new_kq
            self._comp_values = new_vq
            self._comp_len = n_flush

    def _full_keys(self):
        """Decompress compressed keys + concat with buffer."""
        if self._comp_keys is None:
            return self.keys
        q = self._get_quantizer(self.keys.device)
        decompressed = q.dequantize(self._comp_keys).to(self.keys.dtype)
        return torch.cat([decompressed, self.keys], dim=-2)

    def _full_values(self):
        """Dequantize compressed values + concat with buffer."""
        if self._comp_values is None:
            return self.values
        decompressed = dequantize_values(
            self._comp_values, self.value_group_size
        ).to(self.values.dtype)
        return torch.cat([decompressed, self.values], dim=-2)

    def get_seq_length(self):
        """Total sequence length = compressed tokens + buffer tokens."""
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self._comp_len + self.keys.shape[-2]

    def get_mask_sizes(self, cache_position):
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self):
        return -1

    def crop(self, max_length):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        # For crop, decompress everything and truncate, then re-init
        # This is rare (beam search), keep simple
        if self._comp_keys is not None:
            full_k = self._full_keys()[..., :max_length, :]
            full_v = self._full_values()[..., :max_length, :]
            self.keys = full_k
            self.values = full_v
            self._comp_keys = None
            self._comp_values = None
            self._comp_len = 0
        else:
            self.keys = self.keys[..., :max_length, :]
            self.values = self.values[..., :max_length, :]

    def reset(self):
        if self.is_initialized:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self._comp_keys = None
        self._comp_values = None
        self._comp_len = 0


class TurboQuantCache(Cache):
    """
    HuggingFace Cache with TurboQuant KV compression.

    Uses TurboQuantLayer for compressed layers and DynamicLayer for
    skipped layers (first/last N layers kept at full precision).

    Memory savings are realized between generation steps:
    only compressed data + small buffer persist in GPU memory.
    """

    def __init__(self, head_dim, num_layers, key_bits=4, value_bits=4,
                 buffer_size=256, value_group_size=32,
                 skip_first_layers=0, skip_last_layers=0):
        compress_set = set(range(skip_first_layers, num_layers - skip_last_layers))

        layers = []
        for i in range(num_layers):
            if i in compress_set:
                layers.append(TurboQuantLayer(
                    head_dim=head_dim,
                    key_bits=key_bits,
                    value_bits=value_bits,
                    buffer_size=buffer_size,
                    value_group_size=value_group_size,
                    seed=42 + i * 7,
                ))
            else:
                layers.append(DynamicLayer())

        super().__init__(layers=layers)

        self.head_dim = head_dim
        self.num_layers = num_layers
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.buffer_size = buffer_size
        self.value_group_size = value_group_size
        self.skip_first = skip_first_layers
        self.skip_last = skip_last_layers

    def memory_report(self):
        """Report memory usage: compressed vs full precision equivalent."""
        compressed_bytes = 0
        buffer_bytes = 0
        seq_len = self.get_seq_length(0)
        num_kv_heads = 0
        compressed_layers = 0
        full_layers = 0

        for li, layer in enumerate(self.layers):
            if isinstance(layer, TurboQuantLayer):
                if layer.is_initialized and layer.keys.numel() > 0:
                    if num_kv_heads == 0:
                        num_kv_heads = layer.keys.shape[1]
                    buffer_bytes += layer.keys.nelement() * 2    # bf16
                    buffer_bytes += layer.values.nelement() * 2

                if layer._comp_keys is not None:
                    compressed_layers += 1
                    kq = layer._comp_keys
                    compressed_bytes += kq.mse_indices.nelement()
                    compressed_bytes += kq.qjl_signs.nelement()
                    compressed_bytes += kq.residual_norms.nelement() * 2
                    compressed_bytes += kq.norms.nelement() * 2
                    vq = layer._comp_values
                    compressed_bytes += vq.data.nelement()
                    compressed_bytes += vq.scales.nelement() * 2
                    compressed_bytes += vq.zeros.nelement() * 2
                else:
                    full_layers += 1
            elif isinstance(layer, DynamicLayer):
                full_layers += 1
                if layer.is_initialized and layer.keys.numel() > 0:
                    if num_kv_heads == 0:
                        num_kv_heads = layer.keys.shape[1]
                    buffer_bytes += layer.keys.nelement() * 2
                    buffer_bytes += layer.values.nelement() * 2

        comp_tokens = max(
            (l._comp_len for l in self.layers if isinstance(l, TurboQuantLayer)),
            default=0,
        )

        actual = compressed_bytes + buffer_bytes
        full = self.num_layers * 2 * num_kv_heads * seq_len * self.head_dim * 2

        return {
            "seq_len": seq_len,
            "compressed_tokens": comp_tokens,
            "buffer_tokens": seq_len - comp_tokens,
            "compressed_layers": compressed_layers,
            "full_precision_layers": full_layers,
            "actual_mb": actual / (1024 ** 2),
            "full_precision_mb": full / (1024 ** 2),
            "ratio": full / actual if actual > 0 else 0,
            "savings_mb": (full - actual) / (1024 ** 2) if full > actual else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Model Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_model():
    """Load Qwen3-BEST model and tokenizer in bfloat16."""
    print("[Model] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"[Model] Loaded on {model.device}, dtype={model.dtype}")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Streaming Inference with TurboQuant Cache
# ═══════════════════════════════════════════════════════════════════════════════

def llm_stream(model, tokenizer, conversation, use_tq=True):
    """
    Run streaming inference with optional TurboQuant KV cache compression.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        conversation: list of message dicts
        use_tq: if True, use TurboQuantCache; otherwise use default cache

    Returns:
        dict with reasoning, content, usage stats, timing, and memory report
    """
    start_time = time.time()

    text = tokenizer.apply_chat_template(
        conversation, tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Create cache
    if use_tq:
        cache = TurboQuantCache(
            head_dim=model.config.head_dim,
            num_layers=model.config.num_hidden_layers,
            key_bits=TQ_KEY_BITS,
            value_bits=TQ_VALUE_BITS,
            buffer_size=TQ_BUFFER_SIZE,
            value_group_size=TQ_VALUE_GROUP_SIZE,
            skip_first_layers=TQ_SKIP_FIRST_LAYERS,
            skip_last_layers=TQ_SKIP_LAST_LAYERS,
        )
        n_compressed = model.config.num_hidden_layers - TQ_SKIP_FIRST_LAYERS - TQ_SKIP_LAST_LAYERS
        print(f"[TQ] Cache: {TQ_KEY_BITS}-bit keys, {TQ_VALUE_BITS}-bit values, "
              f"buffer={TQ_BUFFER_SIZE}, compressing {n_compressed}/{model.config.num_hidden_layers} layers")
    else:
        cache = None

    # Streaming setup
    buffer = StringIO()

    class CapturingStreamer(TextStreamer):
        def __init__(self, tokenizer, buf):
            super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
            self.buf = buf
            self.token_count = 0

        def on_finalized_text(self, text, stream_end=False):
            self.buf.write(text)
            self.token_count += 1
            print(text, end="", flush=True)

    streamer = CapturingStreamer(tokenizer, buffer)

    # Generate
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        do_sample=DO_SAMPLE,
        top_p=TOP_P,
        top_k=TOP_K,
        repetition_penalty=REPETITION_PENALTY,
        streamer=streamer,
    )
    if cache is not None:
        gen_kwargs["past_key_values"] = cache

    with torch.inference_mode():
        output_ids = model.generate(**gen_kwargs)

    print()  # newline after stream

    # Count actual generated tokens
    total_tokens = output_ids.shape[-1] - input_len

    response = buffer.getvalue()
    time_taken = time.time() - start_time

    # Parse thinking/content — strip <think> and </think> tags
    if "</think>" in response:
        parts = response.rsplit("</think>", 1)
        reasoning = parts[0].strip()
        # Strip leading <think> tag from reasoning
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):].strip()
        content = parts[1].strip()
    else:
        reasoning = ""
        content = response.strip()
        # Strip <think> if present but no </think> (incomplete thinking)
        if content.startswith("<think>"):
            content = content[len("<think>"):].strip()

    # Token counts from actual output
    reasoning_tokens = 0
    content_tokens = total_tokens
    if reasoning:
        # Estimate split between reasoning and content tokens
        reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))
        content_tokens = total_tokens - reasoning_tokens

    # Memory report
    mem_report = cache.memory_report() if use_tq and cache is not None else {}

    return {
        "reasoning": reasoning,
        "content": content,
        "usage": {
            "reasoning_tokens": reasoning_tokens,
            "content_tokens": content_tokens,
            "total_tokens": total_tokens,
        },
        "time_taken": time_taken,
        "tq_memory": mem_report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Quantization Self-Test
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Verify TurboQuant quantization math on synthetic data."""
    print("=" * 60)
    print("TurboQuant Self-Test")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = 128  # matches Qwen3-BEST head_dim

    # Test TurboQuantProd (Algorithm 2) at configured bit-width
    tq = TurboQuantProd(dim=dim, bits=TQ_KEY_BITS, device=device, seed=42)

    # Random key vectors — use larger sample for stable statistics
    x = torch.randn(1, 8, 256, dim, device=device, dtype=torch.bfloat16)

    # Quantize → dequantize
    q = tq.quantize(x)
    x_hat = tq.dequantize(q).to(x.dtype)

    # Cosine similarity
    cos_sim = F.cosine_similarity(x.reshape(-1, dim).float(), x_hat.reshape(-1, dim).float(), dim=-1)
    avg_cos = cos_sim.mean().item()

    # MSE
    mse = ((x.float() - x_hat.float()) ** 2).mean().item()

    print(f"  Key quantization ({TQ_KEY_BITS}-bit TurboQuantProd):")
    print(f"    Avg cosine similarity: {avg_cos:.6f}")
    print(f"    MSE: {mse:.6e}")
    print(f"    Packed MSE indices shape: {q.mse_indices.shape}")
    print(f"    Packed QJL signs shape:   {q.qjl_signs.shape}")

    # Test value quantization
    v = torch.randn(1, 8, 256, dim, device=device, dtype=torch.bfloat16)
    vq = quantize_values(v, bits=4, group_size=32)
    v_hat = dequantize_values(vq, group_size=32).to(v.dtype)

    cos_sim_v = F.cosine_similarity(v.reshape(-1, dim).float(), v_hat.reshape(-1, dim).float(), dim=-1)
    avg_cos_v = cos_sim_v.mean().item()
    mse_v = ((v.float() - v_hat.float()) ** 2).mean().item()

    print(f"  Value quantization (4-bit group):")
    print(f"    Avg cosine similarity: {avg_cos_v:.6f}")
    print(f"    MSE: {mse_v:.6e}")
    print(f"    Packed data shape: {vq.data.shape}")

    # Memory compression ratio
    original_bytes = x.nelement() * 2  # bf16
    compressed_bytes = (
        q.mse_indices.nelement()
        + q.qjl_signs.nelement()
        + q.residual_norms.nelement() * 2
        + q.norms.nelement() * 2
    )
    print(f"  Key compression ratio: {original_bytes / compressed_bytes:.2f}x")

    original_v = v.nelement() * 2
    compressed_v = vq.data.nelement() + vq.scales.nelement() * 2 + vq.zeros.nelement() * 2
    print(f"  Value compression ratio: {original_v / compressed_v:.2f}x")

    # Inner product accuracy test (the core TurboQuant guarantee)
    query = torch.randn(1, 8, 4, dim, device=device, dtype=torch.bfloat16)
    true_dots = torch.matmul(query.float(), x.float().transpose(-2, -1))
    est_dots = torch.matmul(query.float(), x_hat.float().transpose(-2, -1))
    dot_corr = torch.corrcoef(torch.stack([true_dots.flatten(), est_dots.flatten()]))[0, 1].item()
    relative_bias = ((est_dots - true_dots).mean() / true_dots.abs().mean()).item()
    print(f"  Inner product accuracy (TQ guarantee):")
    print(f"    Correlation:    {dot_corr:.6f}")
    print(f"    Relative bias:  {relative_bias:.6f}")

    # 3-bit/128d: cos~0.92 for keys is expected; inner product correlation is the key metric
    passed = avg_cos > 0.85 and avg_cos_v > 0.99 and dot_corr > 0.90
    print(f"\n  Result: {'PASS ✓' if passed else 'FAIL ✗'}")
    print("=" * 60)
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    os.makedirs("./OUT", exist_ok=True)

    # Step 1: Self-test quantization
    if not self_test():
        print("[ERROR] Self-test failed. Aborting.")
        exit(1)

    # Step 2: Load model
    model, tokenizer = load_model()

    # Step 3: Check for interactive mode vs single-shot
    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    if interactive:
        # Multi-turn interactive chat
        print("\n" + "=" * 60)
        print("TurboQuant Interactive Chat (type 'quit' or 'exit' to stop)")
        print("=" * 60 + "\n")

        messages = []
        turn = 0
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Exiting]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("[Exiting]")
                break

            messages.append({"role": "user", "content": user_input})
            turn += 1

            print(f"\n[Turn {turn}]")
            print("Assistant: ", end="", flush=True)

            ret = llm_stream(model, tokenizer, messages, use_tq=True)

            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": ret["content"]})

            # Print stats
            print(f"\n  [{ret['usage']['total_tokens']} tokens, "
                  f"{ret['time_taken']:.1f}s, "
                  f"{ret['usage']['total_tokens'] / ret['time_taken']:.1f} tok/s]")

            if ret["tq_memory"]:
                m = ret["tq_memory"]
                print(f"  [KV: {m['actual_mb']:.1f}MB / {m['full_precision_mb']:.1f}MB, "
                      f"{m['ratio']:.2f}x compression]")
            print()

    else:
        # Single-shot inference
        messages = [{"role": "user", "content": INPUT_MESSAGE}]

        print("\n" + "=" * 60)
        print("Inference with TurboQuant KV Cache")
        print("=" * 60 + "\n")

        ret = llm_stream(model, tokenizer, messages, use_tq=True)

        print("\n" + "-" * 60)
        print("Results:")
        print(f"  Total tokens: {ret['usage']['total_tokens']}")
        print(f"  Time: {ret['time_taken']:.2f}s")
        if ret["usage"]["total_tokens"] > 0 and ret["time_taken"] > 0:
            tps = ret["usage"]["total_tokens"] / ret["time_taken"]
            print(f"  Tokens/sec: {tps:.2f}")

        if ret["tq_memory"]:
            m = ret["tq_memory"]
            print(f"\n  TurboQuant Memory Report:")
            print(f"    Sequence length:    {m['seq_len']}")
            print(f"    Compressed tokens:  {m['compressed_tokens']}")
            print(f"    Buffer tokens:      {m['buffer_tokens']}")
            print(f"    Compressed layers:  {m['compressed_layers']}")
            print(f"    Full prec. layers:  {m['full_precision_layers']}")
            print(f"    Actual KV cache:    {m['actual_mb']:.2f} MB")
            print(f"    Full precision:     {m['full_precision_mb']:.2f} MB")
            print(f"    Compression ratio:  {m['ratio']:.2f}x")
            print(f"    Savings:            {m['savings_mb']:.2f} MB")
        print("-" * 60)
