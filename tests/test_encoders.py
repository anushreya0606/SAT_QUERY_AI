"""
Unit tests for domain-aware vision encoders.
"""

import numpy as np
import torch
import pytest
from PIL import Image

from models.encoders import (
    OpticalEncoder,
    SAREncoder,
    MultispectralEncoder,
    build_encoder,
)


def test_optical_encoder():
    enc = OpticalEncoder(image_size=336)
    dummy_img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    tensor = enc.encode(dummy_img)
    assert tensor.shape == (3, 336, 336)
    assert isinstance(tensor, torch.Tensor)


def test_sar_encoder_single_pol():
    enc = SAREncoder(image_size=336, dual_pol=False)
    dummy_arr = np.random.uniform(0.01, 10.0, (256, 256)).astype(np.float32)
    tensor = enc.encode(dummy_arr)
    assert tensor.shape == (3, 336, 336)


def test_sar_encoder_dual_pol():
    enc = SAREncoder(image_size=336, dual_pol=True)
    hh = np.random.uniform(0.01, 10.0, (256, 256)).astype(np.float32)
    hv = np.random.uniform(0.005, 5.0, (256, 256)).astype(np.float32)
    tensor = enc.encode((hh, hv))
    assert tensor.shape == (3, 336, 336)


def test_multispectral_encoder():
    enc = MultispectralEncoder(image_size=336, band_indices=[3, 2, 1])
    # 4-band image (B, G, R, NIR)
    dummy_ms = np.random.uniform(0, 3000, (4, 256, 256)).astype(np.float32)
    tensor = enc.encode(dummy_ms)
    assert tensor.shape == (3, 336, 336)


def test_build_encoder_factory():
    opt = build_encoder("optical", image_size=256)
    assert isinstance(opt, OpticalEncoder)
    assert opt.image_size == 256

    sar = build_encoder("sar", image_size=256)
    assert isinstance(sar, SAREncoder)

    ms = build_encoder("multispectral", image_size=256)
    assert isinstance(ms, MultispectralEncoder)

    with pytest.raises(ValueError):
        build_encoder("invalid_sensor")
