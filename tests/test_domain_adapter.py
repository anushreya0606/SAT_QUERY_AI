"""
Unit tests for the Domain Gap Adaptation module.
"""

import numpy as np
import torch
import pytest

from models.domain_adapter import (
    SensorMetadata,
    SensorPromptConditioner,
    MultiScaleGSDAdapter,
    RadiometricDomainAligner,
)


def test_sensor_metadata():
    carto = SensorMetadata.cartosat_2s_pan()
    assert carto.name == "Cartosat-2S"
    assert carto.gsd == 0.65
    assert carto.modality == "optical"
    token = carto.to_prefix_token()
    assert "[SENSOR: Cartosat-2S" in token
    assert "GSD: 0.65m" in token

    risat = SensorMetadata.risat_1_frs("dual")
    assert risat.modality == "sar"
    assert "POL: DUAL" in risat.to_prefix_token()


def test_sensor_prompt_conditioner():
    cond = SensorPromptConditioner()
    raw_prompt = "Identify commercial buildings."
    conditioned = cond.condition(raw_prompt, sensor="cartosat")
    assert "[SENSOR: Cartosat-2S" in conditioned
    assert raw_prompt in conditioned
    assert "Very-high-resolution" in conditioned

    # SAR prompt
    sar_prompt = cond.condition("Detect flooded areas.", sensor="risat")
    assert "SAR imagery features radar backscatter" in sar_prompt


def test_multiscale_gsd_adapter():
    adapter = MultiScaleGSDAdapter(in_channels=3, out_channels=3, reference_gsd=10.0)
    assert adapter.calculate_scale_factor(0.65) > 10.0
    assert adapter.calculate_scale_factor(10.0) == 1.0

    dummy_input = torch.randn(1, 3, 336, 336)
    output = adapter(dummy_input, current_gsd=0.65)
    assert output.shape == dummy_input.shape


def test_radiometric_domain_aligner():
    # Test optical alignment
    raw_img = np.random.uniform(100, 4000, (256, 256)).astype(np.float32)
    aligned = RadiometricDomainAligner.align_optical_to_reference(raw_img)
    assert aligned.min() >= 0.0
    assert aligned.max() <= 1.0

    # Test SAR backscatter alignment
    sar_db = np.array([-35.0, -25.0, -10.0, 0.0, 5.0])
    aligned_sar = RadiometricDomainAligner.align_sar_backscatter(sar_db, target_db_range=(-25.0, 0.0))
    assert aligned_sar[0] == 0.0  # clipped at -25dB
    assert aligned_sar[-1] == 1.0 # clipped at 0dB
    assert 0.0 <= aligned_sar[2] <= 1.0
