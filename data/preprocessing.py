"""
Preprocessing utilities for remote sensing imagery.
Handles optical (Cartosat-2S), SAR (RISAT), and multi-spectral (Sentinel-1/2) data.
"""

import os
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image

try:
    import rasterio
    from rasterio.enums import Resampling

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. GeoTIFF loading will be limited.")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to 0-255 uint8 range using percentile clipping."""
    p2, p98 = np.percentile(arr, (2, 98))
    if p98 - p2 == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    clipped = np.clip(arr, p2, p98)
    normalized = ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    return normalized


def load_geotiff(filepath: str, bands: Optional[List[int]] = None) -> np.ndarray:
    """
    Load a GeoTIFF file as a numpy array.

    Args:
        filepath: Path to the .tif file.
        bands: List of 1-indexed band numbers to load. None loads all bands.

    Returns:
        numpy array of shape (H, W, C) or (H, W) for single-band.
    """
    if HAS_RASTERIO:
        with rasterio.open(filepath) as src:
            if bands:
                data = src.read(bands)
            else:
                data = src.read()
            # rasterio returns (C, H, W) — transpose to (H, W, C)
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0))
            return data
    else:
        # Fallback: try PIL (only works for simple TIFFs)
        img = Image.open(filepath)
        return np.array(img)


# ------------------------------------------------------------------ #
#  Sentinel Preprocessing
# ------------------------------------------------------------------ #


def preprocess_sentinel2(
    filepath: str,
    target_size: Tuple[int, int] = (336, 336),
    rgb_bands: Tuple[int, int, int] = (4, 3, 2),
) -> Image.Image:
    """
    Preprocess Sentinel-2 multi-spectral imagery.

    Sentinel-2 has 13 spectral bands. For VLM input, we typically use
    a natural color composite (B4-Red, B3-Green, B2-Blue).

    Args:
        filepath: Path to Sentinel-2 GeoTIFF.
        target_size: Output image size (H, W).
        rgb_bands: Band indices for RGB composite (1-indexed).

    Returns:
        PIL Image in RGB.
    """
    data = load_geotiff(filepath, bands=list(rgb_bands))

    # Handle each band separately for better normalization
    channels = []
    for i in range(data.shape[2] if data.ndim == 3 else 1):
        band = data[:, :, i] if data.ndim == 3 else data
        channels.append(normalize_to_uint8(band))

    rgb = np.stack(channels, axis=2)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


def preprocess_sentinel1(
    filepath: str,
    target_size: Tuple[int, int] = (336, 336),
    polarization: str = "VV",
) -> Image.Image:
    """
    Preprocess Sentinel-1 SAR (Synthetic Aperture Radar) imagery.

    SAR data is typically in decibel scale (dB) and single/dual polarization.
    We apply log-scaling and speckle filtering to produce a viewable image.

    Args:
        filepath: Path to Sentinel-1 GeoTIFF.
        target_size: Output image size (H, W).
        polarization: Polarization channel ('VV', 'VH', or 'dual').

    Returns:
        PIL Image (grayscale converted to RGB for VLM compatibility).
    """
    data = load_geotiff(filepath)
    if data.ndim == 3:
        # Use first band (VV) by default
        data = data[:, :, 0]

    # Apply log transform for SAR backscatter (if not already in dB)
    data = data.astype(np.float32)
    if data.min() >= 0:
        data = 10 * np.log10(data + 1e-10)

    # Simple speckle reduction via median filter
    from scipy.ndimage import median_filter
    data = median_filter(data, size=3)

    # Normalize to uint8
    normalized = normalize_to_uint8(data)

    # Convert grayscale to RGB (VLMs expect 3-channel input)
    rgb = np.stack([normalized] * 3, axis=2)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


# ------------------------------------------------------------------ #
#  Cartosat-2S Preprocessing
# ------------------------------------------------------------------ #


def preprocess_cartosat(
    filepath: str,
    target_size: Tuple[int, int] = (336, 336),
) -> Image.Image:
    """
    Preprocess Cartosat-2S high-resolution optical imagery.

    Cartosat-2S provides panchromatic (~0.6m) and multi-spectral (~2m)
    data. The panchromatic band is single-channel grayscale.

    Args:
        filepath: Path to Cartosat-2S GeoTIFF.
        target_size: Output image size (H, W).

    Returns:
        PIL Image in RGB.
    """
    data = load_geotiff(filepath)

    if data.ndim == 2:
        # Panchromatic (single band) — convert to RGB
        normalized = normalize_to_uint8(data)
        rgb = np.stack([normalized] * 3, axis=2)
    elif data.shape[2] >= 3:
        # Multi-spectral — use first 3 bands as RGB
        channels = [normalize_to_uint8(data[:, :, i]) for i in range(3)]
        rgb = np.stack(channels, axis=2)
    else:
        normalized = normalize_to_uint8(data[:, :, 0])
        rgb = np.stack([normalized] * 3, axis=2)

    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


# ------------------------------------------------------------------ #
#  RISAT Preprocessing
# ------------------------------------------------------------------ #


def preprocess_risat(
    filepath: str,
    target_size: Tuple[int, int] = (336, 336),
    apply_speckle_filter: bool = True,
) -> Image.Image:
    """
    Preprocess RISAT SAR imagery.

    RISAT provides C-band SAR data. Processing is similar to Sentinel-1
    but at different resolution and radar characteristics.

    Args:
        filepath: Path to RISAT GeoTIFF.
        target_size: Output image size (H, W).
        apply_speckle_filter: Whether to apply Lee speckle filter.

    Returns:
        PIL Image (grayscale to RGB for VLM compatibility).
    """
    data = load_geotiff(filepath)
    if data.ndim == 3:
        data = data[:, :, 0]

    data = data.astype(np.float32)

    # Log transform for SAR amplitude data
    if data.min() >= 0:
        data = 10 * np.log10(data + 1e-10)

    # Lee speckle filter (simplified)
    if apply_speckle_filter:
        data = _lee_filter(data, window_size=5)

    normalized = normalize_to_uint8(data)
    rgb = np.stack([normalized] * 3, axis=2)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize(target_size, Image.LANCZOS)
    return img


def _lee_filter(img: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Simplified Lee speckle filter for SAR imagery.
    Reduces speckle noise while preserving edges.

    Args:
        img: 2D numpy array of SAR backscatter values.
        window_size: Size of the filter window (odd number).

    Returns:
        Filtered 2D numpy array.
    """
    from scipy.ndimage import uniform_filter

    img_mean = uniform_filter(img, size=window_size)
    img_sqr_mean = uniform_filter(img ** 2, size=window_size)
    img_var = img_sqr_mean - img_mean ** 2

    overall_var = np.var(img)

    # Weighting factor
    weight = img_var / (img_var + overall_var + 1e-10)

    filtered = img_mean + weight * (img - img_mean)
    return filtered


# ------------------------------------------------------------------ #
#  Unified Preprocessing Interface
# ------------------------------------------------------------------ #


def preprocess_image(
    filepath: str,
    sensor_type: str = "auto",
    target_size: Tuple[int, int] = (336, 336),
) -> Image.Image:
    """
    Unified preprocessing entry point. Auto-detects sensor type if possible.

    Args:
        filepath: Path to the image file.
        sensor_type: One of 'sentinel1', 'sentinel2', 'cartosat', 'risat', 'auto', 'rgb'.
        target_size: Output image size (H, W).

    Returns:
        Preprocessed PIL Image in RGB.
    """
    sensor_type = sensor_type.lower()

    if sensor_type == "auto":
        sensor_type = _detect_sensor_type(filepath)

    if sensor_type == "sentinel2":
        return preprocess_sentinel2(filepath, target_size)
    elif sensor_type == "sentinel1":
        return preprocess_sentinel1(filepath, target_size)
    elif sensor_type == "cartosat":
        return preprocess_cartosat(filepath, target_size)
    elif sensor_type == "risat":
        return preprocess_risat(filepath, target_size)
    else:
        # Standard RGB image (PNG, JPEG, etc.)
        img = Image.open(filepath).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        return img


def _detect_sensor_type(filepath: str) -> str:
    """Heuristic sensor type detection based on filename and file properties."""
    basename = os.path.basename(filepath).lower()

    if "s1" in basename or "sentinel1" in basename or "sar" in basename:
        return "sentinel1"
    elif "s2" in basename or "sentinel2" in basename:
        return "sentinel2"
    elif "cartosat" in basename or "c2s" in basename:
        return "cartosat"
    elif "risat" in basename:
        return "risat"

    # Check file extension and number of bands
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".tif", ".tiff") and HAS_RASTERIO:
        try:
            with rasterio.open(filepath) as src:
                if src.count == 1:
                    return "sentinel1"  # Likely SAR
                elif src.count >= 4:
                    return "sentinel2"  # Likely multi-spectral
                else:
                    return "rgb"
        except Exception:
            pass

    return "rgb"
