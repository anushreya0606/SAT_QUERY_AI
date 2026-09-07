"""
Preprocessing pipelines for remote sensing imagery.

Provides three main functions:
  - preprocess_optical():        Resize + normalise optical imagery (Cartosat, natural colour)
  - preprocess_sar():            dB conversion, speckle reduction, normalise SAR imagery (RISAT)
  - preprocess_multispectral():  Band selection + normalise Sentinel-2 / Cartosat MS imagery
  - batch_preprocess():          Process an entire directory tree in one shot

All functions write to an output directory and return the output path.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Rasterio helper (graceful fallback to Pillow for non-geospatial files)      #
# --------------------------------------------------------------------------- #

def _read_raster(
    path: str,
    bands: Optional[List[int]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Read a raster file and return (array, metadata).

    Tries rasterio first (for GeoTIFF / multi-band), falls back to Pillow.

    Args:
        path:  Path to the image file.
        bands: 1-indexed list of band indices to read. None = all bands.

    Returns:
        array: (C, H, W) float32 ndarray.
        meta:  dict with 'crs', 'transform', 'driver' (empty if PIL fallback).
    """
    try:
        import rasterio
        with rasterio.open(path) as src:
            if bands:
                arr = src.read(bands)  # (len(bands), H, W)
            else:
                arr = src.read()       # (C, H, W)
            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "driver": src.driver,
                "shape": src.shape,
            }
        return arr.astype(np.float32), meta
    except ImportError:
        logger.debug("rasterio not available; using Pillow fallback.")
    except Exception as e:
        logger.warning(f"rasterio failed to read '{path}': {e}. Falling back to Pillow.")

    # Pillow fallback — only reads supported formats (JPEG, PNG, etc.)
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # HWC → CHW
    return arr, {}


def _save_as_png(arr: np.ndarray, output_path: str):
    """Save a (3, H, W) float [0,1] array as a PNG file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    arr_uint8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr_uint8.ndim == 3:
        # CHW → HWC
        img = Image.fromarray(arr_uint8.transpose(1, 2, 0))
    else:
        img = Image.fromarray(arr_uint8)
    img.save(output_path)


# --------------------------------------------------------------------------- #
#  Optical Preprocessing                                                       #
# --------------------------------------------------------------------------- #

def preprocess_optical(
    image_path: str,
    output_path: str,
    size: int = 336,
    percentile_clip: Tuple[float, float] = (2.0, 98.0),
) -> str:
    """
    Preprocess an optical image (Cartosat-2S PAN or natural colour).

    Steps:
      1. Read image (rasterio for GeoTIFF, Pillow for JPEG/PNG).
      2. Select first 3 bands (or replicate single band to RGB).
      3. Percentile-clip to improve contrast.
      4. Normalise to [0, 1].
      5. Resize to (size × size).
      6. Save as PNG.

    Args:
        image_path:      Input file path.
        output_path:     Output PNG path.
        size:            Target spatial resolution in pixels.
        percentile_clip: (lo_pct, hi_pct) for contrast stretching.

    Returns:
        output_path (for chaining / logging).
    """
    arr, _ = _read_raster(image_path)

    # Ensure 3-channel
    if arr.shape[0] == 1:
        arr = np.concatenate([arr, arr, arr], axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3]

    # Percentile stretch per channel
    lo_pct, hi_pct = percentile_clip
    result = np.zeros_like(arr)
    for i in range(3):
        lo = np.percentile(arr[i], lo_pct)
        hi = np.percentile(arr[i], hi_pct)
        if hi > lo:
            result[i] = np.clip((arr[i] - lo) / (hi - lo), 0.0, 1.0)
        else:
            result[i] = np.clip(arr[i] / (arr[i].max() + 1e-8), 0.0, 1.0)

    # Resize via PIL
    pil = Image.fromarray((result.transpose(1, 2, 0) * 255).astype(np.uint8)).convert("RGB")
    pil = pil.resize((size, size), Image.BILINEAR)
    result_resized = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

    _save_as_png(result_resized, output_path)
    logger.info(f"Optical → {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
#  SAR Preprocessing                                                           #
# --------------------------------------------------------------------------- #

def preprocess_sar(
    image_path: str,
    output_path: str,
    size: int = 336,
    db_range: Tuple[float, float] = (-30.0, 0.0),
    apply_speckle_filter: bool = True,
) -> str:
    """
    Preprocess a SAR image (RISAT-1 / RISAT-2 HH or HV polarisation).

    Steps:
      1. Read image via rasterio.
      2. Convert linear amplitude/power → log (dB) scale.
      3. Optional: apply 3×3 box-car speckle filter.
      4. Clip to db_range and normalise to [0, 1].
      5. Replicate to 3 channels (or use HH, HV, HH-HV if 2-band).
      6. Resize and save as PNG.

    Args:
        image_path:           Input file path (GeoTIFF).
        output_path:          Output PNG path.
        size:                 Target spatial resolution.
        db_range:             (min_dB, max_dB) clipping range.
        apply_speckle_filter: Whether to apply box-car averaging.

    Returns:
        output_path
    """
    arr, _ = _read_raster(image_path)  # (C, H, W)

    def to_db(band: np.ndarray) -> np.ndarray:
        return 10.0 * np.log10(np.clip(band, 1e-10, None))

    def speckle_filter(band: np.ndarray) -> np.ndarray:
        try:
            from scipy.ndimage import uniform_filter
            return uniform_filter(band, size=3)
        except ImportError:
            return band

    def normalise(band: np.ndarray) -> np.ndarray:
        db_min, db_max = db_range
        band = np.clip(band, db_min, db_max)
        return (band - db_min) / (db_max - db_min)

    processed = []
    for i in range(min(arr.shape[0], 2)):
        band = arr[i]
        if band.max() > 1.0:   # assume linear amplitude/power
            band = to_db(band)
        if apply_speckle_filter:
            band = speckle_filter(band)
        band = normalise(band).astype(np.float32)
        processed.append(band)

    if len(processed) == 1:
        result = np.stack([processed[0]] * 3, axis=0)
    elif len(processed) == 2:
        hh, hv = processed[0], processed[1]
        ratio = np.clip(hh - hv, 0.0, 1.0)
        result = np.stack([hh, hv, ratio], axis=0)
    else:
        result = np.stack(processed[:3], axis=0)

    # Resize
    pil = Image.fromarray((result.transpose(1, 2, 0) * 255).astype(np.uint8)).convert("RGB")
    pil = pil.resize((size, size), Image.BILINEAR)
    result_resized = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

    _save_as_png(result_resized, output_path)
    logger.info(f"SAR → {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
#  Multispectral Preprocessing                                                 #
# --------------------------------------------------------------------------- #

def preprocess_multispectral(
    image_path: str,
    output_path: str,
    size: int = 336,
    band_indices: Optional[List[int]] = None,
    reflectance_scale: float = 10000.0,
) -> str:
    """
    Preprocess a multispectral image (Sentinel-2 13-band or Cartosat-2S MS).

    Steps:
      1. Read all bands.
      2. Select the specified bands (default: Sentinel-2 Red-Green-Blue → bands 4,3,2).
      3. Scale reflectance values to [0, 1] (divide by reflectance_scale).
      4. Percentile-stretch for visualisation.
      5. Resize and save as PNG.

    Args:
        image_path:        Input file path.
        output_path:       Output PNG path.
        size:              Target spatial resolution.
        band_indices:      1-indexed band numbers to select (e.g. [4, 3, 2] for S2 RGB).
                           Defaults to [4, 3, 2].
        reflectance_scale: Divisor to convert DN → reflectance (Sentinel-2 default: 10000).

    Returns:
        output_path
    """
    band_indices = band_indices or [4, 3, 2]

    arr, _ = _read_raster(image_path, bands=band_indices)  # (3, H, W)

    # Scale to reflectance
    arr = arr / reflectance_scale

    # Percentile stretch
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        lo = np.percentile(arr[i], 2)
        hi = np.percentile(arr[i], 98)
        if hi > lo:
            result[i] = np.clip((arr[i] - lo) / (hi - lo), 0.0, 1.0)
        else:
            result[i] = np.clip(arr[i], 0.0, 1.0)

    # Resize
    pil = Image.fromarray((result.transpose(1, 2, 0) * 255).astype(np.uint8)).convert("RGB")
    pil = pil.resize((size, size), Image.BILINEAR)
    result_resized = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0

    _save_as_png(result_resized, output_path)
    logger.info(f"Multispectral → {output_path}")
    return output_path


# --------------------------------------------------------------------------- #
#  Batch Processing                                                            #
# --------------------------------------------------------------------------- #

def batch_preprocess(
    input_dir: str,
    output_dir: str,
    sensor_type: str = "optical",
    size: int = 336,
    extensions: Optional[List[str]] = None,
    **kwargs,
) -> List[str]:
    """
    Batch-process all image files in a directory tree.

    Args:
        input_dir:   Root directory to search for images.
        output_dir:  Root output directory (mirrored structure).
        sensor_type: One of 'optical', 'sar', 'multispectral'.
        size:        Target spatial resolution.
        extensions:  File extensions to include. Defaults to common raster formats.
        **kwargs:    Additional kwargs forwarded to the preprocessing function.

    Returns:
        List of output paths.
    """
    extensions = extensions or [".tif", ".tiff", ".jpg", ".jpeg", ".png"]
    fn_map = {
        "optical": preprocess_optical,
        "sar": preprocess_sar,
        "multispectral": preprocess_multispectral,
    }
    preprocess_fn = fn_map.get(sensor_type.lower())
    if preprocess_fn is None:
        raise ValueError(f"Unknown sensor_type '{sensor_type}'. Choose from: {list(fn_map)}")

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_paths = []

    files = [
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    logger.info(f"Found {len(files)} files in '{input_dir}' for '{sensor_type}' preprocessing.")

    for file_path in files:
        rel = file_path.relative_to(input_root)
        out_path = output_root / rel.with_suffix(".png")
        try:
            preprocess_fn(str(file_path), str(out_path), size=size, **kwargs)
            output_paths.append(str(out_path))
        except Exception as e:
            logger.error(f"Failed to preprocess '{file_path}': {e}")

    logger.info(f"Batch preprocessing complete: {len(output_paths)}/{len(files)} files processed.")
    return output_paths
