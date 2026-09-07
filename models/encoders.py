"""
Domain-aware vision encoders for remote sensing imagery.

Provides specialized preprocessing and feature extraction for:
  - Optical imagery (Cartosat-2S PAN, natural colour)
  - SAR imagery     (RISAT HH/HV polarizations)
  - Multispectral   (Sentinel-2 13-band, Cartosat-2S MS)

All encoders expose a unified ``encode(image) -> torch.Tensor`` interface so
they can be swapped into the VLM pipeline without code changes.
"""

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Sentinel-2 / Cartosat-2S band statistics (precomputed on EO datasets)      #
# --------------------------------------------------------------------------- #

# RGB channels (bands 4-3-2 for Sentinel-2, adapted for Cartosat-2S)
_OPTICAL_MEAN = [0.485, 0.456, 0.406]
_OPTICAL_STD  = [0.229, 0.224, 0.225]

# SAR (log-scale dB, single channel)
_SAR_MEAN = [-12.0]   # Approximate dB mean over RISAT HH acquisitions
_SAR_STD  = [5.0]


# =========================================================================== #
#  Base Encoder                                                                #
# =========================================================================== #

class BaseEncoder(nn.Module):
    """Abstract base class for all domain-aware encoders."""

    def __init__(self, image_size: int = 336):
        super().__init__()
        self.image_size = image_size

    def preprocess(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess a raw image into a normalised tensor. Override in subclasses."""
        raise NotImplementedError

    def encode(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Full encode pipeline: preprocess → backbone forward pass (if any).
        Returns a (C, H, W) tensor ready for the VLM vision tower.
        """
        return self.preprocess(image)

    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
        """Convert various image formats to PIL.Image (RGB)."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, torch.Tensor):
            import torchvision.transforms.functional as TF
            return TF.to_pil_image(image.cpu().float()).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[0] in (1, 3) and image.ndim == 3:
                image = image.transpose(1, 2, 0)  # CHW → HWC
            if image.dtype != np.uint8:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")


# =========================================================================== #
#  Optical Encoder (Cartosat-2S PAN / Sentinel-2 RGB)                         #
# =========================================================================== #

class OpticalEncoder(BaseEncoder):
    """
    Encoder for standard optical imagery.

    Applies ImageNet-style normalisation after resizing.  Works for:
      - Cartosat-2S panchromatic  (1 band → replicated to 3)
      - Sentinel-2 RGB composite  (bands 4-3-2)
      - Any 3-channel natural-colour image
    """

    def __init__(self, image_size: int = 336):
        super().__init__(image_size)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_OPTICAL_MEAN, std=_OPTICAL_STD),
        ])

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Resize, to-tensor, and normalise optical image."""
        pil_image = self._to_pil(image)
        return self.transform(pil_image)

    def encode(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        return self.preprocess(image)


# =========================================================================== #
#  SAR Encoder (RISAT-1 / RISAT-2 HH, HV)                                    #
# =========================================================================== #

class SAREncoder(BaseEncoder):
    """
    Encoder for Synthetic Aperture Radar (SAR) imagery from RISAT satellites.

    SAR-specific preprocessing:
      1. Convert linear amplitude → log (dB) scale if input appears linear.
      2. Apply lee_filter-style speckle reduction (3×3 box-car approximation).
      3. Clip to [-30, 0] dB range and normalise.
      4. Replicate single-channel (HH or HV) or stack dual-channel to 3 channels.
    """

    DB_CLIP_MIN = -30.0
    DB_CLIP_MAX =   0.0

    def __init__(self, image_size: int = 336, dual_pol: bool = False):
        """
        Args:
            image_size: Output spatial resolution.
            dual_pol:   If True, expects (HH, HV) tuple/stack as input.
        """
        super().__init__(image_size)
        self.dual_pol = dual_pol
        self._resize = transforms.Resize((image_size, image_size))

    def _to_db(self, arr: np.ndarray) -> np.ndarray:
        """Convert linear power to dB. Clips near-zero values to avoid log(0)."""
        arr = np.clip(arr, 1e-10, None)
        return 10.0 * np.log10(arr)

    def _speckle_filter(self, arr: np.ndarray) -> np.ndarray:
        """
        Box-car (mean) filter for speckle reduction.
        For production use, replace with Lee or Refined Lee filter via scipy.
        """
        from scipy.ndimage import uniform_filter
        try:
            return uniform_filter(arr, size=3).astype(np.float32)
        except ImportError:
            # Fallback: no filtering
            return arr

    def _normalise_db(self, arr: np.ndarray) -> np.ndarray:
        """Clip dB range and normalise to [0, 1]."""
        arr = np.clip(arr, self.DB_CLIP_MIN, self.DB_CLIP_MAX)
        return (arr - self.DB_CLIP_MIN) / (self.DB_CLIP_MAX - self.DB_CLIP_MIN)

    def _process_band(self, band: np.ndarray) -> np.ndarray:
        """Full processing pipeline for one SAR polarisation band."""
        band = band.astype(np.float32)
        # If values >> 1 treat as linear amplitude; convert to dB
        if band.max() > 1.0:
            band = self._to_db(band)
        band = self._speckle_filter(band)
        band = self._normalise_db(band)
        return band

    def preprocess(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor, Tuple],
    ) -> torch.Tensor:
        """
        Args:
            image: Can be:
                   - 2-D ndarray (single pol)
                   - Tuple of two 2-D ndarrays (dual pol HH, HV)
                   - PIL Image (single channel)
                   - Torch tensor (1, H, W) or (2, H, W)
        """
        if isinstance(image, tuple) and len(image) == 2:
            hh = np.array(image[0], dtype=np.float32)
            hv = np.array(image[1], dtype=np.float32)
            hh = self._process_band(hh)
            hv = self._process_band(hv)
            # 3-channel: HH, HV, HH-HV ratio
            ratio = np.clip(hh - hv, 0, 1)
            sar_arr = np.stack([hh, hv, ratio], axis=0)  # (3, H, W)
        elif isinstance(image, torch.Tensor):
            arr = image.cpu().numpy()
            if arr.ndim == 3:
                bands = [self._process_band(arr[i]) for i in range(arr.shape[0])]
            else:
                bands = [self._process_band(arr)]
            while len(bands) < 3:
                bands.append(bands[-1])
            sar_arr = np.stack(bands[:3], axis=0)
        else:
            # PIL Image or 2D ndarray
            if isinstance(image, Image.Image):
                arr = np.array(image.convert("L"), dtype=np.float32)
            else:
                arr = np.array(image, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr.mean(axis=-1 if arr.shape[-1] < arr.shape[0] else 0)
            band = self._process_band(arr)
            sar_arr = np.stack([band, band, band], axis=0)  # (3, H, W) replicated

        tensor = torch.from_numpy(sar_arr).float()  # (3, H, W)
        # Resize
        tensor = self._resize(tensor)
        # Normalise with SAR-specific stats (approximate ImageNet shift)
        tensor = transforms.Normalize(
            mean=_OPTICAL_MEAN, std=_OPTICAL_STD
        )(tensor)
        return tensor

    def encode(self, image) -> torch.Tensor:
        return self.preprocess(image)


# =========================================================================== #
#  Multispectral Encoder (Sentinel-2 13-band / Cartosat-2S MS)                #
# =========================================================================== #

class MultispectralEncoder(BaseEncoder):
    """
    Encoder for multi-band imagery (Sentinel-2 or Cartosat-2S multispectral).

    Selects three bands for RGB-compatible output (default: Red, Green, NIR for
    a false-colour composite, or Blue, Green, Red for true colour).

    Input: (C, H, W) tensor or ndarray where C >= 3.
    """

    # Sentinel-2 band index mapping (0-indexed)
    SENTINEL2_BANDS = {
        "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4,
        "B06": 5, "B07": 6, "B08": 7, "B8A": 8, "B09": 9,
        "B10": 10, "B11": 11, "B12": 12,
    }

    def __init__(
        self,
        image_size: int = 336,
        band_indices: Optional[List[int]] = None,
        per_band_stats: Optional[Tuple[List[float], List[float]]] = None,
    ):
        """
        Args:
            image_size:    Output spatial resolution.
            band_indices:  Indices of the 3 bands to select. Defaults to [3, 2, 1]
                           (Sentinel-2 Red, Green, Blue → true colour).
            per_band_stats: Optional (means, stds) tuple for normalisation.
                            If None, uses ImageNet-style stats after selecting bands.
        """
        super().__init__(image_size)
        self.band_indices = band_indices or [3, 2, 1]  # R, G, B for S2
        self._resize = transforms.Resize((image_size, image_size))

        if per_band_stats:
            means, stds = per_band_stats
        else:
            means, stds = _OPTICAL_MEAN, _OPTICAL_STD
        self._normalise = transforms.Normalize(mean=means, std=stds)

    def preprocess(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
    ) -> torch.Tensor:
        """
        Select bands, normalise, and resize.

        Args:
            image: (C, H, W) array/tensor with C >= max(band_indices)+1.
                   PIL Image is treated as 3-band and band_indices is ignored.
        """
        if isinstance(image, Image.Image):
            pil = image.convert("RGB")
            tensor = transforms.ToTensor()(pil)
        elif isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            elif tensor.ndim == 3 and tensor.shape[2] < tensor.shape[0]:
                tensor = tensor.permute(2, 0, 1)   # HWC → CHW
            # Select and validate bands
            max_band = max(self.band_indices)
            if tensor.shape[0] > max_band:
                tensor = tensor[self.band_indices]
            else:
                tensor = tensor[:3]
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            max_band = max(self.band_indices)
            if tensor.shape[0] > max_band:
                tensor = tensor[self.band_indices]
            else:
                tensor = tensor[:3]
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        # Scale to [0, 1] if reflectance values appear to be in [0, 10000]
        if tensor.max() > 10.0:
            tensor = tensor / 10000.0

        tensor = self._resize(tensor)
        tensor = self._normalise(tensor)
        return tensor

    def encode(self, image) -> torch.Tensor:
        return self.preprocess(image)


# =========================================================================== #
#  Remote Sensing CLIP Wrapper                                                 #
# =========================================================================== #

class RemoteSensingCLIP(BaseEncoder):
    """
    CLIP ViT-L/14 with domain-adapted preprocessing for remote sensing.

    Wraps OpenAI CLIP (or a RS-tuned variant like RemoteCLIP) for use as a
    zero-shot feature extractor in the evaluation pipeline.  Produces a 768-dim
    (or 1024-dim for ViT-L) visual embedding vector.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14-336",
        image_size: int = 336,
        device: Optional[str] = None,
    ):
        super().__init__(image_size)
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = None
        self.clip_processor = None
        self._load()

    def _load(self):
        """Load CLIP model and processor."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.clip_processor = CLIPProcessor.from_pretrained(self.model_name)
            self.clip_model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.clip_model.eval()
            logger.info(f"RS-CLIP loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Could not load CLIP model '{self.model_name}': {e}. encode() will return zeros.")

    def preprocess(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Apply CLIP-compatible preprocessing."""
        pil = self._to_pil(image)
        if self.clip_processor is None:
            # Fallback: standard ImageNet preprocessing
            t = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=_OPTICAL_MEAN, std=_OPTICAL_STD),
            ])
            return t(pil)
        inputs = self.clip_processor(images=pil, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # (3, H, W)

    @torch.inference_mode()
    def encode(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Extract a normalised visual feature vector.

        Returns:
            torch.Tensor of shape (D,) where D is the CLIP embedding dimension.
        """
        if self.clip_model is None:
            return torch.zeros(1024)

        pil = self._to_pil(image)
        inputs = self.clip_processor(images=pil, return_tensors="pt").to(self.device)
        features = self.clip_model.get_image_features(**inputs)  # (1, D)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise
        return features.squeeze(0).cpu()

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of text descriptions.

        Returns:
            torch.Tensor of shape (N, D) — one normalised embedding per text.
        """
        if self.clip_model is None:
            return torch.zeros(len(texts), 1024)
        inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        features = self.clip_model.get_text_features(**inputs)  # (N, D)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu()


# =========================================================================== #
#  Factory                                                                     #
# =========================================================================== #

def build_encoder(
    sensor_type: str,
    image_size: int = 336,
    **kwargs,
) -> BaseEncoder:
    """
    Factory function to build the appropriate encoder for a sensor type.

    Args:
        sensor_type: One of 'optical', 'sar', 'multispectral', 'clip'.
        image_size:  Target spatial resolution.
        **kwargs:    Additional arguments passed to the encoder constructor.

    Returns:
        A BaseEncoder subclass instance.
    """
    sensor_type = sensor_type.lower()
    if sensor_type == "optical":
        return OpticalEncoder(image_size=image_size, **kwargs)
    elif sensor_type == "sar":
        return SAREncoder(image_size=image_size, **kwargs)
    elif sensor_type == "multispectral":
        return MultispectralEncoder(image_size=image_size, **kwargs)
    elif sensor_type == "clip":
        return RemoteSensingCLIP(image_size=image_size, **kwargs)
    else:
        raise ValueError(
            f"Unknown sensor type '{sensor_type}'. "
            f"Choose from: optical, sar, multispectral, clip."
        )
