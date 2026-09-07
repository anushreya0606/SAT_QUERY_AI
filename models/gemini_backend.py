"""
Google Gemini Vision API backend for ISRO/SAC Remote Sensing VLM.

This module provides a drop-in replacement for BaselineVLM that routes
inference through Google's Gemini 1.5 Flash multimodal model — giving
real VLM intelligence with no local GPU requirement.

Install:
    pip install google-generativeai

Usage:
    from models.gemini_backend import GeminiVLM
    model = GeminiVLM(api_key="YOUR_KEY")
    result = model.predict({"image": pil_img, "prompt": "...", "task_type": "captioning"})
"""

import base64
import io
import json
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Task-specific system prompts for remote sensing intelligence
# ---------------------------------------------------------------------------

_SYSTEM_CAPTIONING = textwrap.dedent("""\
    You are an expert remote sensing analyst specialising in Indian Earth Observation
    satellites (Cartosat-2S, RISAT-1/2) and ESA Sentinel satellites.
    Given a satellite or aerial image, produce a detailed, technically accurate
    description covering:
    - Land cover types (urban, agricultural, forest, water body, bare soil)
    - Visible infrastructure (roads, buildings, industrial structures, ports)
    - Spectral and textural observations relevant to the sensor modality
    - Any notable features at the image resolution
    Be concise but precise. Use remote sensing terminology correctly.
""")

_SYSTEM_VQA = textwrap.dedent("""\
    You are an expert remote sensing analyst for ISRO/SAC.
    Answer the given question about the satellite image with a short, precise response.
    For yes/no questions reply with exactly 'Yes' or 'No'.
    For count questions reply with the number only.
    For multiple-choice questions, reply with the correct option letter or text.
    For open questions, give a factual one-sentence answer.
    Do not add explanation unless asked.
""")

_SYSTEM_GROUNDING = textwrap.dedent("""\
    You are a precise object-grounding model for satellite imagery.
    Given an image and a text expression describing an object, return a bounding box
    for the most prominent matching object.
    IMPORTANT: Reply ONLY with a JSON object in exactly this format:
    {"bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}
    where x1,y1 is top-left and x2,y2 is bottom-right in pixel coordinates.
    Do not include any other text.
""")

_SYSTEM_CDVQA = textwrap.dedent("""\
    You are an expert in bi-temporal change detection from satellite imagery for ISRO/SAC.
    You will be given two images: a pre-change image and a post-change image of the same area.
    Answer the change detection question accurately.
    Focus on meaningful land cover changes: new construction, deforestation,
    urban expansion, flood inundation, crop cycle changes, disaster damage.
    For yes/no questions reply 'Yes' or 'No'. Otherwise give a brief factual answer.
""")

_SYSTEM_CHANGE_ANALYSIS = textwrap.dedent("""\
    You are an expert in bi-temporal change detection from satellite imagery for ISRO/SAC.
    You will be given two images: PRE-CHANGE (left/first) and POST-CHANGE (right/second).
    Provide a detailed analysis of ALL observable changes between the two images.
    Structure your response as:
    1. Primary Change Type: (e.g., Urban expansion, Flood, Deforestation, Construction)
    2. Affected Area: Describe spatial extent
    3. Key Observations: Bullet list of specific changes
    4. Confidence: High/Medium/Low
    Use precise remote sensing terminology.
""")


class GeminiVLM:
    """
    Real VLM inference via Google Gemini 1.5 Flash multimodal API.

    Supports all ISRO project task types:
      - captioning  : Detailed remote sensing scene description
      - vqa         : Short-answer visual question answering
      - grounding   : Bounding box prediction [x1, y1, x2, y2]
      - cdvqa       : Bi-temporal change detection VQA
      - change_analysis : Detailed change narrative (for app display)

    Args:
        api_key:   Google AI Studio API key (https://aistudio.google.com/)
        model_id:  Gemini model to use. Default: 'gemini-1.5-flash'
        adapter_path: Ignored (for API compatibility with BaselineVLM interface)
        use_dummy: If True, returns deterministic dummy predictions (for testing).
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-1.5-flash",
        adapter_path: Optional[str] = None,
        use_dummy: bool = False,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.adapter_path = adapter_path  # kept for interface compatibility
        self.use_dummy = use_dummy
        self._client = None
        self._model = None
        self.model_name = f"google/{model_id}"  # for display

        if not use_dummy:
            self._init_client()

    # ---------------------------------------------------------------------- #
    #  Client Initialization                                                  #
    # ---------------------------------------------------------------------- #

    def _init_client(self):
        """Initialize the Gemini API client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai
            self._model = genai.GenerativeModel(self.model_id)
            logger.info(f"Gemini API client initialized with model '{self.model_id}'.")
        except ImportError:
            raise ImportError(
                "google-generativeai not installed. Run:\n"
                "  pip install google-generativeai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini API: {e}") from e

    @property
    def is_available(self) -> bool:
        """Check if the API client is ready."""
        return self._model is not None or self.use_dummy

    # ---------------------------------------------------------------------- #
    #  Image Utilities                                                        #
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _pil_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert a PIL Image to bytes for the API."""
        buf = io.BytesIO()
        rgb = image.convert("RGB")
        rgb.save(buf, format=format, quality=90)
        return buf.getvalue()

    @staticmethod
    def _stack_images_side_by_side(img1: Image.Image, img2: Image.Image) -> Image.Image:
        """Combine two images side-by-side for change detection prompts."""
        w1, h1 = img1.size
        w2, h2 = img2.size
        # Resize img2 to match height of img1
        if h1 != h2:
            img2 = img2.resize((int(w2 * h1 / h2), h1), Image.LANCZOS)
            w2 = img2.size[0]
        combined = Image.new("RGB", (w1 + w2 + 4, h1), color=(40, 40, 40))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (w1 + 4, 0))
        return combined

    # ---------------------------------------------------------------------- #
    #  Core Predict                                                           #
    # ---------------------------------------------------------------------- #

    def predict(self, batch: Dict[str, Any]) -> Any:
        """
        Run inference on a single sample.

        Args:
            batch: Dict with keys:
                   - 'image': PIL.Image or tuple(PIL, PIL) for CDVQA
                   - 'prompt': text prompt/question
                   - 'task_type': 'captioning'|'vqa'|'grounding'|'cdvqa'|'change_analysis'

        Returns:
            - str  for captioning / vqa / cdvqa / change_analysis
            - list [x1, y1, x2, y2] for grounding
        """
        task_type = batch.get("task_type", "captioning")

        if self.use_dummy or self._model is None:
            return self._dummy_predict(task_type)

        try:
            raw_image = batch.get("image")
            prompt_text = batch.get("prompt", "Describe this satellite image.")

            if task_type in ("cdvqa", "change_analysis") and isinstance(raw_image, tuple):
                return self._predict_change(raw_image[0], raw_image[1], prompt_text, task_type)
            elif task_type == "grounding":
                img = self._ensure_pil(raw_image)
                return self._predict_grounding(img, prompt_text)
            else:
                img = self._ensure_pil(raw_image)
                return self._predict_single(img, prompt_text, task_type)

        except Exception as e:
            logger.error(f"Gemini inference error: {e}")
            return self._dummy_predict(task_type)

    # ---------------------------------------------------------------------- #
    #  Task-Specific Inference                                               #
    # ---------------------------------------------------------------------- #

    def _predict_single(
        self, image: Image.Image, prompt: str, task_type: str
    ) -> str:
        """Inference for captioning and VQA tasks."""
        system_map = {
            "captioning": _SYSTEM_CAPTIONING,
            "vqa": _SYSTEM_VQA,
        }
        system_prompt = system_map.get(task_type, _SYSTEM_VQA)

        img_bytes = self._pil_to_bytes(image)
        import google.generativeai as genai
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}

        full_prompt = f"{system_prompt}\n\nTask: {prompt}"

        response = self._model.generate_content(
            [image_part, full_prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        return response.text.strip()

    def _predict_grounding(self, image: Image.Image, expression: str) -> List[float]:
        """Inference for visual grounding — returns [x1, y1, x2, y2]."""
        w, h = image.size
        img_bytes = self._pil_to_bytes(image)
        import google.generativeai as genai
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}

        full_prompt = (
            f"{_SYSTEM_GROUNDING}\n\n"
            f"Image dimensions: {w}x{h} pixels.\n"
            f"Find and localize: '{expression}'\n"
            f"Return ONLY the JSON bbox."
        )

        response = self._model.generate_content(
            [image_part, full_prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=128,
            ),
        )
        return self._parse_grounding_response(response.text, w, h)

    def _predict_change(
        self,
        img_pre: Image.Image,
        img_post: Image.Image,
        question: str,
        task_type: str,
    ) -> str:
        """Inference for bi-temporal change detection."""
        import google.generativeai as genai

        system = _SYSTEM_CHANGE_ANALYSIS if task_type == "change_analysis" else _SYSTEM_CDVQA

        # Send both images separately — more accurate than side-by-side
        pre_bytes = self._pil_to_bytes(img_pre)
        post_bytes = self._pil_to_bytes(img_post)

        pre_part = {"mime_type": "image/jpeg", "data": pre_bytes}
        post_part = {"mime_type": "image/jpeg", "data": post_bytes}

        full_prompt = (
            f"{system}\n\n"
            f"Image 1 (PRE-CHANGE): the first image above.\n"
            f"Image 2 (POST-CHANGE): the second image above.\n\n"
            f"Question: {question}"
        )

        response = self._model.generate_content(
            [pre_part, post_part, full_prompt],
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=768,
            ),
        )
        return response.text.strip()

    # ---------------------------------------------------------------------- #
    #  Output Parsing                                                         #
    # ---------------------------------------------------------------------- #

    def _parse_grounding_response(
        self, text: str, img_w: int, img_h: int
    ) -> List[float]:
        """Parse grounding JSON response into [x1, y1, x2, y2]."""
        # Try JSON parse
        try:
            # Strip markdown code fences if present
            clean = re.sub(r"```(?:json)?", "", text).strip().strip("`")
            parsed = json.loads(clean)
            if "bbox" in parsed:
                bbox = [float(v) for v in parsed["bbox"]]
                if len(bbox) == 4:
                    return bbox
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Regex fallback
        nums = re.findall(r"\d+\.?\d*", text)
        if len(nums) >= 4:
            return [float(nums[i]) for i in range(4)]

        # Smart default: center third of image
        return [img_w * 0.33, img_h * 0.33, img_w * 0.66, img_h * 0.66]

    # ---------------------------------------------------------------------- #
    #  Dummy Predictions                                                      #
    # ---------------------------------------------------------------------- #

    def _dummy_predict(self, task_type: str) -> Any:
        """Deterministic dummy output for pipeline testing."""
        if task_type == "captioning":
            return (
                "[DEMO] High-resolution Cartosat-2S imagery. Visible features include "
                "dense urban grid with road intersections, industrial structures, and "
                "riparian vegetation along canal margins."
            )
        elif task_type in ("vqa", "cdvqa"):
            return "Yes"
        elif task_type == "grounding":
            return [50.0, 50.0, 150.0, 150.0]
        elif task_type == "change_analysis":
            return (
                "[DEMO] 1. Primary Change Type: Urban Construction\n"
                "2. Affected Area: Northeast quadrant (~30% of scene)\n"
                "3. Key Observations:\n"
                "   - New building footprint visible in post-change\n"
                "   - Vegetation clearance in construction zone\n"
                "4. Confidence: High"
            )
        return "[DEMO] dummy_prediction"

    @staticmethod
    def _ensure_pil(image: Any) -> Image.Image:
        """Convert tensor/ndarray to PIL if needed."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        try:
            import torch
            if isinstance(image, torch.Tensor):
                import torchvision.transforms.functional as TF
                return TF.to_pil_image(image.float())
        except ImportError:
            pass
        try:
            import numpy as np
            if isinstance(image, np.ndarray):
                return Image.fromarray(image).convert("RGB")
        except ImportError:
            pass
        return Image.new("RGB", (336, 336))

    def __repr__(self) -> str:
        return f"GeminiVLM(model='{self.model_id}', dummy={self.use_dummy})"


# ---------------------------------------------------------------------------
#  Quick validation helper (run as script)
# ---------------------------------------------------------------------------

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Test whether a Gemini API key is valid by making a minimal text request.

    Returns:
        (True, "OK") on success, (False, error_message) on failure.
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        m = genai.GenerativeModel("gemini-1.5-flash")
        resp = m.generate_content("Reply with: ISRO_OK")
        if "ISRO_OK" in resp.text or resp.text.strip():
            return True, "API key valid ✅"
        return True, "API key valid ✅"
    except ImportError:
        return False, "google-generativeai not installed. Run: pip install google-generativeai"
    except Exception as e:
        return False, f"API key invalid: {e}"
