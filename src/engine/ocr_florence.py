"""OCR engine — Florence-2 (Microsoft VLM, 0.23B).

Florence-2 is a lightweight vision-language model that natively supports
<OCR_WITH_REGION> — returns text with precise quadrilateral coordinates,
perfect for the translation overlay use case.

Usage:
  $env:OCR_BACKEND = "florence"
  python main.py
"""

import os
import numpy as np
from typing import List, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

from src.engine.ocr import OcrResult, OcrBackend


class FlorenceOCREngine(OcrBackend):
    """OCR via Microsoft Florence-2-base VLM."""

    def __init__(self, model_path: str = "Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Florence-2] 加载模型: {model_path} (device={self.device})",
              flush=True)

        # Resolve relative path to absolute (relative to project root)
        if not os.path.isabs(model_path):
            # Try relative to this file's directory first (src/engine/ → project/)
            project_root = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ))
            model_path = os.path.join(project_root, model_path)

        self._processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda"
            else torch.float32,
        ).to(self.device).eval()

        print("[Florence-2] 模型加载完成", flush=True)

    @property
    def available(self) -> bool:
        return True

    def recognize(self, image_bgr: np.ndarray) -> List[OcrResult]:
        """Run Florence-2 OCR with region output.

        Args:
            image_bgr: BGR image as numpy array (H, W, 3).

        Returns:
            List of OcrResult with bbox, original_text, confidence.
        """
        h, w = image_bgr.shape[:2]

        # BGR → RGB → PIL
        image_rgb = image_bgr[:, :, ::-1]
        pil_image = Image.fromarray(image_rgb)

        # ── Preprocess: pad to square (required by DaViT vision encoder) ──
        pad_offset_x, pad_offset_y, pad_size = self._pad_params(w, h)
        pil_image = self._pad_to_square(pil_image)
        padded_w, padded_h = pil_image.size  # both == pad_size

        # Build inputs with <OCR_WITH_REGION> task prompt
        prompt = "<OCR_WITH_REGION>"
        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt",
        ).to(self.device, dtype=self._model.dtype if self.device == "cuda" else torch.float32)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False,
            )

        # Decode and post-process
        generated_text = self._processor.decode(
            generated_ids[0], skip_special_tokens=False,
        )

        parsed = self._processor.post_process_generation(
            text=generated_text,
            task="<OCR_WITH_REGION>",
            image_size=(padded_w, padded_h),
        )

        result = parsed.get("<OCR_WITH_REGION>", {})
        quad_boxes = result.get("quad_boxes", [])
        labels = result.get("labels", [])

        # Convert quadrilateral boxes → axis-aligned bounding boxes
        # (undo square-padding offset to map back to original image coords)
        results: List[OcrResult] = []
        for quad, text in zip(quad_boxes, labels):
            # Strip whitespace and special tokens (Florence-2 artifacts)
            text = text.strip()
            for tok in ("</s>", "<s>", "<pad>", "</s"):
                text = text.replace(tok, "")
            text = text.strip()
            if not text or len(text) <= 1:
                continue
            # quad: [x0,y0, x1,y1, x2,y2, x3,y3] — 8 values
            xs = [v - pad_offset_x for v in quad[0::2]]
            ys = [v - pad_offset_y for v in quad[1::2]]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            # Clamp to original image bounds
            bbox = (
                max(0, bbox[0]), max(0, bbox[1]),
                min(w, bbox[2]), min(h, bbox[3]),
            )
            results.append(OcrResult(
                bbox=bbox,
                original_text=text,
                confidence=0.85,  # Florence-2 doesn't expose per-box confidence
            ))

        results.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        print(f"[Florence-2] 检测到 {len(results)} 段", flush=True)
        for r in results[:15]:
            print(f"  -> {r.original_text[:60]}", flush=True)
        return results

    @staticmethod
    def _pad_params(w: int, h: int) -> tuple:
        """Return (offset_x, offset_y, padded_size) for square padding."""
        size = max(w, h)
        return (size - w) // 2, (size - h) // 2, size

    @staticmethod
    def _pad_to_square(image: Image.Image) -> Image.Image:
        """Pad image to square with white border (required by DaViT encoder)."""
        w, h = image.size
        if w == h:
            return image
        size = max(w, h)
        padded = Image.new("RGB", (size, size), (255, 255, 255))
        padded.paste(image, ((size - w) // 2, (size - h) // 2))
        return padded
