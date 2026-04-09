"""
SigLIP2 vision engine — discriminative image scoring.

Wraps SigLIP2 as a measurement instrument, not a conversational model.
Returns calibrated sigmoid scores: one image-text pair → one float.

No MCP dependency. Can be used standalone or from the MCP server.

Key design decisions:
  - Lazy model loading (first call triggers download/load)
  - Sigmoid scores are independent per query (not softmax)
  - All images converted to RGB (alpha stripped)
  - Thread-safe via torch.no_grad context per call
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults (overridable via env vars)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_ID = "google/siglip2-so400m-patch14-384"
DEFAULT_CACHE_DIR = os.environ.get("AI_EYES_MODEL_DIR", None)
DEFAULT_DEVICE = os.environ.get("AI_EYES_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_THRESHOLD = float(os.environ.get("AI_EYES_DEFAULT_THRESHOLD", "0.02"))


class SigLIPEngine:
    """SigLIP2 vision scoring engine.

    Lazy-loads the model on first inference call. All scoring methods
    return raw sigmoid floats — the caller decides what threshold means.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        device: str = DEFAULT_DEVICE,
    ):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = device
        self._model = None
        self._processor = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self):
        """Load model and processor on first use."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoProcessor

        print(f"[ai-eyes] Loading {self.model_id} ...", file=sys.stderr)

        kwargs = {}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self._processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModel.from_pretrained(self.model_id, **kwargs)
        self._model = self._model.eval().to(self.device)

        param_count = sum(p.numel() for p in self._model.parameters())
        print(f"[ai-eyes] Loaded on {self.device}, {param_count/1e6:.0f}M params",
              file=sys.stderr)

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image from path, convert to RGB."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return Image.open(path).convert("RGB")

    def score(self, image_path: str, query: str) -> float:
        """Score a single image against a single text query.

        Returns a sigmoid probability (0-1). Independent per query —
        not relative to other queries. Higher = stronger match.
        """
        self._ensure_loaded()

        image = self._load_image(image_path)
        inputs = self._processor(
            text=[query],
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            prob = torch.sigmoid(outputs.logits_per_image[0, 0]).item()

        return prob

    def score_multi(self, image_path: str, queries: list[str]) -> dict[str, float]:
        """Score one image against multiple text queries.

        Returns a dict mapping each query to its independent sigmoid score.
        Scores are NOT softmax — each query is evaluated independently.
        """
        self._ensure_loaded()

        image = self._load_image(image_path)
        inputs = self._processor(
            text=queries,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.sigmoid(outputs.logits_per_image[0]).cpu().numpy()

        return {q: float(p) for q, p in zip(queries, probs)}

    def score_batch(self, image_paths: list[str], query: str) -> list[float]:
        """Score multiple images against a single text query.

        Encodes the text once, scores each image independently.
        Returns list of sigmoid scores in same order as input paths.
        """
        self._ensure_loaded()

        scores = []
        for path in image_paths:
            scores.append(self.score(path, query))
        return scores

    def embed_image(self, image_path: str) -> np.ndarray:
        """Extract the image embedding vector.

        Returns a 1D numpy array (normalized). Use for cosine similarity
        comparisons between images.
        """
        self._ensure_loaded()

        image = self._load_image(image_path)
        inputs = self._processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)

        return emb.cpu().numpy().squeeze()

    def compare(self, image_a: str, image_b: str) -> float:
        """Compute cosine similarity between two images.

        Returns a float in [-1, 1]. Higher = more similar.
        """
        emb_a = self.embed_image(image_a)
        emb_b = self.embed_image(image_b)
        return float(np.dot(emb_a, emb_b))

    def status(self) -> dict:
        """Return engine status info."""
        info = {
            "model_id": self.model_id,
            "device": self.device,
            "loaded": self.loaded,
            "cache_dir": self.cache_dir or "default",
        }
        if self.loaded:
            param_count = sum(p.numel() for p in self._model.parameters())
            info["parameters"] = f"{param_count/1e6:.0f}M"
            if self.device == "cuda" and torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
                info["vram_mb"] = round(vram_mb)
        return info
