"""
ai-eyes MCP server — grounded visual evaluator.

Exposes SigLIP2 as MCP tools for honest image judgment.
Returns calibrated scores, not prose. Measures, not narrates.

Tools:
  image_contains    — "Does this image contain X?"
  image_classify    — Score image against N candidate labels
  image_compare     — Cosine similarity between two images
  image_score_batch — Score N images against one query
  eyes_status       — Health check
"""

import os
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field

from ai_eyes_mcp.engine import SigLIPEngine, DEFAULT_MODEL_ID, DEFAULT_CACHE_DIR, DEFAULT_DEVICE, DEFAULT_THRESHOLD

# ---------------------------------------------------------------------------
# Server + engine setup
# ---------------------------------------------------------------------------

mcp = FastMCP(name="ai-eyes")

engine = SigLIPEngine(
    model_id=os.environ.get("AI_EYES_MODEL_ID", DEFAULT_MODEL_ID),
    cache_dir=os.environ.get("AI_EYES_MODEL_DIR", DEFAULT_CACHE_DIR),
    device=os.environ.get("AI_EYES_DEVICE", DEFAULT_DEVICE),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool
def image_contains(
    image_path: Annotated[str, Field(description="Absolute path to the image file")],
    query: Annotated[str, Field(description="What to look for (e.g. 'a person holding a sword', 'a login button', 'a red car')")],
    threshold: Annotated[float, Field(description="Score threshold for 'present' verdict (default 0.02, tuned for pixel art)")] = DEFAULT_THRESHOLD,
) -> dict:
    """Check if an image contains something described by the query.

    Returns an independent sigmoid score (0-1) — higher means stronger visual match.
    The score is NOT relative to other queries. Each query is evaluated on its own.

    Uses SigLIP2 discriminative vision — measures visual similarity, does not
    generate text or hallucinate. If the image doesn't clearly match, the score
    will be low.
    """
    try:
        score = engine.score(image_path, query)
    except FileNotFoundError:
        raise ToolError(f"Image not found: {image_path}")
    except Exception as e:
        raise ToolError(f"Scoring failed: {e}")

    return {
        "present": score > threshold,
        "score": round(score, 4),
        "threshold": threshold,
        "query": query,
    }


@mcp.tool
def image_classify(
    image_path: Annotated[str, Field(description="Absolute path to the image file")],
    labels: Annotated[list[str], Field(description="Candidate labels to score (e.g. ['cat', 'dog', 'bird'])")],
) -> dict:
    """Score an image against multiple candidate labels.

    Returns independent sigmoid scores for EACH label — these are NOT softmax.
    Multiple labels can score high simultaneously (e.g. an image with both a
    cat and a dog). A label scoring low means it's NOT confidently present.

    This is measurement, not conversation. The model looks at the pixels and
    reports similarity to each text description.
    """
    if not labels:
        raise ToolError("At least one label is required")
    if len(labels) > 20:
        raise ToolError("Maximum 20 labels per call")

    try:
        scores = engine.score_multi(image_path, labels)
    except FileNotFoundError:
        raise ToolError(f"Image not found: {image_path}")
    except Exception as e:
        raise ToolError(f"Classification failed: {e}")

    rounded = {k: round(v, 4) for k, v in scores.items()}
    best_label = max(rounded, key=rounded.get)

    return {
        "scores": rounded,
        "best": best_label,
        "best_score": rounded[best_label],
    }


@mcp.tool
def image_compare(
    image_a: Annotated[str, Field(description="Absolute path to the first image")],
    image_b: Annotated[str, Field(description="Absolute path to the second image")],
) -> dict:
    """Compute visual similarity between two images.

    Returns cosine similarity (-1 to 1) of their SigLIP2 embeddings.
    Higher values mean the images look more alike to the vision model.

    Use cases: comparing sprite poses, checking if two renders match,
    detecting visual duplicates.
    """
    for path, name in [(image_a, "image_a"), (image_b, "image_b")]:
        from pathlib import Path
        if not Path(path).exists():
            raise ToolError(f"{name} not found: {path}")

    try:
        similarity = engine.compare(image_a, image_b)
    except Exception as e:
        raise ToolError(f"Comparison failed: {e}")

    return {
        "similarity": round(similarity, 4),
        "image_a": image_a,
        "image_b": image_b,
    }


@mcp.tool
def image_score_batch(
    image_paths: Annotated[list[str], Field(description="List of absolute image file paths")],
    query: Annotated[str, Field(description="What to look for in each image")],
    threshold: Annotated[float, Field(description="Score threshold for 'present' verdict")] = DEFAULT_THRESHOLD,
) -> dict:
    """Score multiple images against a single query.

    Returns per-image scores and a summary. Useful for batch evaluation,
    cherry-picking the best match, or scanning a directory of images.
    """
    if not image_paths:
        raise ToolError("At least one image path is required")
    if len(image_paths) > 100:
        raise ToolError("Maximum 100 images per batch")

    results = []
    errors = []
    for path in image_paths:
        try:
            score = engine.score(path, query)
            results.append({
                "path": path,
                "score": round(score, 4),
                "present": score > threshold,
            })
        except FileNotFoundError:
            errors.append({"path": path, "error": "not found"})
        except Exception as e:
            errors.append({"path": path, "error": str(e)})

    present_count = sum(1 for r in results if r["present"])

    return {
        "query": query,
        "threshold": threshold,
        "total": len(image_paths),
        "scored": len(results),
        "present": present_count,
        "absent": len(results) - present_count,
        "errors": len(errors),
        "results": results,
        "error_details": errors if errors else None,
    }


@mcp.tool
def eyes_status() -> dict:
    """Check ai-eyes server status.

    Returns model info, device, and whether the model is currently loaded.
    The model loads lazily on first tool call — this tool does NOT trigger loading.
    """
    return engine.status()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    mcp.run()
