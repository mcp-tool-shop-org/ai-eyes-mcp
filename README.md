# ai-eyes-mcp

Grounded visual evaluator MCP server. Gives Claude honest image judgment via SigLIP2.

## The Problem

When Claude needs to verify what's in an image — "does this sprite have a sword?", "is there a login button?" — generative VLMs (LLaVA, GPT-4V) hallucinate confident answers. They complete narratives, not observations. LLaVA 13B reported "the character is holding a greatsword" on images where no weapon existed, at 0.90 confidence, on every single crop.

## The Solution

SigLIP2 is a discriminative vision model. It doesn't generate text — it measures similarity between an image and a text description, returning a calibrated sigmoid score. When the weapon is present, the score is 10-100x higher than when it's absent. When it can't tell, the score is low. It doesn't hallucinate.

This MCP server wraps SigLIP2 as tools that any Claude workflow can call.

## Tools

| Tool | What it does |
|------|-------------|
| `image_contains` | "Does this image contain X?" → sigmoid score |
| `image_classify` | Score image against N candidate labels |
| `image_compare` | Cosine similarity between two images |
| `image_score_batch` | Score N images against one query |
| `eyes_status` | Health check: model, device, loaded state |

## Quick Start

```bash
pip install -e .
ai-eyes-mcp  # starts STDIO server
```

### Claude Code config

```json
{
  "mcpServers": {
    "ai-eyes": {
      "command": "ai-eyes-mcp",
      "env": {
        "AI_EYES_MODEL_DIR": "F:/AI-Models/huggingface"
      }
    }
  }
}
```

## Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `AI_EYES_MODEL_ID` | `google/siglip2-so400m-patch14-384` | HuggingFace model |
| `AI_EYES_MODEL_DIR` | HF default cache | Model cache directory |
| `AI_EYES_DEVICE` | `cuda` (auto) | torch device |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Default threshold for `image_contains` |

## How Scores Work

SigLIP2 uses **sigmoid** scoring, not softmax. Each image-text pair gets an independent probability (0-1):

- **High score** (>0.1): Strong visual match — the described object is likely present
- **Low score** (<0.01): No match — the object is not visible
- **Mid score** (0.01-0.1): Ambiguous — may need human review

Scores are NOT relative. Multiple queries can score high on the same image (e.g., an image with both a sword and a shield).

The default threshold (0.02) was calibrated on pixel art sprites. For photographic images, a higher threshold (0.1-0.3) may work better. The caller can override per-call.

## Requirements

- Python >= 3.10
- CUDA GPU recommended (~800MB VRAM at FP16)
- CPU fallback available (slower, ~10x)
- Model downloads ~1.6GB on first use
