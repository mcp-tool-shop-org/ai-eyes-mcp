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
        "AI_EYES_MODEL_DIR": "/path/to/model/cache"
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
| `AI_EYES_DEVICE` | `cuda` (auto-detect) | torch device |
| `AI_EYES_DEFAULT_THRESHOLD` | `0.02` | Default threshold for `image_contains` |

## How Scores Work

SigLIP2 uses **sigmoid** scoring, not softmax. Each image-text pair gets an independent probability (0-1):

- **High score** (>0.1): Strong visual match — the described object is likely present
- **Low score** (<0.01): No match — the object is not visible
- **Mid score** (0.01-0.1): Ambiguous — may need human review

Scores are NOT relative. Multiple queries can score high on the same image (e.g., an image with both a sword and a shield).

The default threshold (0.02) was calibrated on pixel art sprites. For photographic images, a higher threshold (0.1-0.3) may work better. The caller can override per-call.

## Tool Reference

### `image_contains`

```
image_contains(image_path, query, threshold=0.02)
```

Check if an image contains something described by the query. Returns an independent sigmoid score (0-1).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | yes | Absolute path to image file |
| `query` | string | yes | What to look for (e.g., "a person holding a sword") |
| `threshold` | float | no | Score threshold for present verdict (default 0.02) |

Returns: `{present: bool, score: float, threshold: float, query: string}`

### `image_classify`

```
image_classify(image_path, labels)
```

Score an image against multiple candidate labels. Returns independent sigmoid scores — NOT softmax.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_path` | string | yes | Absolute path to image file |
| `labels` | string[] | yes | Candidate labels to score (max 20) |

Returns: `{scores: {label: float}, best: string, best_score: float}`

### `image_compare`

```
image_compare(image_a, image_b)
```

Compute visual similarity between two images using cosine similarity of SigLIP2 embeddings.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_a` | string | yes | Absolute path to first image |
| `image_b` | string | yes | Absolute path to second image |

Returns: `{similarity: float, image_a: string, image_b: string}`

### `image_score_batch`

```
image_score_batch(image_paths, query, threshold=0.02)
```

Score multiple images against a single query. Max 100 images per call.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image_paths` | string[] | yes | List of absolute image paths |
| `query` | string | yes | What to look for |
| `threshold` | float | no | Score threshold (default 0.02) |

Returns: `{query, threshold, total, scored, present, absent, errors, results: [{path, score, present}]}`

### `eyes_status`

```
eyes_status()
```

Check server status. Does not trigger model loading.

Returns: `{model_id, device, loaded, cache_dir, parameters?, vram_mb?}`

## Security and Trust

This tool operates **locally only**.

- **Data touched:** Local image files (read-only), HuggingFace model cache (downloaded once)
- **No network egress** at runtime — model downloads happen once on first use, then all inference is local
- **No secrets handling** — does not read, store, or transmit credentials or API keys
- **No telemetry** — nothing is collected or sent
- **No file mutation** — image files are opened read-only, never modified
- **No dangerous actions** — no delete, kill, or restart operations
- **Structured errors only** — stack traces never exposed to clients

## Requirements

- Python >= 3.10
- CUDA GPU recommended (~800MB VRAM at FP16)
- CPU fallback available (slower, ~10x)
- Model downloads ~1.6GB on first use

## License

MIT

---

Built by [MCP Tool Shop](https://mcp-tool-shop.github.io/)
