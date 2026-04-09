# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-04-09

### Added

- SigLIP2 SO400M vision engine with lazy model loading
- `image_contains` tool — sigmoid score for "does image contain X?"
- `image_classify` tool — score image against N candidate labels
- `image_compare` tool — cosine similarity between two images
- `image_score_batch` tool — score N images against one query
- `eyes_status` tool — health check (model, device, VRAM)
- FastMCP v3 server with STDIO transport
- Configurable model, device, cache dir, and threshold via environment variables
