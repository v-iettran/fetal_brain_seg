#!/usr/bin/env bash
set -euo pipefail
# Production A6000 launch. Mounts data read-only; cache and results are writable.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-feta-openevolve:latest}"
docker build -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"
docker run --rm --gpus all \
  -e FETA_PROFILE=production \
  -v "$ROOT/mri_gz:/workspace/mri_gz:ro" \
  -v "$ROOT/cache:/workspace/cache" \
  -v "$ROOT/results:/workspace/results" \
  -v "$ROOT/openevolve:/workspace/openevolve:ro" \
  -v "$ROOT/src:/workspace/src:ro" \
  "$IMAGE" "$@"
