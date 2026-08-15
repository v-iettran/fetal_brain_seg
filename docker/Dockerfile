# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY pyproject.toml /workspace/pyproject.toml
RUN pip install --no-cache-dir "openevolve==0.3.2" "openai>=1,<2" nibabel scipy "scikit-image>=0.26" optuna pyyaml dacite matplotlib tqdm pytest

COPY config.py /workspace/config.py
COPY src /workspace/src
COPY openevolve /workspace/openevolve
COPY tests /workspace/tests

ENV PYTHONPATH=/workspace:/workspace/openevolve
ENV FETA_PROFILE=production
ENV NVIDIA_VISIBLE_DEVICES=all

# Data is mounted read-only at /workspace/mri_gz
# Cache is mounted at /workspace/cache
# Frozen hashes are verified at evaluator start.

WORKDIR /workspace
CMD ["python", "openevolve/run_evolution.py"]
