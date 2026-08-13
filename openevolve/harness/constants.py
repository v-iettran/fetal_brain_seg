"""Frozen constants. Changing these requires a deliberate edit to frozen code."""

from __future__ import annotations

NUM_CLASSES = 8
IN_CHANNELS = 1
FOREGROUND_CLASSES = tuple(range(1, NUM_CLASSES))
CONTRACT_FUNCTIONS = (
    "build_model",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "build_sampler",
    "build_augmentation",
)
ALLOWED_IMPORT_ROOTS = (
    "torch",
    "numpy",
    "np",
    "math",
    "random",
    "typing",
    "scipy",
    "scipy.ndimage",
)
FORBIDDEN_IMPORT_SUBSTRINGS = (
    "nnunet",
    "monai",
    "segmentation_models_pytorch",
    "timm",
    "torch.hub",
    "urllib",
    "requests",
    "subprocess",
    "os.system",
    "huggingface",
    "pretrained",
)
EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"
