"""Custom nnU-Net trainer variants for the three Track A configurations.

nnU-Net v2 selects a trainer class by name via the CLI's -tr flag, so these
need to be importable (nnUNetv2_train ... -tr PengyyTrainer, etc). Stock
nnU-Net 3d_fullres uses nnU-Net's own default trainer directly -- no class
needed here for that one.

NOTE: written without a working nnunetv2 install to check against (no CUDA
machine available today). The general subclassing pattern (override
build_network_architecture / num_epochs) is stable across recent nnU-Net v2
releases, but the exact signature should be diffed against whatever
nnunetv2 version ends up pinned before this is trusted.
"""

from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class PengyyTrainer(nnUNetTrainer):
    """48 base features instead of nnU-Net's default 32 -- the FeTA 2021
    3rd-place (`pengyy`) configuration. Everything else (loss, optimizer,
    augmentation, deep supervision) stays as nnU-Net's default.

    Expected to land near 72.1M parameters per Project_Pipeline_FeTA.md;
    verify by checking the printed parameter count against that once this
    actually runs, and flag it in the report if it doesn't match.
    """

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision: bool = True,
    ):
        arch_init_kwargs = dict(arch_init_kwargs)
        # nnU-Net's default plan uses 32 as the base feature count; this key
        # name is plan-dependent (usually "features_per_stage" or a scalar
        # under "n_conv_per_stage" territory) -- confirm against the actual
        # generated plans.json once plan_and_preprocess has run, and adjust
        # the override key if it doesn't match.
        if "features_per_stage" in arch_init_kwargs:
            base = arch_init_kwargs["features_per_stage"][0]
            scale = 48 / base
            arch_init_kwargs["features_per_stage"] = [
                int(round(f * scale)) for f in arch_init_kwargs["features_per_stage"]
            ]
        return nnUNetTrainer.build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision,
        )


class VanillaUNetTrainer(nnUNetTrainer):
    """Plain Cicek 2016 3D U-Net -- the project's floor baseline.

    No deep supervision, no residual connections, none of nnU-Net's
    self-configuration beyond patch size / resampling (which we keep, since
    re-deriving those from scratch is out of scope for "the floor").
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_deep_supervision = False


class NoiseFloorSeedTrainer(nnUNetTrainer):
    """Identical to stock nnU-Net, but the RNG seed is settable independently
    of the fold number, for the noise-floor runs (same config/fold, three
    different seeds).
    """

    def __init__(self, *args, seed_override: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if seed_override is not None:
            self.my_init_kwargs["seed_override"] = seed_override