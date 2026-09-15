from __future__ import annotations
from .dflash import DFlashConfig, DFlashModel
from ..model.config import no_default


class DFlash2Config(DFlashConfig):
    """
    DFlash 2 draft model (block diffusion drafter with grouped dynamic convolution and a
    candidate path selector). Checkpoint arch string: DFlash2DraftModel.

    Adds over DFlash v1 (all under dflash_config-> or top level, mirroring the reference
    z-lab/dflash implementation):
      - conv_kernel_size, conv_group_size: per-sublayer grouped dynamic depthwise convolution
      - selector_rank, selector_top_k: candidate path selector (bilinear adjacent-pair scoring)
    """

    arch_string = "DFlash2DraftModel"

    def __init__(
        self,
        directory: str,
        model_classes: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            directory,
            model_classes or {"text": DFlash2Model},
            **kwargs
        )

        # DFlash 2 additions
        self.conv_kernel_size = self.read_cfg(int, ["dflash_config->conv_kernel_size", "conv_kernel_size"], no_default)
        self.conv_group_size = self.read_cfg(int, ["dflash_config->conv_group_size", "conv_group_size"], no_default)
        self.selector_rank = self.read_cfg(int, ["dflash_config->selector_rank", "selector_rank"], no_default)
        self.selector_top_k = self.read_cfg(int, ["dflash_config->selector_top_k", "selector_top_k"], no_default)


class DFlash2Model(DFlashModel):
    """
    DFlash 2 draft. v1 backbone (fc + SWA layers + shared target head) plus:
      - GroupedDynamicCausalConv wrapped around each attention and MLP sublayer
      - CandidateSelector producing the drafted token path (greedy walk or sampled walk
        with q over the per-position candidate lists, for the lossless verify)
    Conv and selector parameters stay in fp16 (small, uncalibrated); the backbone Linears
    quantize normally.
    """

    config_class = DFlash2Config

    def __init__(
        self,
        config: DFlash2Config,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.caps.update({
            "dflash2_draft": True,
        })

    # Conv/selector wiring, propose(), and the fp16 exclusion set land with the module
    # implementation (Tasks 2.3-2.5).
