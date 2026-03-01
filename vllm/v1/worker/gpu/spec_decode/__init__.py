# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.use_eagle():
        if (
            speculative_config.method == "eagle3"
            and speculative_config.use_ngram_drafter()
        ):
            from vllm.v1.spec_decode.ngram_eagle3 import NgramEagle3Speculator

            return NgramEagle3Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.eagle import EagleSpeculator

        return EagleSpeculator(vllm_config, device)
    raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
