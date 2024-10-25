from typing import Dict, Tuple, Type, Union

import torch
import numpy as np
from vllm.config import ModelConfig, WhisperConfig
from vllm.inputs.registry import InputContext
from vllm.logger import init_logger
from vllm.sequence import SequenceData
from vllm.transformers_utils.whisper_processor import cached_get_whisper_processor

from vllm.multimodal.base import MultiModalInputs, MultiModalPlugin

logger = init_logger(__name__)

class WhisperData:
    pass


def _get_dummy_seq_data(seq_len: int,
                        whisper_config: WhisperConfig) -> SequenceData:

    # '<|startoftranscript|><|en|><|transcribe|>'
    token_ids = [50258, 50259, 50360]
    return SequenceData(token_ids)


def _get_dummy_values(whisper_config: WhisperConfig) -> torch.Tensor:
    values_dtype = torch.float16

    return torch.zeros((30 * whisper_config.sample_rate), dtype=values_dtype)


def get_dummy_audio_data(
    seq_len: int,
    model_config: ModelConfig,
    whisper_config: WhisperConfig,
) -> Tuple[SequenceData, MultiModalData]:
    """Standard dummy data factory for image data (to be used in
    :meth:`vlm.multimodal.MultiModalRegistry.register_dummy_data`)."""
    seq_data = _get_dummy_seq_data(seq_len, whisper_config)
    values = _get_dummy_values(whisper_config)

    fake_mm_data = AudioData(values)
    return seq_data, fake_mm_data


class AudioData(MultiModalData):

    def __init__(self, audio: Union[np.array, torch.Tensor]) -> None:

        self.audio = audio

    def __repr__(self) -> str:
        return str(self.audio)


class AudioPlugin(MultiModalPlugin[AudioData]):

    def get_data_type(self) -> Type[AudioData]:
        return AudioData

    def _get_hf_whisper_processor(self, model_config: ModelConfig,
                                whisper_config: WhisperConfig):
        if whisper_config is None or whisper_config.whisper_processor is None:
            return None

        return cached_get_whisper_processor(
            whisper_config.whisper_processor,
            revision=whisper_config.whisper_processor_revision,
        )

    def _default_input_processor(
            self, data: AudioData, model_config: ModelConfig,
            whisper_config: WhisperConfig) -> Dict[str, torch.Tensor]:
        audio = data.audio

        processor = self._get_hf_whisper_processor(model_config, whisper_config)
        if processor is None:
            raise RuntimeError("No HuggingFace processor is available"
                                "to process the audio object")
        try:
            return processor(audio, return_tensors="pt", sampling_rate = 16000).to(model_config.dtype)
        except Exception:
            logger.error("Failed to process audio (%s)", audio)
            raise


    def _default_input_mapper(self, ctx: InputContext, data: object,
                              **mm_processor_kwargs) -> MultiModalInputs:
        raise NotImplementedError("There is no default audio input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens")
