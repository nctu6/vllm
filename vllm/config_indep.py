import enum
import json
from dataclasses import dataclass, field, fields
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, List, Mapping,
                    Optional, Tuple, Type, Union)

import torch
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.transformers_utils.config import (ConfigFormat, get_config,
                                            get_hf_image_processor_config,
                                            get_hf_text_config)

logger = init_logger(__name__)

_GB = 1 << 30
_EMBEDDING_MODEL_MAX_NUM_BATCHED_TOKENS = 32768
_WHISPER_MAX_NUM_BATCHED_TOKENS = 448
_MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS = 5120


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Name or path of the huggingface model to use.
            It is also used as the content for `model_name` tag in metrics
            output when `served_model_name` is not specified.
        tokenizer: Name or path of the huggingface tokenizer to use.
        tokenizer_mode: Tokenizer mode. "auto" will use the fast tokenizer if
            available, "slow" will always use the slow tokenizer, and
            "mistral" will always use the tokenizer from `mistral_common`.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        dtype: Data type for model weights and activations. The "auto" option
            will use FP16 precision for FP32 and FP16 models, and BF16 precision
            for BF16 models.
        seed: Random seed for reproducibility.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id. If unspecified, will use the default
            version.
        code_revision: The specific revision to use for the model code on
            Hugging Face Hub. It can be a branch name, a tag name, or a
            commit id. If unspecified, will use the default version.
        rope_scaling: Dictionary containing the scaling configuration for the
            RoPE embeddings. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id. If unspecified, will use
            the default version.
        max_model_len: Maximum length of a sequence (including prompt and
            output). If None, will be derived from the model.
        quantization: Quantization method that was used to quantize the model
            weights. If None, we assume the model weights are not quantized.
        quantization_param_path: Path to JSON file containing scaling factors.
            Used to load KV cache scaling factors into the model when KV cache
            type is FP8_E4M3 on ROCm (AMD GPU). In the future these will also
            be used to load activation and weight scaling factors when the
            model dtype is FP8_E4M3 on ROCm.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
            If None, the user did not specify, so default to False.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use max_seq_len_to_capture instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_sliding_window: Whether to disable sliding window. If True,
            we will disable the sliding window functionality of the model.
            If the model does not support sliding window, this argument is
            ignored.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer.
        served_model_name: The model name used in metrics tag `model_name`,
            matches the model name exposed via the APIs. If multiple model
            names provided, the first name will be used. If not specified,
            the model name will be the same as `model`.
        limit_mm_per_prompt: Maximum number of data instances per modality
            per prompt. Only applicable for multimodal models.
        override_neuron_config: Initialize non default neuron config or
            override default neuron config that are specific to Neuron devices,
            this argument will be used to configure the neuron config that
            can not be gathered from the vllm arguments.
        config_format: The config format which shall be loaded.
            Defaults to 'auto' which defaults to 'hf'.
        mm_processor_kwargs: Arguments to be forwarded to the model's processor
            for multi-modal data, e.g., image processor.
    """

    def __init__(self,
                 model: str,
                 tokenizer: str,
                 tokenizer_mode: str,
                 trust_remote_code: bool,
                 dtype: Union[str, torch.dtype],
                 seed: int,
                 revision: Optional[str] = None,
                 code_revision: Optional[str] = None,
                 rope_scaling: Optional[dict] = None,
                 rope_theta: Optional[float] = None,
                 tokenizer_revision: Optional[str] = None,
                 max_model_len: Optional[int] = None,
                 spec_target_max_model_len: Optional[int] = None,
                 quantization: Optional[str] = None,
                 quantization_param_path: Optional[str] = None,
                 enforce_eager: Optional[bool] = None,
                 max_context_len_to_capture: Optional[int] = None,
                 max_seq_len_to_capture: Optional[int] = None,
                 max_logprobs: int = 20,
                 disable_sliding_window: bool = False,
                 skip_tokenizer_init: bool = False,
                 served_model_name: Optional[Union[str, List[str]]] = None,
                 limit_mm_per_prompt: Optional[Mapping[str, int]] = None,
                 use_async_output_proc: bool = True,
                 override_neuron_config: Optional[Dict[str, Any]] = None,
                 config_format: ConfigFormat = ConfigFormat.AUTO,
                 mm_processor_kwargs: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_mode = tokenizer_mode
        self.trust_remote_code = trust_remote_code
        self.seed = seed
        self.revision = revision
        self.code_revision = code_revision
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        # The tokenizer version is consistent with the model version by default.
        if tokenizer_revision is None:
            self.tokenizer_revision = revision
        else:
            self.tokenizer_revision = tokenizer_revision
        self.quantization = quantization
        self.quantization_param_path = quantization_param_path
        self.enforce_eager = enforce_eager
        if max_context_len_to_capture is not None:
            raise ValueError("`max_context_len_to_capture` is deprecated. "
                             "Use `max_seq_len_to_capture` instead.")
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_logprobs = max_logprobs
        self.disable_sliding_window = disable_sliding_window
        self.skip_tokenizer_init = skip_tokenizer_init

        self.hf_config = get_config(self.model, trust_remote_code, revision,
                                    code_revision, rope_scaling, rope_theta,
                                    config_format)
        self.hf_text_config = get_hf_text_config(self.hf_config)
        self.hf_image_processor_config = get_hf_image_processor_config(
            self.model, revision)
        self.dtype = _get_and_verify_dtype(self.hf_text_config, dtype)
        self.use_async_output_proc = use_async_output_proc
        self.mm_processor_kwargs = mm_processor_kwargs

        # Set enforce_eager to False if the value is unset.
        if self.enforce_eager is None:
            self.enforce_eager = False

        if (not self.disable_sliding_window
                and self.hf_text_config.model_type == "gemma2"
                and self.hf_text_config.sliding_window is not None):
            print_warning_once(
                "Gemma 2 uses sliding window attention for every odd layer, "
                "which is currently not supported by vLLM. Disabling sliding "
                "window and capping the max length to the sliding window size "
                f"({self.hf_text_config.sliding_window}).")
            self.disable_sliding_window = True

        self.max_model_len = _get_and_verify_max_len(
            hf_config=self.hf_text_config,
            max_model_len=max_model_len,
            disable_sliding_window=self.disable_sliding_window,
            sliding_window_len=self.get_hf_config_sliding_window(),
            spec_target_max_model_len=spec_target_max_model_len)
        self.served_model_name = get_served_model_name(model,
                                                       served_model_name)
        self.multimodal_config = self._init_multimodal_config(
            limit_mm_per_prompt)
        if not self.skip_tokenizer_init:
            self._verify_tokenizer_mode()

        self.is_attention_free = self._init_attention_free()
        self.has_inner_state = self._init_has_inner_state()

        self.override_neuron_config = override_neuron_config if is_neuron(
        ) else None
        self._verify_embedding_mode()
        self._verify_whisper_mode()
        self._verify_quantization()
        self._verify_cuda_graph()
        self._verify_bnb_config()

    def _init_multimodal_config(
        self, limit_mm_per_prompt: Optional[Mapping[str, int]]
    ) -> Optional["MultiModalConfig"]:
        architectures = getattr(self.hf_config, "architectures", [])
        if ModelRegistry.is_multimodal_model(architectures):
            return MultiModalConfig(limit_per_prompt=limit_mm_per_prompt or {})

        if limit_mm_per_prompt:
            raise ValueError("`limit_mm_per_prompt` is only supported for "
                             "multimodal models.")

        return None

    def _init_attention_free(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.is_attention_free_model(architectures)

    def _init_has_inner_state(self) -> bool:
        architectures = getattr(self.hf_config, "architectures", [])
        return ModelRegistry.model_has_inner_state(architectures)

    def _verify_tokenizer_mode(self) -> None:
        tokenizer_mode = self.tokenizer_mode.lower()
        if tokenizer_mode not in ["auto", "slow", "mistral"]:
            raise ValueError(
                f"Unknown tokenizer mode: {self.tokenizer_mode}. Must be "
                "either 'auto', 'slow' or 'mistral'.")
        self.tokenizer_mode = tokenizer_mode

    def _verify_embedding_mode(self) -> None:
        architectures = getattr(self.hf_config, "architectures", [])
        self.embedding_mode = ModelRegistry.is_embedding_model(architectures)

    def _verify_whisper_mode(self) -> None:
        architectures = getattr(self.hf_config, "architectures", [])
        self.whisper_mode = ModelRegistry.is_whisper_model(architectures)

    def _parse_quant_hf_config(self):
        quant_cfg = getattr(self.hf_config, "quantization_config", None)
        if quant_cfg is None:
            # compressed-tensors uses a "compression_config" key
            quant_cfg = getattr(self.hf_config, "compression_config", None)
        return quant_cfg

    def _verify_quantization(self) -> None:
        supported_quantization = [*QUANTIZATION_METHODS]
        rocm_supported_quantization = [
            "awq", "gptq", "fp8", "compressed_tensors", "compressed-tensors",
            "fbgemm_fp8"
        ]
        optimized_quantization_methods = [
            "fp8", "marlin", "modelopt", "gptq_marlin_24", "gptq_marlin",
            "awq_marlin", "fbgemm_fp8", "compressed_tensors",
            "compressed-tensors", "experts_int8"
        ]
        tpu_supported_quantization = ["tpu_int8"]
        neuron_supported_quantization = ["neuron_quant"]
        if self.quantization is not None:
            self.quantization = self.quantization.lower()

        # Parse quantization method from the HF model config, if available.
        quant_cfg = self._parse_quant_hf_config()

        if quant_cfg is not None:
            quant_method = quant_cfg.get("quant_method", "").lower()

            # Detect which checkpoint is it
            for _, method in QUANTIZATION_METHODS.items():
                quantization_override = method.override_quantization_method(
                    quant_cfg, self.quantization)
                if quantization_override:
                    quant_method = quantization_override
                    self.quantization = quantization_override
                    break

            # Verify quantization configurations.
            if self.quantization is None:
                self.quantization = quant_method
            elif self.quantization != quant_method:
                raise ValueError(
                    "Quantization method specified in the model config "
                    f"({quant_method}) does not match the quantization "
                    f"method specified in the `quantization` argument "
                    f"({self.quantization}).")

        if self.quantization is not None:
            if self.quantization not in supported_quantization:
                raise ValueError(
                    f"Unknown quantization method: {self.quantization}. Must "
                    f"be one of {supported_quantization}.")
            if is_hip(
            ) and self.quantization not in rocm_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in ROCm.")
            if current_platform.is_tpu(
            ) and self.quantization not in tpu_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in TPU Backend.")
            if self.quantization not in optimized_quantization_methods:
                logger.warning(
                    "%s quantization is not fully "
                    "optimized yet. The speed can be slower than "
                    "non-quantized models.", self.quantization)
            if (self.quantization == "awq" and is_hip()
                    and not envs.VLLM_USE_TRITON_AWQ):
                logger.warning(
                    "Using AWQ quantization with ROCm, but VLLM_USE_TRITON_AWQ"
                    " is not set, enabling VLLM_USE_TRITON_AWQ.")
                envs.VLLM_USE_TRITON_AWQ = True
            if is_neuron(
            ) and self.quantization not in neuron_supported_quantization:
                raise ValueError(
                    f"{self.quantization} quantization is currently not "
                    f"supported in Neuron Backend.")

    def _verify_cuda_graph(self) -> None:
        if self.max_seq_len_to_capture is None:
            self.max_seq_len_to_capture = self.max_model_len
        self.max_seq_len_to_capture = min(self.max_seq_len_to_capture,
                                          self.max_model_len)

    def _verify_bnb_config(self) -> None:
        """
        The current version of bitsandbytes (0.44.0) with 8-bit models does not
        yet support CUDA graph.
        """
        is_bitsandbytes = self.quantization == "bitsandbytes"
        has_quantization_config = (getattr(self.hf_config,
                                           "quantization_config", None)
                                   is not None)
        is_8bit = (self.hf_config.quantization_config.get(
            "load_in_8bit", False) if has_quantization_config else False)
        if all([
                is_bitsandbytes,
                has_quantization_config,
                is_8bit,
                not self.enforce_eager,
        ]):
            logger.warning(
                "CUDA graph is not supported on BitAndBytes 8bit yet, "
                "fallback to the eager mode.")
            self.enforce_eager = True

    def verify_async_output_proc(self, parallel_config, speculative_config,
                                 device_config) -> None:
        if not self.use_async_output_proc:
            # Nothing to check
            return

        if parallel_config.pipeline_parallel_size > 1:
            logger.warning("Async output processing can not be enabled "
                           "with pipeline parallel")
            self.use_async_output_proc = False
            return

        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if device_config.device_type not in ("cuda", "tpu", "hpu"):
            logger.warning(
                "Async output processing is only supported for CUDA, TPU "
                "and HPU. "
                "Disabling it for other platforms.")
            self.use_async_output_proc = False
            return

        if envs.VLLM_USE_RAY_SPMD_WORKER:
            logger.warning(
                "Async output processing can not be enabled with ray spmd")
            self.use_async_output_proc = False
            return

        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if device_config.device_type == "cuda" and self.enforce_eager:
            logger.warning(
                "To see benefits of async output processing, enable CUDA "
                "graph. Since, enforce-eager is enabled, async output "
                "processor cannot be used")
            self.use_async_output_proc = not self.enforce_eager
            return

        # Async postprocessor is not necessary with embedding mode
        # since there is no token generation
        if self.embedding_mode:
            self.use_async_output_proc = False

        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if speculative_config:
            logger.warning("Async output processing is not supported with"
                           " speculative decoding currently.")
            self.use_async_output_proc = False

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = getattr(self.hf_text_config,
                                            "num_attention_heads", 0)
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size}).")

        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if pipeline_parallel_size > 1:
            architectures = getattr(self.hf_config, "architectures", [])
            if not ModelRegistry.is_pp_supported_model(architectures):
                raise NotImplementedError(
                    "Pipeline parallelism is not supported for this model. "
                    "Supported models implement the `SupportsPP` interface.")

            if self.use_async_output_proc:
                logger.warning("Async output processor is not supported with "
                               "pipeline parallelism currently. Disabling it.")
                self.use_async_output_proc = False

    def get_hf_config_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled."""

        # Some models, like Qwen2 and Qwen1.5, use `use_sliding_window` in
        # addition to sliding window size. We check if that field is present
        # and if it's False, return None.
        if (hasattr(self.hf_text_config, "use_sliding_window")
                and not self.hf_text_config.use_sliding_window):
            return None
        return getattr(self.hf_text_config, "sliding_window", None)

    def get_sliding_window(self) -> Optional[int]:
        """Get the sliding window size, or None if disabled.
        """
        # If user disables sliding window, return None.
        if self.disable_sliding_window:
            return None
        # Otherwise get the value from the hf config.
        return self.get_hf_config_sliding_window()

    def get_vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    def get_hidden_size(self) -> int:
        return self.hf_text_config.hidden_size

    def get_head_size(self) -> int:
        # TODO remove hard code
        if hasattr(self.hf_text_config, "model_type"
                   ) and self.hf_text_config.model_type == 'deepseek_v2':
            # FlashAttention supports only head_size 32, 64, 128, 256,
            # we need to pad head_size 192 to 256
            return 256

        if self.is_attention_free:
            return 0

        if hasattr(self.hf_text_config, "head_dim"):
            return self.hf_text_config.head_dim
        # FIXME(woosuk): This may not be true for all models.
        return (self.hf_text_config.hidden_size //
                self.hf_text_config.num_attention_heads)

    def get_total_num_kv_heads(self) -> int:
        """Returns the total number of KV heads."""
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False))
        if not new_decoder_arch_falcon and getattr(self.hf_text_config,
                                                   "multi_query", False):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(self.hf_config.attn_config, "kv_n_heads",
                           self.hf_config.num_attention_heads)

        if self.is_attention_free:
            return 0

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1,
                   total_num_kv_heads // parallel_config.tensor_parallel_size)

    def get_num_attention_heads(self,
                                parallel_config: "ParallelConfig") -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        from vllm.distributed.utils import get_pp_indices
        total_num_hidden_layers = getattr(self.hf_text_config,
                                          "num_hidden_layers", 0)
        pp_rank = parallel_config.rank // parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        start, end = get_pp_indices(total_num_hidden_layers, pp_rank, pp_size)
        return end - start

    def get_num_attention_layers(self,
                                 parallel_config: "ParallelConfig") -> int:
        if self.is_attention_free:
            return 0

        num_layers = self.get_num_layers(parallel_config)

        # Transformers supports layers_block_type @property
        layers = getattr(self.hf_config, "layers_block_type",
                         ["attention"] * num_layers)
        return len([t for t in layers if t == "attention"])

    def get_multimodal_config(self) -> "MultiModalConfig":
        """
        Get the multimodal configuration of the model.

        Raises:
            ValueError: If the model is not multimodal.
        """
        if self.multimodal_config is None:
            raise ValueError("The model is not multimodal.")

        return self.multimodal_config

    @property
    def is_encoder_decoder_model(self) -> bool:
        """Extract the HF encoder/decoder model flag."""
        return getattr(self.hf_config, "is_encoder_decoder", False) or (
            (hasattr(self.hf_config, "text_config") and getattr(
                self.hf_config.text_config, "is_encoder_decoder", False)))

    @property
    def is_embedding_model(self) -> bool:
        """Extract the embedding model flag."""
        return self.embedding_mode

    @property
    def is_whisper_model(self) -> bool:
        """Extract the embedding model flag."""
        return self.whisper_mode

    @property
    def is_multimodal_model(self) -> bool:
        return self.multimodal_config is not None


class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
        num_gpu_blocks_override: Number of GPU blocks to use. This overrides the
            profiled num_gpu_blocks if specified. Does nothing if None.
    """

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: float,
        cache_dtype: str,
        is_attention_free: bool = False,
        num_gpu_blocks_override: Optional[int] = None,
        sliding_window: Optional[int] = None,
        enable_prefix_caching: bool = False,
        cpu_offload_gb: float = 0,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * GiB_bytes
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.cache_dtype = cache_dtype
        self.is_attention_free = is_attention_free
        self.sliding_window = sliding_window
        self.enable_prefix_caching = enable_prefix_caching
        self.cpu_offload_gb = cpu_offload_gb
        self._verify_args()
        self._verify_cache_dtype()
        self._verify_prefix_caching()

        # Will be set after profiling.
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0. Got "
                f"{self.gpu_memory_utilization}.")

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc"):
            logger.info(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor. "
                "Intel Gaudi (HPU) supports fp8 (using fp8_inc).")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_prefix_caching(self) -> None:
        if not self.enable_prefix_caching:
            return

        if self.sliding_window is not None:
            raise NotImplementedError(
                "Prefix caching is not supported with sliding window. "
                "Run with --disable-sliding-window to use prefix caching.")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): Here, it is assumed that the GPUs in a tensor parallel
        # group are in the same node. However, the GPUs may span multiple nodes.
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (f"{cpu_memory_usage / GiB_bytes:.2f} GiB out of the "
               f"{total_cpu_memory / GiB_bytes:.2f} GiB total CPU memory "
               "is allocated for the swap space.")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("Too large swap space. " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warning("Possibly too large swap space. %s", msg)


class DeviceConfig:
    device: Optional[torch.device]

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            # Automated device type detection
            if current_platform.is_cuda_alike():
                self.device_type = "cuda"
            elif is_neuron():
                self.device_type = "neuron"
            elif current_platform.is_hpu():
                self.device_type = "hpu"
            elif is_openvino():
                self.device_type = "openvino"
            elif current_platform.is_tpu():
                self.device_type = "tpu"
            elif current_platform.is_cpu():
                self.device_type = "cpu"
            elif is_xpu():
                self.device_type = "xpu"
            else:
                raise RuntimeError("Failed to infer device type")
        else:
            # Device type is assigned explicitly
            self.device_type = device

        # Some device types require processing inputs on CPU
        if self.device_type in ["neuron", "openvino"]:
            self.device = torch.device("cpu")
        elif self.device_type in ["tpu"]:
            self.device = None
        else:
            # Set device with device type
            self.device = torch.device(self.device_type)


@dataclass
class TokenizerPoolConfig:
    """Configuration for the tokenizer pool.

    Args:
        pool_size: Number of tokenizer workers in the pool.
        pool_type: Type of the pool.
        extra_config: Additional config for the pool.
            The way the config will be used depends on the
            pool type.
    """
    pool_size: int
    pool_type: Union[str, Type["BaseTokenizerGroup"]]
    extra_config: dict

    def __post_init__(self):
        if self.pool_type not in ("ray", ) and not isinstance(
                self.pool_type, type):
            raise ValueError(f"Unknown pool type: {self.pool_type}")
        if not isinstance(self.extra_config, dict):
            raise ValueError("extra_config must be a dictionary.")

    @classmethod
    def create_config(
        cls, tokenizer_pool_size: int, tokenizer_pool_type: str,
        tokenizer_pool_extra_config: Optional[Union[str, dict]]
    ) -> Optional["TokenizerPoolConfig"]:
        """Create a TokenizerPoolConfig from the given parameters.

        If tokenizer_pool_size is 0, return None.

        Args:
            tokenizer_pool_size: Number of tokenizer workers in the pool.
            tokenizer_pool_type: Type of the pool.
            tokenizer_pool_extra_config: Additional config for the pool.
                The way the config will be used depends on the
                pool type. This can be a JSON string (will be parsed).
        """
        if tokenizer_pool_size:
            if isinstance(tokenizer_pool_extra_config, str):
                tokenizer_pool_extra_config_parsed = json.loads(
                    tokenizer_pool_extra_config)
            else:
                tokenizer_pool_extra_config_parsed = (
                    tokenizer_pool_extra_config or {})
            tokenizer_pool_config = cls(tokenizer_pool_size,
                                        tokenizer_pool_type,
                                        tokenizer_pool_extra_config_parsed)
        else:
            tokenizer_pool_config = None
        return tokenizer_pool_config


class LoadFormat(str, enum.Enum):
    AUTO = "auto"
    PT = "pt"
    SAFETENSORS = "safetensors"
    NPCACHE = "npcache"
    DUMMY = "dummy"
    TENSORIZER = "tensorizer"
    SHARDED_STATE = "sharded_state"
    GGUF = "gguf"
    BITSANDBYTES = "bitsandbytes"
    MISTRAL = "mistral"


@dataclass
class LoadConfig:
    """
        download_dir: Directory to download and load the weights, default to the
            default cache directory of huggingface.
        load_format: The format of the model weights to load:
            "auto" will try to load the weights in the safetensors format and
                fall back to the pytorch bin format if safetensors format is
                not available.
            "pt" will load the weights in the pytorch bin format.
            "safetensors" will load the weights in the safetensors format.
            "npcache" will load the weights in pytorch format and store
                a numpy cache to speed up the loading.
            "dummy" will initialize the weights with random values, which is
                mainly for profiling.
            "tensorizer" will use CoreWeave's tensorizer library for
                fast weight loading.
            "bitsandbytes" will load nf4 type weights.
        ignore_patterns: The list of patterns to ignore when loading the model.
            Default to "original/**/*" to avoid repeated loading of llama's
            checkpoints.
        device: Device to which model weights will be loaded, default to
            device_config.device
    """

    load_format: Union[str, LoadFormat, "BaseModelLoader"] = LoadFormat.AUTO
    download_dir: Optional[str] = None
    device: Optional[str] = None
    model_loader_extra_config: Optional[Union[str, dict]] = field(
        default_factory=dict)
    ignore_patterns: Optional[Union[List[str], str]] = None

    def __post_init__(self):
        model_loader_extra_config = self.model_loader_extra_config or {}
        if isinstance(model_loader_extra_config, str):
            self.model_loader_extra_config = json.loads(
                model_loader_extra_config)
        self._verify_load_format()

        if self.ignore_patterns is not None and len(self.ignore_patterns) > 0:
            logger.info(
                "Ignoring the following patterns when downloading weights: %s",
                self.ignore_patterns)
        else:
            self.ignore_patterns = ["original/**/*"]

    def _verify_load_format(self) -> None:
        if not isinstance(self.load_format, str):
            return

        load_format = self.load_format.lower()
        self.load_format = LoadFormat(load_format)

        rocm_not_supported_load_format: List[str] = []
        if is_hip() and load_format in rocm_not_supported_load_format:
            rocm_supported_load_format = [
                f for f in LoadFormat.__members__
                if (f not in rocm_not_supported_load_format)
            ]
            raise ValueError(
                f"load format '{load_format}' is not supported in ROCm. "
                f"Supported load formats are "
                f"{rocm_supported_load_format}")


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        pipeline_parallel_size: Number of pipeline parallel groups.
        tensor_parallel_size: Number of tensor parallel groups.
        worker_use_ray: Deprecated, use distributed_executor_backend instead.
        max_parallel_loading_workers: Maximum number of multiple batches
            when load model sequentially. To avoid RAM OOM when using tensor
            parallel and large models.
        disable_custom_all_reduce: Disable the custom all-reduce kernel and
            fall back to NCCL.
        tokenizer_pool_config: Config for the tokenizer pool.
            If None, will use synchronous tokenization.
        ray_workers_use_nsight: Whether to profile Ray workers with nsight, see
            https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.
        placement_group: ray distributed model workers placement group.
        distributed_executor_backend: Backend to use for distributed model
            workers, either "ray" or "mp" (multiprocessing). If either
            pipeline_parallel_size or tensor_parallel_size is greater than 1,
            will default to "ray" if Ray is installed or "mp" otherwise.
    """

    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: Optional[bool] = None,
        max_parallel_loading_workers: Optional[int] = None,
        disable_custom_all_reduce: bool = False,
        tokenizer_pool_config: Optional[TokenizerPoolConfig] = None,
        ray_workers_use_nsight: bool = False,
        placement_group: Optional["PlacementGroup"] = None,
        distributed_executor_backend: Optional[Union[
            str, Type["ExecutorBase"]]] = None,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.distributed_executor_backend = distributed_executor_backend
        self.max_parallel_loading_workers = max_parallel_loading_workers
        self.disable_custom_all_reduce = disable_custom_all_reduce
        self.tokenizer_pool_config = tokenizer_pool_config
        self.ray_workers_use_nsight = ray_workers_use_nsight
        self.placement_group = placement_group
        self.world_size = pipeline_parallel_size * self.tensor_parallel_size

        if worker_use_ray:
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            elif not self.use_ray:
                raise ValueError(f"worker-use-ray can't be used with "
                                 f"distributed executor backend "
                                 f"'{self.distributed_executor_backend}'.")

        if current_platform.is_tpu() and self.world_size > 1:
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    "TPU backend only supports Ray for distributed inference.")

        if current_platform.is_hpu() and self.world_size > 1:
            if self.distributed_executor_backend is None:
                self.distributed_executor_backend = "ray"
            if self.distributed_executor_backend != "ray":
                raise ValueError(
                    "HPU backend only supports Ray for distributed inference.")

        if self.distributed_executor_backend is None and self.world_size > 1:
            # We use multiprocessing by default if world_size fits on the
            # current node and we aren't in a ray placement group.

            from vllm.executor import ray_utils
            backend = "mp"
            ray_found = ray_utils.ray_is_available()
            if (current_platform.is_cuda()
                    and cuda_device_count_stateless() < self.world_size):
                if not ray_found:
                    raise ValueError("Unable to load Ray which is "
                                     "required for multi-node inference, "
                                     "please install Ray with `pip install "
                                     "ray`.") from ray_utils.ray_import_err
                backend = "ray"
            elif ray_found:
                if self.placement_group:
                    backend = "ray"
                else:
                    from ray import is_initialized as ray_is_initialized
                    if ray_is_initialized():
                        from ray.util import get_current_placement_group
                        if get_current_placement_group():
                            backend = "ray"
            self.distributed_executor_backend = backend
            logger.info("Defaulting to use %s for distributed inference",
                        backend)

        self._verify_args()
        self.rank: int = 0

    @property
    def use_ray(self) -> bool:
        return self.distributed_executor_backend == "ray" or (
            isinstance(self.distributed_executor_backend, type)
            and self.distributed_executor_backend.uses_ray)

    def _verify_args(self) -> None:
        # Lazy import to avoid circular import
        from vllm.executor.executor_base import ExecutorBase

        if self.distributed_executor_backend not in (
                "ray", "mp", None) and not (isinstance(
                    self.distributed_executor_backend, type) and issubclass(
                        self.distributed_executor_backend, ExecutorBase)):
            raise ValueError(
                "Unrecognized distributed executor backend "
                f"{self.distributed_executor_backend}. Supported "
                "values are 'ray', 'mp' or custom ExecutorBase subclass.")
        if self.use_ray:
            from vllm.executor import ray_utils
            ray_utils.assert_ray_available()
        if is_hip():
            self.disable_custom_all_reduce = True
            logger.info(
                "Disabled the custom all-reduce kernel because it is not "
                "supported on AMD GPUs.")
        if self.ray_workers_use_nsight and not self.use_ray:
            raise ValueError("Unable to use nsight profiling unless workers "
                             "run with Ray.")


class SchedulerConfig:
    """Scheduler configuration.

    Args:
        max_num_batched_tokens: Maximum number of tokens to be processed in
            a single iteration.
        max_num_seqs: Maximum number of sequences to be processed in a single
            iteration.
        max_num_prefill_seqs: Maximum number of prefill sequences to be
             processed in a single iteration. Used only with padding-aware
             scheduling.
        max_model_len: Maximum length of a sequence (including prompt
            and generated text).
        use_v2_block_manager: Whether to use the BlockSpaceManagerV2 or not.
        num_lookahead_slots: The number of slots to allocate per sequence per
            step, beyond the known token ids. This is used in speculative
            decoding to store KV activations of tokens which may or may not be
            accepted.
        delay_factor: Apply a delay (of delay factor multiplied by previous
            prompt latency) before scheduling next prompt.
        enable_chunked_prefill: If True, prefill requests can be chunked based
            on the remaining max_num_batched_tokens.
        embedding_mode: Whether the running model is for embedding.
        preemption_mode: Whether to perform preemption by swapping or
            recomputation. If not specified, we determine the mode as follows:
            We use recomputation by default since it incurs lower overhead than
            swapping. However, when the sequence group has multiple sequences
            (e.g., beam search), recomputation is not currently supported. In
            such a case, we use swapping instead.
        send_delta_data: Private API. If used, scheduler sends delta data to
            workers instead of an entire data. It should be enabled only
            when SPMD worker architecture is enabled. I.e.,
            VLLM_USE_RAY_SPMD_WORKER=1
        policy: The scheduling policy to use. "fcfs" (default) or "priority".
        use_padding_aware_scheduling: If True, scheduler will consider padded
            tokens in prefill.
    """

    def __init__(self,
                 max_num_batched_tokens: Optional[int],
                 max_num_seqs: int,
                 max_num_prefill_seqs: Optional[int],
                 max_model_len: int,
                 use_v2_block_manager: bool = True,
                 num_lookahead_slots: int = 0,
                 delay_factor: float = 0.0,
                 enable_chunked_prefill: bool = False,
                 embedding_mode: bool = False,
                 whisper_mode: Optional[bool] = False,
                 preemption_mode: Optional[str] = None,
                 num_scheduler_steps: int = 1,
                 multi_step_stream_outputs: bool = False,
                 send_delta_data: bool = False,
                 policy: str = "fcfs",
                 use_padding_aware_scheduling=False) -> None:
        if max_num_batched_tokens is None:
            if enable_chunked_prefill:
                if num_scheduler_steps > 1:
                    # Multi-step Chunked-Prefill doesn't allow prompt-chunking
                    # for now. Have max_num_batched_tokens set to max_model_len
                    # so we don't reject sequences on account of a short
                    # max_num_batched_tokens.
                    max_num_batched_tokens = max(max_model_len, 2048)
            else:
                # If max_model_len is too short, use 2048 as the default value
                # for higher throughput.
                max_num_batched_tokens = max(max_model_len, 2048)

            if whisper_mode:
                    # It is the values that have the best balance between ITL
                    # and TTFT on A100. Note it is not optimized for throughput.
                max_num_batched_tokens = max(
                    max_model_len, _WHISPER_MAX_NUM_BATCHED_TOKENS)
            if embedding_mode:
                # For embedding, choose specific value for higher throughput
                max_num_batched_tokens = max(
                    max_num_batched_tokens,
                    _EMBEDDING_MODEL_MAX_NUM_BATCHED_TOKENS,
                )
            if is_multimodal_model:
                # The value needs to be at least the number of multimodal tokens
                max_num_batched_tokens = max(
                    max_num_batched_tokens,
                    _MULTIMODAL_MODEL_MAX_NUM_BATCHED_TOKENS,
                )

        self.max_num_batched_tokens = max_num_batched_tokens

        if enable_chunked_prefill:
            logger.info(
                "Chunked prefill is enabled with max_num_batched_tokens=%d.",
                self.max_num_batched_tokens)

        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.max_model_len = max_model_len
        self.use_v2_block_manager = use_v2_block_manager
        self.num_lookahead_slots = num_lookahead_slots
        self.delay_factor = delay_factor
        self.chunked_prefill_enabled = enable_chunked_prefill
        self.embedding_mode = embedding_mode
        self.preemption_mode = preemption_mode
        self.num_scheduler_steps = num_scheduler_steps
        self.multi_step_stream_outputs = multi_step_stream_outputs
        self.send_delta_data = send_delta_data
        self.policy = policy
        self.use_padding_aware_scheduling = use_padding_aware_scheduling
        self._verify_args()

    def _verify_args(self) -> None:
        if (self.max_num_batched_tokens < self.max_model_len
                and not self.chunked_prefill_enabled):
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) is "
                f"smaller than max_model_len ({self.max_model_len}). "
                "This effectively limits the maximum sequence length to "
                "max_num_batched_tokens and makes vLLM reject longer "
                "sequences. Please increase max_num_batched_tokens or "
                "decrease max_model_len.")

        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.num_lookahead_slots < 0:
            raise ValueError(
                "num_lookahead_slots "
                f"({self.num_lookahead_slots}) must be greater than or "
                "equal to 0.")

        if self.num_scheduler_steps < 1:
            raise ValueError(
                "num_scheduler_steps "
                f"({self.num_scheduler_steps}) must be greater than or "
                "equal to 1.")
        if self.max_num_prefill_seqs is not None \
            and not self.use_padding_aware_scheduling:
            raise ValueError("max_num_prefill_seqs can be only "
                             "used with padding-aware-scheduling. ")
        if self.use_padding_aware_scheduling and self.chunked_prefill_enabled:
            raise ValueError("Padding-aware scheduling currently "
                             "does not work with chunked prefill ")

        if (not self.use_v2_block_manager \
            and not envs.VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1):
            raise ValueError(
                "The use of BlockSpaceManagerV1 is deprecated and will "
                "be removed in a future release. Please switch to "
                "BlockSpaceManagerV2 by setting --use-v2-block-manager to "
                "True. If you wish to suppress this error temporarily, "
                "you can set the environment variable "
                "`VLLM_ALLOW_DEPRECATED_BLOCK_MANAGER_V1=1. If your use "
                "case is not supported in BlockSpaceManagerV2, please "
                "file an issue with detailed information.")

    @property
    def is_multi_step(self) -> bool:
        return self.num_scheduler_steps > 1


@dataclass
class LoRAConfig:
    max_lora_rank: int
    max_loras: int
    fully_sharded_loras: bool = False
    max_cpu_loras: Optional[int] = None
    lora_dtype: Optional[torch.dtype] = None
    lora_extra_vocab_size: int = 256
    # This is a constant.
    lora_vocab_padding_size: ClassVar[int] = 256
    long_lora_scaling_factors: Optional[Tuple[float]] = None

    def __post_init__(self):
        # Setting the maximum rank to 256 should be able to satisfy the vast
        # majority of applications.
        possible_max_ranks = (8, 16, 32, 64, 128, 256)
        possible_lora_extra_vocab_size = (0, 256, 512)
        if self.max_lora_rank not in possible_max_ranks:
            raise ValueError(
                f"max_lora_rank ({self.max_lora_rank}) must be one of "
                f"{possible_max_ranks}.")
        if self.lora_extra_vocab_size not in possible_lora_extra_vocab_size:
            raise ValueError(
                f"lora_extra_vocab_size ({self.lora_extra_vocab_size}) "
                f"must be one of {possible_lora_extra_vocab_size}.")
        if self.max_loras < 1:
            raise ValueError(f"max_loras ({self.max_loras}) must be >= 1.")
        if self.max_cpu_loras is None:
            self.max_cpu_loras = self.max_loras
        elif self.max_cpu_loras < self.max_loras:
            raise ValueError(
                f"max_cpu_loras ({self.max_cpu_loras}) must be >= "
                f"max_loras ({self.max_loras})")

    def verify_with_model_config(self, model_config: ModelConfig):
        if self.lora_dtype in (None, "auto"):
            self.lora_dtype = model_config.dtype
        elif isinstance(self.lora_dtype, str):
            self.lora_dtype = getattr(torch, self.lora_dtype)
        if model_config.quantization and model_config.quantization not in [
                "awq", "gptq"
        ]:
            # TODO support marlin
            logger.warning("%s quantization is not tested with LoRA yet.",
                           model_config.quantization)

    def verify_with_scheduler_config(self, scheduler_config: SchedulerConfig):
        # Reminder: Please update docs/source/serving/compatibility_matrix.rst
        # If the feature combo become valid
        if scheduler_config.chunked_prefill_enabled:
            raise ValueError("LoRA is not supported with chunked prefill yet.")


@dataclass
class MultiModalConfig:
    """Controls the behavior of multimodal models."""

    limit_per_prompt: Mapping[str, int] = field(default_factory=dict)
    """
    The maximum number of multi-modal input instances allowed per prompt
    for each :class:`~vllm.multimodal.MultiModalPlugin`.
    """

    # TODO: Add configs to init vision tower or not.

