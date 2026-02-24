# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio
import itertools
import os
import sys
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.inputs import PromptType
from vllm.inputs.data import is_tokens_prompt
from vllm.logger import init_logger
from vllm.outputs import CompletionOutput, PoolingRequestOutput, RequestOutput
from vllm.plugins.io_processors import get_io_processor
from vllm.pooling_params import PoolingParams
from vllm.renderers import BaseRenderer
from vllm.sampling_params import RequestOutputKind, SamplingParams, SamplingType
from vllm.tasks import SupportedTask
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.engine.utils import get_prompt_text

logger = init_logger(__name__)

# Prefer vendored lmdeploy package/binary locations when present, so vLLM
# TurboMind integration does not accidentally import unrelated site-packages.
_VENDORED_VLLM_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_LMDEPLOY_PKG = _VENDORED_VLLM_ROOT / "lmdeploy"
_VENDORED_LMDEPLOY_LIB = _VENDORED_LMDEPLOY_PKG / "lib"

if _VENDORED_LMDEPLOY_PKG.exists() and (_VENDORED_LMDEPLOY_PKG / "__init__.py").exists():
    _vendored_root = str(_VENDORED_VLLM_ROOT)
    if _vendored_root not in sys.path:
        sys.path.insert(0, _vendored_root)

if _VENDORED_LMDEPLOY_LIB.exists():
    _vendored_lib = str(_VENDORED_LMDEPLOY_LIB)
    if _vendored_lib not in sys.path:
        sys.path.insert(0, _vendored_lib)

try:
    import _turbomind as _tm
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    _TM_IMPORT_ERROR = exc
    _tm = None  # type: ignore[assignment]


def _require_turbomind() -> None:
    if _tm is None:
        raise ModuleNotFoundError(
            "TurboMind backend requires the `_turbomind` extension module. "
            "Build/install TurboMind and ensure it is on PYTHONPATH."
        ) from _TM_IMPORT_ERROR


def _construct_stop_or_bad_words(words: list[int] | None = None):
    if words is None or len(words) == 0:
        return None
    offsets = list(range(1, len(words) + 1))
    return [words, offsets]


def _is_tm_debug_enabled() -> bool:
    val = os.getenv("VLLM_TURBOMIND_DEBUG", "")
    return val.lower() in ("1", "true", "yes", "on")


class StreamingSemaphore:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.fut: asyncio.Future | None = None
        self.val = 0

    async def acquire(self) -> None:
        if self.val:
            self.val = 0
            return
        self.fut = self.loop.create_future()
        await self.fut
        self.fut = None
        self.val = 0

    def release(self) -> None:
        if not self.val:
            self.val = 1
            if self.fut and not self.fut.done():
                self.fut.set_result(None)

    def release_threadsafe(self) -> None:
        self.loop.call_soon_threadsafe(self.release)


@dataclass
class TMOutput:
    token_ids: list[int]
    status: int
    logprobs: list[dict[int, float]] | None = None


def _get_logprobs_impl(
    logprob_vals: torch.Tensor,
    logprob_idxs: torch.Tensor,
    logprob_nums: torch.Tensor,
    output_ids: list[int],
    logprobs: int,
    offset: int,
) -> list[dict[int, float]]:
    out_logprobs: list[dict[int, float]] = []
    length = len(output_ids) + offset
    for pos, idx, val, n in zip(
        range(len(output_ids)),
        logprob_idxs[offset:length],
        logprob_vals[offset:length],
        logprob_nums[offset:length],
    ):
        topn = min(n.item(), logprobs)
        tok_res = {idx[i].item(): val[i].item() for i in range(topn)}
        token_id = output_ids[pos]
        if token_id not in tok_res:
            valid_n = n.item()
            tok_res[token_id] = val[:valid_n][idx[:valid_n] == token_id].item()
        ids = list(tok_res.keys())
        for k in ids:
            if tok_res[k] == float("-inf"):
                tok_res.pop(k)
        out_logprobs.append(tok_res)
    return out_logprobs


def _get_logprobs(outputs: dict, output_logprobs: int):
    logprob_vals = outputs["logprob_vals"]
    logprob_idxs = outputs["logprob_indexes"]
    logprob_nums = outputs["logprob_nums"]
    offset = 0

    def _func(out: TMOutput) -> None:
        nonlocal offset
        out.logprobs = _get_logprobs_impl(
            logprob_vals, logprob_idxs, logprob_nums, out.token_ids, output_logprobs, offset
        )
        offset += len(out.token_ids)

    return _func


@dataclass
class TurbomindEngineConfig:
    tp: int = 1
    dp: int = 1
    cp: int = 1
    session_len: int | None = None
    max_batch_size: int | None = None
    cache_max_entry_count: float = 0.8
    cache_chunk_size: int = -1
    cache_block_seq_len: int = 64
    enable_prefix_caching: bool = False
    quant_policy: int = 0
    max_prefill_token_num: int = 8192
    num_tokens_per_iter: int = 0
    max_prefill_iters: int = 1
    async_: int = 1
    devices: list[int] | None = None
    nnodes: int = 1
    node_rank: int = 0
    communicator: str = "nccl"
    enable_metrics: bool = False
    outer_dp_size: int = 1
    attn_dp_size: int = 1
    attn_tp_size: int = 1
    attn_cp_size: int = 1
    mlp_tp_size: int = 1

    def finalize(self) -> None:
        if self.devices is None:
            self.devices = list(range(self.tp))
        self.attn_dp_size = 1
        self.attn_tp_size = self.tp
        self.attn_cp_size = self.cp
        self.outer_dp_size = 1
        self.mlp_tp_size = self.tp

    def to_engine_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "max_batch_size": self.max_batch_size or 0,
            "max_prefill_token_num": self.max_prefill_token_num,
            "max_context_token_num": 0,
            "cache_max_entry_count": self.cache_max_entry_count,
            "cache_chunk_size": self.cache_chunk_size,
            "enable_prefix_caching": self.enable_prefix_caching,
            "quant_policy": self.quant_policy,
            "num_tokens_per_iter": self.num_tokens_per_iter,
            "max_prefill_iters": self.max_prefill_iters,
            "async_": self.async_,
            "outer_dp_size": self.outer_dp_size,
            "attn_dp_size": self.attn_dp_size,
            "attn_tp_size": self.attn_tp_size,
            "attn_cp_size": self.attn_cp_size,
            "mlp_tp_size": self.mlp_tp_size,
            "devices": self.devices,
            "nnodes": self.nnodes,
            "node_rank": self.node_rank,
            "communicator": self.communicator,
            "enable_metrics": self.enable_metrics,
        }


class TurboMindRuntime:
    def __init__(
        self,
        model_dir: str,
        engine_config: TurbomindEngineConfig,
        vllm_config: VllmConfig,
    ) -> None:
        _require_turbomind()
        self.vllm_config = vllm_config
        self.model_dir = self._resolve_model_dir(model_dir)
        self.engine_config = engine_config
        self._use_in_memory_weights = False
        self._tm_model = None
        self.config_dict = self._load_config()
        self.model_comm = self._create_model_comm()
        self._create_weights()
        if self._use_in_memory_weights:
            self._load_weights_from_hf()
        else:
            self._load_weights_from_dir()
        self._process_weights()
        self._create_engine()

    @staticmethod
    def _is_hf_repo_id(model_dir: str) -> bool:
        # Hub repo ids follow `namespace/repo_name`.
        if model_dir.startswith(("/", ".", "~")):
            return False
        parts = model_dir.split("/")
        return len(parts) == 2 and all(parts)

    def _resolve_model_dir(self, model_dir: str) -> str:
        if os.path.exists(model_dir):
            return model_dir

        if not self._is_hf_repo_id(model_dir):
            return model_dir

        try:
            from lmdeploy.utils import get_model
        except Exception:
            logger.exception(
                "TurboMind backend: failed to import lmdeploy hub downloader. "
                "Proceeding with unresolved model path `%s`.",
                model_dir,
            )
            return model_dir

        download_dir = self.vllm_config.load_config.download_dir
        revision = self.vllm_config.model_config.revision
        token = self.vllm_config.model_config.hf_token
        try:
            resolved = get_model(
                model_dir,
                download_dir=download_dir,
                revision=revision,
                token=token,
            )
        except Exception as exc:
            raise RuntimeError(
                "TurboMind backend: failed to download model "
                f"`{model_dir}` from hub. Set `--hf-token` for private repos "
                "or pass a local model path."
            ) from exc

        if not resolved or not os.path.exists(resolved):
            raise RuntimeError(
                "TurboMind backend: hub download returned invalid path "
                f"for `{model_dir}`: `{resolved}`"
            )

        logger.info(
            "TurboMind backend: resolved hub model `%s` to local path `%s`.",
            model_dir,
            resolved,
        )
        return resolved

    def _load_config(self) -> dict[str, Any]:
        config_path = os.path.join(self.model_dir, "config.yaml")
        if not os.path.exists(config_path):
            logger.info(
                "TurboMind backend: `%s` not found. Using HF->TurboMind "
                "in-memory conversion and weight export.",
                config_path,
            )
            config_dict = self._build_config_from_hf()
            self._use_in_memory_weights = True
        else:
            logger.info(
                "TurboMind backend: found `%s`. Loading TurboMind checkpoint "
                "weights directly from model directory.",
                config_path,
            )
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
            if "model_config" not in config_dict:
                raise ValueError(f"Invalid TurboMind config: {config_path}")

        model_cfg = config_dict.setdefault("model_config", {})
        attn_cfg = config_dict.setdefault("attention_config", {})

        if self.engine_config.session_len:
            model_cfg["session_len"] = self.engine_config.session_len
        if self.engine_config.cache_block_seq_len:
            attn_cfg["cache_block_seq_len"] = self.engine_config.cache_block_seq_len

        # Keep engine_config serialization aligned with lmdeploy TurboMind
        # runtime (asdict(TurbomindEngineConfig) after update_parallel_config).
        try:
            lm_engine_cfg = self._build_lmdeploy_engine_config()
            config_dict["engine_config"] = asdict(lm_engine_cfg)
        except Exception:
            logger.exception(
                "TurboMind backend: failed to build lmdeploy-style engine_config; "
                "falling back to internal engine dict."
            )
            config_dict["engine_config"] = self.engine_config.to_engine_dict()
        if _is_tm_debug_enabled():
            model_cfg = config_dict.get("model_config", {})
            logger.info(
                "TurboMind debug config: model_dir=%s, model_arch=%s, vocab_size=%s, "
                "weight_type=%s, data_type=%s, session_len=%s, tp=%s, cp=%s, devices=%s",
                self.model_dir,
                model_cfg.get("model_arch"),
                model_cfg.get("vocab_size"),
                model_cfg.get("weight_type"),
                model_cfg.get("data_type"),
                model_cfg.get("session_len"),
                self.engine_config.tp,
                self.engine_config.cp,
                self.engine_config.devices,
            )
        return config_dict

    def _sanitize_model_id(self, model_path: str) -> str:
        model_id = os.path.basename(model_path.rstrip("/")) or "model"
        safe = model_id.replace("/", "_").replace(":", "_")
        return safe[:80]

    def _resolve_lmdeploy_dtype(self) -> str:
        dtype = self.vllm_config.model_config.dtype
        resolved = "auto"
        if isinstance(dtype, torch.dtype):
            if dtype == torch.bfloat16:
                resolved = "bfloat16"
            elif dtype == torch.float16:
                resolved = "float16"
        elif isinstance(dtype, str):
            lowered = dtype.lower()
            if lowered in ("auto", "float16", "bfloat16"):
                resolved = lowered

        # Some BF16-native models (e.g., Qwen3) can degrade severely when
        # forced to FP16 in TurboMind. Prefer BF16 unless user explicitly
        # forces FP16 through env var.
        if resolved == "float16":
            hf_cfg = getattr(self.vllm_config.model_config, "hf_config", None)
            hf_torch_dtype = getattr(hf_cfg, "torch_dtype", None)
            hf_is_bf16 = (
                hf_torch_dtype == torch.bfloat16
                or str(hf_torch_dtype).lower() in ("bfloat16", "torch.bfloat16")
            )
            force_fp16 = os.getenv("VLLM_TURBOMIND_FORCE_FP16", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if hf_is_bf16 and bf16_supported and not force_fp16:
                logger.warning(
                    "TurboMind backend: overriding requested float16 to bfloat16 "
                    "for a BF16-native model to avoid generation corruption. "
                    "Set VLLM_TURBOMIND_FORCE_FP16=1 to keep float16."
                )
                return "bfloat16"

        return resolved

    def _build_lmdeploy_engine_config(self):
        from lmdeploy.messages import TurbomindEngineConfig as LmdeployEngineConfig
        from lmdeploy.turbomind.turbomind import update_parallel_config

        if _is_tm_debug_enabled():
            try:
                import lmdeploy
                import lmdeploy.archs as lmdeploy_archs
                import lmdeploy.messages as lmdeploy_messages

                logger.info(
                    "TurboMind debug import paths: lmdeploy=%s lmdeploy.archs=%s "
                    "lmdeploy.messages=%s _turbomind=%s",
                    getattr(lmdeploy, "__file__", None),
                    getattr(lmdeploy_archs, "__file__", None),
                    getattr(lmdeploy_messages, "__file__", None),
                    getattr(_tm, "__file__", None),
                )
            except Exception:
                logger.exception("TurboMind debug: failed to inspect lmdeploy import paths.")

        lm_cfg = LmdeployEngineConfig(
            dtype=self._resolve_lmdeploy_dtype(),
            model_format=None,
            tp=self.engine_config.tp,
            dp=self.engine_config.dp,
            cp=self.engine_config.cp,
            session_len=self.engine_config.session_len,
            max_batch_size=self.engine_config.max_batch_size,
            cache_max_entry_count=self.engine_config.cache_max_entry_count,
            cache_chunk_size=self.engine_config.cache_chunk_size,
            cache_block_seq_len=self.engine_config.cache_block_seq_len,
            enable_prefix_caching=self.engine_config.enable_prefix_caching,
            quant_policy=self.engine_config.quant_policy,
            max_prefill_token_num=self.engine_config.max_prefill_token_num,
            num_tokens_per_iter=self.engine_config.num_tokens_per_iter,
            max_prefill_iters=self.engine_config.max_prefill_iters,
            async_=self.engine_config.async_,
            devices=self.engine_config.devices,
            nnodes=self.engine_config.nnodes,
            node_rank=self.engine_config.node_rank,
            communicator=self.engine_config.communicator,
            enable_metrics=self.engine_config.enable_metrics,
            download_dir=self.vllm_config.load_config.download_dir,
            revision=self.vllm_config.model_config.revision,
        )
        lm_cfg.attn_tp_size = self.engine_config.tp
        lm_cfg.attn_cp_size = self.engine_config.cp
        lm_cfg.mlp_tp_size = self.engine_config.tp
        # Align with lmdeploy runtime config finalization.
        update_parallel_config(lm_cfg)
        return lm_cfg

    def _build_config_from_hf(self) -> dict[str, Any]:
        try:
            from lmdeploy.turbomind.deploy.converter import get_tm_model
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "TurboMind conversion requires the vendored lmdeploy pipeline "
                "and its Python dependencies (e.g., mmengine)."
            ) from exc

        if _is_tm_debug_enabled():
            try:
                import lmdeploy.turbomind.deploy.converter as converter_mod

                logger.info(
                    "TurboMind debug converter module path: %s",
                    getattr(converter_mod, "__file__", None),
                )
            except Exception:
                logger.exception("TurboMind debug: failed to inspect converter module path.")

        engine_cfg = self._build_lmdeploy_engine_config()
        model_name = self._sanitize_model_id(self.model_dir)
        tm_model = get_tm_model(
            self.model_dir,
            model_name,
            "",
            engine_cfg,
            out_dir="",
        )
        tm_cfg = tm_model.tm_config
        tm_cfg.update_from_engine_config(engine_cfg)
        if _is_tm_debug_enabled():
            mcfg = tm_cfg.model_config
            logger.info(
                "TurboMind debug converter: model_name=%s model_arch=%s "
                "vocab_size=%s embedding_size=%s num_layer=%s hidden_units=%s "
                "head_num=%s kv_head_num=%s qk_norm=%s attn_bias=%s model_format=%s",
                getattr(mcfg, "model_name", None),
                getattr(mcfg, "model_arch", None),
                getattr(mcfg, "vocab_size", None),
                getattr(mcfg, "embedding_size", None),
                getattr(mcfg, "num_layer", None),
                getattr(mcfg, "hidden_units", None),
                getattr(mcfg, "head_num", None),
                getattr(mcfg, "kv_head_num", None),
                getattr(mcfg, "qk_norm", None),
                getattr(mcfg, "attn_bias", None),
                getattr(mcfg, "model_format", None),
            )
        self._tm_model = tm_model
        return tm_cfg.to_dict()

    def _create_model_comm(self):
        weight_type = self.config_dict.get("model_config", {}).get("weight_type", "float16")
        return _tm.TurboMind.create(
            model_dir="",
            config=yaml.safe_dump(self.config_dict),
            weight_type=weight_type,
        )

    def _create_weights(self) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.create_weights(device_id)

    def _process_weights(self) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.process_weight(device_id)

    def _create_engine(self) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.create_engine(device_id)

    def _load_weights_from_dir(self) -> None:
        tensor_maps = []
        for device_id in range(len(self.engine_config.devices or [])):
            tensor_maps.append(self.model_comm.get_weights(device_id))

        missing: list[str] = []
        loaded = 0
        total = 0
        for tensor_map in tensor_maps:
            for name, tm_tensor in tensor_map.items():
                total += 1
                file_path = os.path.join(self.model_dir, name)
                if not os.path.exists(file_path):
                    missing.append(name)
                    continue
                tensor = self._load_tensor_file(file_path, tm_tensor)
                tm_tensor.copy_from(tensor)
                loaded += 1
        if _is_tm_debug_enabled():
            logger.info(
                "TurboMind debug file-load: loaded %d/%d tensors from `%s`.",
                loaded,
                total,
                self.model_dir,
            )
            self._debug_log_weight_probe()
        if missing:
            sample = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"TurboMind weight files missing for {len(missing)} tensors. "
                f"Examples: {sample}"
            )

    def _load_weights_from_hf(self) -> None:
        if self._tm_model is None:
            raise RuntimeError("TurboMind conversion model is not initialized.")
        tm_params: dict[str, list] = {}
        for device_id in range(len(self.engine_config.devices or [])):
            tensor_map = self.model_comm.get_weights(device_id)
            for name, tm_tensor in tensor_map.items():
                tm_params.setdefault(name, []).append(tm_tensor)
        if _is_tm_debug_enabled():
            logger.info(
                "TurboMind debug export: collected %d tensor groups for HF->TM export.",
                len(tm_params),
            )
        self._tm_model.tm_params = tm_params
        device_id = (self.engine_config.devices or [0])[0]
        with torch.cuda.device(device_id):
            self._tm_model.export()
        if _is_tm_debug_enabled():
            logger.info(
                "TurboMind debug export: remaining tensors after export=%d",
                len(self._tm_model.tm_params),
            )
            self._debug_log_weight_probe()
        if tm_params:
            missing = ", ".join(sorted(tm_params.keys())[:32])
            logger.warning("TurboMind conversion left unused weights: %s", missing)

    def _load_tensor_file(self, file_path: str, tm_tensor) -> torch.Tensor:
        dtype = tm_tensor.type
        shape = tm_tensor.shape
        np_dtype = self._np_dtype_from_tm(dtype)
        data = np.fromfile(file_path, dtype=np_dtype)
        expected = int(np.prod(shape))
        if data.size != expected:
            raise ValueError(
                f"Invalid weight file {file_path}: expected {expected} elements, got {data.size}."
            )
        data = data.reshape(shape)
        if dtype == _tm.DataType.TYPE_BF16:
            torch_tensor = torch.from_numpy(data).view(torch.bfloat16)
        else:
            torch_tensor = torch.from_numpy(data)
        return torch_tensor

    def _np_dtype_from_tm(self, dtype) -> np.dtype:
        if dtype == _tm.DataType.TYPE_FP16:
            return np.float16
        if dtype == _tm.DataType.TYPE_BF16:
            return np.uint16
        if dtype == _tm.DataType.TYPE_FP32:
            return np.float32
        if dtype == _tm.DataType.TYPE_INT32:
            return np.int32
        if dtype == _tm.DataType.TYPE_INT16:
            return np.int16
        if dtype == _tm.DataType.TYPE_INT8:
            return np.int8
        if dtype == _tm.DataType.TYPE_UINT32:
            return np.uint32
        raise ValueError(f"Unsupported TurboMind dtype: {dtype}")

    def _debug_log_weight_probe(self) -> None:
        if not _is_tm_debug_enabled():
            return
        try:
            tensor_map = self.model_comm.get_weights(0)
            keys = list(tensor_map.keys())
            logger.info(
                "TurboMind debug weight probe: total tensors=%d first_keys=%s",
                len(keys),
                keys[:16],
            )
            selected: list[str] = []
            probe_patterns = (
                # Embedding / output
                "tok_embeddings",
                "output",
                # Attention projections
                "attention.w_qkv",
                "attention.wo",
                # FFN projections
                "ffn.w1",
                "ffn.w2",
                "ffn.w3",
                # QK norm
                "attention.q_norm",
                "attention.k_norm",
            )
            for pattern in probe_patterns:
                for k in keys:
                    if pattern in k:
                        selected.append(k)
                        break
            if len(selected) < 3:
                for k in sorted(keys):
                    if k not in selected:
                        selected.append(k)
                    if len(selected) >= 8:
                        break
            # Keep first occurrence order while deduplicating.
            selected_unique: list[str] = []
            for name in selected:
                if name not in selected_unique:
                    selected_unique.append(name)
            for name in selected_unique[:12]:
                tm_tensor = tensor_map[name]
                t = torch.from_dlpack(tm_tensor)
                if t.numel() == 0:
                    logger.info("TurboMind debug weight probe %s: empty tensor.", name)
                    continue
                n = min(4096, t.numel())
                sample = t.reshape(-1)[:n].float().cpu()
                finite_ratio = torch.isfinite(sample).float().mean().item()
                mean = sample.mean().item()
                std = sample.std(unbiased=False).item()
                min_val = sample.min().item()
                max_val = sample.max().item()
                logger.info(
                    "TurboMind debug weight probe %s: n=%d finite=%.6f "
                    "mean=%.6e std=%.6e min=%.6e max=%.6e",
                    name,
                    n,
                    finite_ratio,
                    mean,
                    std,
                    min_val,
                    max_val,
                )
        except Exception:
            logger.exception("TurboMind debug weight probe failed.")

    def create_request(self) -> "TurboMindRequestHandle":
        return TurboMindRequestHandle(self)

    def sleep(self, level: int = 1) -> None:
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.sleep(device_id, level)

    def wakeup(self, tags: list[str] | None = None) -> None:
        tags = tags or ["weights", "kv_cache"]
        for device_id in range(len(self.engine_config.devices or [])):
            self.model_comm.wakeup(device_id, tags)

    def close(self) -> None:
        self.model_comm = None


class TurboMindRequestHandle:
    def __init__(self, runtime: TurboMindRuntime):
        self.runtime = runtime
        self.model_inst = runtime.model_comm.create_request()

    async def async_cancel(self) -> None:
        self.model_inst.cancel()

    def async_end_cb(self, fut: asyncio.Future, status: int):
        fut.get_loop().call_soon_threadsafe(fut.set_result, status)

    async def async_end(self, session_id: int) -> None:
        fut = asyncio.get_running_loop().create_future()
        self.model_inst.end(lambda status: self.async_end_cb(fut, status), session_id)
        await fut

    async def async_stream_infer(
        self,
        session_id: int,
        input_ids: list[int],
        gen_cfg,
        stream_output: bool = True,
        sequence_start: bool = True,
        sequence_end: bool = True,
        step: int = 0,
    ) -> AsyncGenerator[TMOutput, None]:
        input_ids_tensor = torch.IntTensor(input_ids)
        inputs = {"input_ids": input_ids_tensor}
        inputs = _np_dict_to_tm_dict(inputs)

        session = _tm.SessionParam(
            id=session_id, step=step, start=sequence_start, end=sequence_end
        )

        sem = StreamingSemaphore()
        outputs, shared_state, _metrics = self.model_inst.forward(
            inputs,
            session,
            gen_cfg,
            stream_output,
            False,
            sem.release_threadsafe,
        )

        outputs = _tm_dict_to_torch_dict(outputs)
        output_ids_buf = outputs["output_ids"]

        output_logprobs = getattr(gen_cfg, "output_logprobs", None)
        extra_logprobs = (
            _get_logprobs(outputs, output_logprobs)
            if output_logprobs
            else None
        )

        prev_len = step + len(input_ids)
        state = None
        try:
            while True:
                await sem.acquire()
                state = shared_state.consume()
                status, seq_len = state.status, state.seq_len

                if seq_len == prev_len and status == 0:
                    continue

                if seq_len < prev_len:
                    yield TMOutput(token_ids=[], status=status)
                    break

                token_ids = output_ids_buf[prev_len:seq_len].tolist()
                out = TMOutput(token_ids=token_ids, status=status)
                if extra_logprobs is not None:
                    extra_logprobs(out)
                prev_len = seq_len
                yield out
                if status in (7, 8):  # finish / cancel
                    break
        finally:
            while not state or state.status == 0:
                await sem.acquire()
                state = shared_state.consume()


def _np_dict_to_tm_dict(np_dict: dict):
    ret = _tm.TensorMap()
    for k, v in np_dict.items():
        ret[k] = _tm.from_dlpack(v)
    return ret


def _tm_dict_to_torch_dict(tm_dict):
    ret = {}
    for k, v in tm_dict.items():
        if v.type == _tm.DataType.TYPE_UINT32:
            v = v.view(_tm.DataType.TYPE_INT32)
        ret[k] = torch.from_dlpack(v)
    return ret


class TurbomindEngineClient(EngineClient):
    """EngineClient adapter that uses TurboMind backend directly."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        engine_config: TurbomindEngineConfig | None = None,
    ) -> None:
        _require_turbomind()
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.input_processor = InputProcessor(vllm_config)
        self.io_processor = get_io_processor(
            vllm_config,
            self.model_config.io_processor_plugin,
        )
        self._engine_config = engine_config or self._build_engine_config(vllm_config)
        self._runtime = TurboMindRuntime(
            model_dir=self.model_config.model,
            engine_config=self._engine_config,
            vllm_config=self.vllm_config,
        )
        logger.info(
            "TurboMind backend binary: _turbomind=%s",
            getattr(_tm, "__file__", None),
        )
        try:
            import lmdeploy

            lmdeploy_path = getattr(lmdeploy, "__file__", None)
            logger.info(
                "TurboMind backend lmdeploy module: lmdeploy=%s",
                lmdeploy_path,
            )
            expected_prefix = str(_VENDORED_LMDEPLOY_PKG)
            if lmdeploy_path and expected_prefix not in lmdeploy_path:
                logger.warning(
                    "TurboMind backend: lmdeploy module is not loaded from vendored "
                    "path `%s` (actual `%s`). This can cause conversion/runtime "
                    "mismatch.",
                    expected_prefix,
                    lmdeploy_path,
                )
        except Exception:
            logger.exception("TurboMind backend: failed to inspect lmdeploy module path.")

        if _is_tm_debug_enabled():
            try:
                gen_probe = _tm.GenerationConfig()
                logger.info(
                    "TurboMind debug ABI: GenerationConfig fields "
                    "(top_k=%s top_p=%s min_p=%s temperature=%s "
                    "repetition_penalty=%s eos_ids=%s stop_ids=%s bad_ids=%s)",
                    hasattr(gen_probe, "top_k"),
                    hasattr(gen_probe, "top_p"),
                    hasattr(gen_probe, "min_p"),
                    hasattr(gen_probe, "temperature"),
                    hasattr(gen_probe, "repetition_penalty"),
                    hasattr(gen_probe, "eos_ids"),
                    hasattr(gen_probe, "stop_ids"),
                    hasattr(gen_probe, "bad_ids"),
                )
            except Exception:
                logger.exception("TurboMind debug ABI probe failed.")
        self._lmdeploy_tokenizer = None
        try:
            from lmdeploy.tokenizer import Tokenizer as LmdeployTokenizer

            self._lmdeploy_tokenizer = LmdeployTokenizer(self.model_config.model)
        except Exception:
            logger.exception(
                "TurboMind backend: failed to initialize lmdeploy tokenizer. "
                "Falling back to vLLM tokenizer path."
            )
        self._req_id_to_session: dict[str, int] = {}
        self._session_id_gen = itertools.count(1)
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stopped = False
        self._errored = False
        self._dead_error: BaseException | None = None
        logger.info(
            "TurboMind backend enabled (no external lmdeploy). "
            "HF models will be auto-converted in-memory if config.yaml is missing."
        )

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> "TurbomindEngineClient":
        return cls(vllm_config=vllm_config)

    @property
    def renderer(self) -> BaseRenderer:
        return self.input_processor.renderer

    @property
    def is_running(self) -> bool:
        return not self._stopped

    @property
    def is_stopped(self) -> bool:
        return self._stopped

    @property
    def errored(self) -> bool:
        return self._errored

    @property
    def dead_error(self) -> BaseException:
        if self._dead_error is None:
            return RuntimeError("TurboMind engine error state not set.")
        return self._dead_error

    async def generate(
        self,
        prompt: EngineCoreRequest | PromptType | AsyncGenerator[Any, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request=None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        if isinstance(prompt, AsyncGenerator):
            raise ValueError("Streaming inputs are not supported by turbomind backend.")
        if lora_request is not None:
            raise ValueError("LoRA is not supported by turbomind backend.")
        if tokenization_kwargs:
            logger.debug("tokenization_kwargs ignored by turbomind backend.")
        if trace_headers:
            logger.debug("trace_headers ignored by turbomind backend.")
        if priority or data_parallel_rank:
            logger.debug("priority/data_parallel_rank ignored by turbomind backend.")

        await self._pause_event.wait()

        input_ids: list[int] | None = None
        prompt_str: str | None = None
        params = sampling_params

        if isinstance(prompt, EngineCoreRequest):
            if prompt_text is not None:
                logger.debug(
                    "prompt_text provided with EngineCoreRequest; it is used only for output metadata."
                )
            if prompt.prompt_token_ids is None:
                raise ValueError("EngineCoreRequest.prompt_token_ids is required.")
            input_ids = prompt.prompt_token_ids
            prompt_str = prompt_text
        else:
            if prompt_text is not None:
                raise ValueError("prompt_text should only be provided with EngineCoreRequest.")
            if is_tokens_prompt(prompt):
                input_ids = prompt["prompt_token_ids"]
                prompt_str = prompt.get("prompt")
            else:
                prompt_str = get_prompt_text(prompt)
                if prompt_str is None:
                    raise ValueError("Unsupported prompt type for turbomind backend.")
                req = self.input_processor.process_inputs(
                    request_id=request_id,
                    prompt=prompt,
                    params=params,
                    tokenization_kwargs=None,
                )
                input_ids = req.prompt_token_ids
                params = req.params

        if input_ids is None:
            raise ValueError("prompt_token_ids are required for turbomind backend.")

        use_lmdeploy_tokenizer = os.getenv("VLLM_TURBOMIND_USE_LMDEPLOY_TOKENIZER", "1").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if use_lmdeploy_tokenizer and self._lmdeploy_tokenizer is not None and prompt_str is not None:
            try:
                lm_input_ids = self._lmdeploy_tokenizer.encode(prompt_str, add_bos=True)
                if lm_input_ids != input_ids:
                    mismatch_idx = None
                    limit = min(len(lm_input_ids), len(input_ids))
                    for i in range(limit):
                        if lm_input_ids[i] != input_ids[i]:
                            mismatch_idx = i
                            break
                    if mismatch_idx is None and len(lm_input_ids) != len(input_ids):
                        mismatch_idx = limit
                    logger.warning(
                        "TurboMind backend: prompt token ids differ from lmdeploy "
                        "tokenizer at idx=%s (vllm_len=%d lmdeploy_len=%d). "
                        "Using lmdeploy-tokenized prompt ids for TurboMind parity.",
                        mismatch_idx,
                        len(input_ids),
                        len(lm_input_ids),
                    )
                    input_ids = lm_input_ids
            except Exception:
                logger.exception(
                    "TurboMind backend: failed to retokenize prompt with lmdeploy tokenizer."
                )

        # Align with lmdeploy TurboMind prompt path: sequence_start prompts are
        # encoded with add_bos=True. Some models (e.g., Qwen family) can
        # degrade badly when BOS is omitted.
        disable_bos_compat = os.getenv("VLLM_TURBOMIND_DISABLE_BOS_COMPAT", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not disable_bos_compat and prompt_str is not None:
            bos_token_id = getattr(self.input_processor.tokenizer, "bos_token_id", None)
            if (
                bos_token_id is not None
                and len(input_ids) > 0
                and input_ids[0] != bos_token_id
            ):
                input_ids = [int(bos_token_id)] + input_ids
                logger.info(
                    "TurboMind backend: prepended BOS token id %d to prompt for "
                    "lmdeploy-compatible TurboMind tokenization.",
                    bos_token_id,
                )

        debug_enabled = _is_tm_debug_enabled()
        tokenizer = self.input_processor.tokenizer
        decode_tokenizer = (
            self._lmdeploy_tokenizer if (use_lmdeploy_tokenizer and self._lmdeploy_tokenizer is not None) else tokenizer
        )
        debug_events_left = 8
        tokenizer_vocab_size = None
        tokenizer_len = None
        lmdeploy_vocab_size = None
        tm_vocab_size = None
        if debug_enabled:
            if tokenizer is not None:
                try:
                    tokenizer_vocab_size = len(tokenizer.get_vocab())
                except Exception:
                    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
                try:
                    tokenizer_len = len(tokenizer)
                except Exception:
                    tokenizer_len = None
            if self._lmdeploy_tokenizer is not None:
                try:
                    lmdeploy_vocab_size = len(self._lmdeploy_tokenizer.get_vocab())
                except Exception:
                    lmdeploy_vocab_size = getattr(self._lmdeploy_tokenizer, "vocab_size", None)
            try:
                tm_vocab_size = self._runtime.config_dict.get("model_config", {}).get("vocab_size")
            except Exception:
                tm_vocab_size = None
            max_prompt_id = max(input_ids) if input_ids else None
            logger.info(
                "TurboMind debug request=%s prompt_len=%d max_tokens=%s detokenize=%s "
                "output_kind=%s tokenizer_vocab_size=%s tokenizer_len=%s "
                "lmdeploy_vocab_size=%s use_lmdeploy_tokenizer=%s "
                "tm_vocab_size=%s max_prompt_id=%s first_prompt_ids=%s "
                "last_prompt_ids=%s prompt_preview=%r",
                request_id,
                len(input_ids),
                params.max_tokens,
                params.detokenize,
                params.output_kind,
                tokenizer_vocab_size,
                tokenizer_len,
                lmdeploy_vocab_size,
                use_lmdeploy_tokenizer,
                tm_vocab_size,
                max_prompt_id,
                input_ids[:16],
                input_ids[-16:],
                (prompt_str[:200] if prompt_str else None),
            )
            if tm_vocab_size is not None and max_prompt_id is not None and max_prompt_id >= tm_vocab_size:
                logger.warning(
                    "TurboMind debug request=%s has prompt token id %s >= tm_vocab_size %s",
                    request_id,
                    max_prompt_id,
                    tm_vocab_size,
                )

        gen_cfg, stop_ids = self._sampling_params_to_tm_config(params)
        output_kind = params.output_kind
        # Align with lmdeploy async engine semantics:
        # non-stream responses use stream_output=False.
        stream_response = output_kind != RequestOutputKind.FINAL_ONLY
        if os.getenv("VLLM_TURBOMIND_FORCE_STREAM_OUTPUT", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            stream_response = True

        if debug_enabled:
            logger.info(
                "TurboMind debug sampling request=%s sampling_type=%s top_k=%s top_p=%s min_p=%s "
                "temperature=%s repetition_penalty=%s random_seed=%s max_new_tokens=%s "
                "stop_token_ids=%s bad_words=%s eos_ids=%s stop_ids=%s bad_ids=%s "
                "stream_output=%s",
                request_id,
                getattr(params, "sampling_type", None),
                getattr(gen_cfg, "top_k", None),
                getattr(gen_cfg, "top_p", None),
                getattr(gen_cfg, "min_p", None),
                getattr(gen_cfg, "temperature", None),
                getattr(gen_cfg, "repetition_penalty", None),
                getattr(gen_cfg, "random_seed", None),
                getattr(gen_cfg, "max_new_tokens", None),
                params.stop_token_ids,
                params.bad_words,
                getattr(gen_cfg, "eos_ids", None),
                getattr(gen_cfg, "stop_ids", None),
                getattr(gen_cfg, "bad_ids", None),
                stream_response,
            )

        session_id = self._req_id_to_session.get(request_id)
        if session_id is None:
            session_id = next(self._session_id_gen)
            self._req_id_to_session[request_id] = session_id

        cumulative_text = ""
        cumulative_token_ids: list[int] = []
        cumulative_logprobs: list[dict[int, float]] = []
        max_output_token_id = -1
        oov_vs_tokenizer_count = 0
        logprob_debug_emitted = False

        handle = self._runtime.create_request()
        try:
            async for out in handle.async_stream_infer(
                session_id=session_id,
                input_ids=input_ids,
                gen_cfg=gen_cfg,
                stream_output=stream_response,
                sequence_start=True,
                sequence_end=True,
                step=0,
            ):
                delta_token_ids = out.token_ids or []
                delta_logprobs = out.logprobs
                if delta_token_ids:
                    local_max = max(delta_token_ids)
                    if local_max > max_output_token_id:
                        max_output_token_id = local_max
                    if tokenizer_vocab_size is not None:
                        oov_vs_tokenizer_count += sum(
                            1 for tid in delta_token_ids if tid >= tokenizer_vocab_size
                        )

                if debug_enabled and debug_events_left > 0 and delta_token_ids:
                    debug_events_left -= 1
                    max_delta = max(delta_token_ids)
                    if tokenizer_vocab_size is not None and max_delta >= tokenizer_vocab_size:
                        logger.warning(
                            "TurboMind debug request=%s produced token id >= tokenizer vocab "
                            "(max_delta=%d >= tokenizer_vocab_size=%d).",
                            request_id,
                            max_delta,
                            tokenizer_vocab_size,
                        )
                    preview_text = None
                    preview_text_hf = None
                    preview_text_lmdeploy = None
                    if decode_tokenizer is not None:
                        try:
                            preview_text = decode_tokenizer.decode(
                                delta_token_ids[:16], skip_special_tokens=False
                            )
                        except Exception:
                            logger.exception(
                                "TurboMind debug request=%s failed to decode delta preview ids.",
                                request_id,
                            )
                    if tokenizer is not None:
                        try:
                            preview_text_hf = tokenizer.decode(
                                delta_token_ids[:16], skip_special_tokens=False
                            )
                        except Exception:
                            logger.exception(
                                "TurboMind debug request=%s failed HF decode for delta preview ids.",
                                request_id,
                            )
                    if self._lmdeploy_tokenizer is not None:
                        try:
                            preview_text_lmdeploy = self._lmdeploy_tokenizer.decode(
                                delta_token_ids[:16], skip_special_tokens=False
                            )
                        except Exception:
                            logger.exception(
                                "TurboMind debug request=%s failed lmdeploy decode for delta preview ids.",
                                request_id,
                            )
                    logger.info(
                        "TurboMind debug stream request=%s status=%s delta_len=%d total_len=%d "
                        "delta_ids=%s delta_text_preview=%r hf_preview=%r "
                        "lmdeploy_preview=%r",
                        request_id,
                        out.status,
                        len(delta_token_ids),
                        len(cumulative_token_ids) + len(delta_token_ids),
                        delta_token_ids[:16],
                        preview_text,
                        preview_text_hf,
                        preview_text_lmdeploy,
                    )

                if (
                    debug_enabled
                    and not logprob_debug_emitted
                    and delta_logprobs
                    and len(delta_logprobs) > 0
                    and decode_tokenizer is not None
                ):
                    try:
                        top = sorted(
                            delta_logprobs[0].items(),
                            key=lambda kv: kv[1],
                            reverse=True,
                        )[:10]
                        top_readable = [
                            (tid, lp, decode_tokenizer.decode([tid], skip_special_tokens=False))
                            for tid, lp in top
                        ]
                        logger.info(
                            "TurboMind debug request=%s first-token top-logprobs=%s",
                            request_id,
                            top_readable,
                        )
                        logprob_debug_emitted = True
                    except Exception:
                        logger.exception(
                            "TurboMind debug request=%s failed to format top-logprobs.",
                            request_id,
                        )

                if output_kind == RequestOutputKind.FINAL_ONLY:
                    cumulative_token_ids.extend(delta_token_ids)
                    if delta_logprobs is not None:
                        cumulative_logprobs.extend(delta_logprobs)
                    if out.status not in (7, 8):
                        continue
                    output_token_ids = list(cumulative_token_ids)
                    output_logprobs = (
                        list(cumulative_logprobs) if cumulative_logprobs else None
                    )
                elif output_kind == RequestOutputKind.CUMULATIVE:
                    cumulative_token_ids.extend(delta_token_ids)
                    if delta_logprobs is not None:
                        cumulative_logprobs.extend(delta_logprobs)
                    output_token_ids = list(cumulative_token_ids)
                    output_logprobs = (
                        list(cumulative_logprobs) if cumulative_logprobs else None
                    )
                else:
                    cumulative_token_ids.extend(delta_token_ids)
                    output_token_ids = list(delta_token_ids)
                    output_logprobs = delta_logprobs

                if params.detokenize:
                    if decode_tokenizer is None:
                        raise ValueError("Tokenizer unavailable for detokenize.")
                    if output_kind == RequestOutputKind.DELTA:
                        decoded_text = decode_tokenizer.decode(
                            cumulative_token_ids,
                            skip_special_tokens=params.skip_special_tokens,
                        )
                        if decoded_text.startswith(cumulative_text):
                            output_text = decoded_text[len(cumulative_text) :]
                        else:
                            output_text = decoded_text
                        cumulative_text = decoded_text
                    else:
                        output_text = decode_tokenizer.decode(
                            output_token_ids,
                            skip_special_tokens=params.skip_special_tokens,
                        )
                else:
                    output_text = ""

                finish_reason = None
                if out.status == 7:
                    finish_reason = "stop" if (stop_ids and delta_token_ids and delta_token_ids[-1] in stop_ids) else "length"
                elif out.status == 8:
                    finish_reason = "abort"
                elif out.status != 0:
                    finish_reason = "error"

                completion = CompletionOutput(
                    index=0,
                    text=output_text,
                    token_ids=output_token_ids,
                    cumulative_logprob=None,
                    logprobs=output_logprobs,
                    routed_experts=None,
                    finish_reason=finish_reason,
                    stop_reason=None,
                    lora_request=lora_request,
                )

                yield RequestOutput(
                    request_id=request_id,
                    prompt=prompt_str,
                    prompt_token_ids=input_ids,
                    prompt_logprobs=None,
                    outputs=[completion],
                    finished=finish_reason is not None,
                    lora_request=lora_request,
                )
            if debug_enabled:
                hf_full_preview = None
                lmdeploy_full_preview = None
                if cumulative_token_ids:
                    if tokenizer is not None:
                        try:
                            hf_full_preview = tokenizer.decode(
                                cumulative_token_ids[:64], skip_special_tokens=False
                            )
                        except Exception:
                            logger.exception(
                                "TurboMind debug request=%s failed HF full preview decode.",
                                request_id,
                            )
                    if self._lmdeploy_tokenizer is not None:
                        try:
                            lmdeploy_full_preview = self._lmdeploy_tokenizer.decode(
                                cumulative_token_ids[:64], skip_special_tokens=False
                            )
                        except Exception:
                            logger.exception(
                                "TurboMind debug request=%s failed lmdeploy full preview decode.",
                                request_id,
                            )
                logger.info(
                    "TurboMind debug request=%s finished: total_output_tokens=%d status=%s "
                    "max_output_token_id=%d oov_vs_tokenizer_count=%d "
                    "first_output_ids=%s hf_full_preview=%r lmdeploy_full_preview=%r",
                    request_id,
                    len(cumulative_token_ids),
                    out.status if "out" in locals() else None,
                    max_output_token_id,
                    oov_vs_tokenizer_count,
                    cumulative_token_ids[:16],
                    hf_full_preview,
                    lmdeploy_full_preview,
                )
        except Exception as exc:
            self._errored = True
            self._dead_error = exc
            logger.exception("TurboMind generation failed for request %s", request_id)
            await handle.async_cancel()
            raise
        finally:
            self._req_id_to_session.pop(request_id, None)
            # TurboMind ends sessions internally when sequence_end is True.

    async def encode(
        self,
        prompt: PromptType,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request=None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]:
        raise NotImplementedError("Pooling is not supported by turbomind backend.")

    async def abort(self, request_id: str | Iterable[str]) -> None:
        req_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        for req_id in req_ids:
            session_id = self._req_id_to_session.pop(req_id, None)
            if session_id is None:
                continue
            handle = self._runtime.create_request()
            await handle.async_cancel()

    async def is_tracing_enabled(self) -> bool:
        return False

    async def do_log_stats(self) -> None:
        return

    async def check_health(self) -> None:
        if self.errored:
            raise self.dead_error

    async def start_profile(self) -> None:
        logger.warning("Profiling is not supported by turbomind backend.")

    async def stop_profile(self) -> None:
        logger.warning("Profiling is not supported by turbomind backend.")

    async def reset_mm_cache(self) -> None:
        self.input_processor.clear_mm_cache()

    async def reset_encoder_cache(self) -> None:
        return

    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        return False

    async def sleep(self, level: int = 1) -> None:
        self._runtime.sleep(level)

    async def wake_up(self, tags: list[str] | None = None) -> None:
        self._runtime.wakeup(tags)

    async def is_sleeping(self) -> bool:
        return False

    async def add_lora(self, lora_request) -> bool:
        logger.warning("LoRA is not supported by turbomind backend.")
        return False

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None:
        self._paused = True
        self._pause_event.clear()

    async def resume_generation(self) -> None:
        self._paused = False
        self._pause_event.set()

    async def is_paused(self) -> bool:
        return self._paused

    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ):
        raise NotImplementedError("collective_rpc is not supported by turbomind backend.")

    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def shutdown(self) -> None:
        if self._stopped:
            return
        self._runtime.close()
        self._stopped = True

    def _build_engine_config(self, vllm_config: VllmConfig) -> TurbomindEngineConfig:
        config = TurbomindEngineConfig()
        config.tp = vllm_config.parallel_config.tensor_parallel_size
        config.cp = vllm_config.parallel_config.decode_context_parallel_size
        if vllm_config.model_config.max_model_len and vllm_config.model_config.max_model_len > 0:
            config.session_len = vllm_config.model_config.max_model_len
        if vllm_config.scheduler_config.max_num_seqs is not None:
            config.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        # vLLM's paged-attention block size is not equivalent to TurboMind's
        # cache_block_seq_len. Mapping it directly (often 16) can make decode
        # path unstable for some models. Keep TurboMind default unless user
        # explicitly overrides it.
        tm_cache_block_env = os.getenv("VLLM_TURBOMIND_CACHE_BLOCK_SEQ_LEN", "").strip()
        if tm_cache_block_env:
            try:
                tm_cache_block = int(tm_cache_block_env)
            except ValueError:
                logger.warning(
                    "TurboMind backend: invalid VLLM_TURBOMIND_CACHE_BLOCK_SEQ_LEN=%r; "
                    "using default %d.",
                    tm_cache_block_env,
                    config.cache_block_seq_len,
                )
            else:
                if tm_cache_block > 0:
                    config.cache_block_seq_len = tm_cache_block
                else:
                    logger.warning(
                        "TurboMind backend: VLLM_TURBOMIND_CACHE_BLOCK_SEQ_LEN must be > 0; "
                        "got %d. Using default %d.",
                        tm_cache_block,
                        config.cache_block_seq_len,
                    )
        elif _is_tm_debug_enabled() and vllm_config.cache_config.block_size:
            logger.info(
                "TurboMind debug: ignoring vLLM cache_config.block_size=%s for "
                "cache_block_seq_len; using TurboMind default=%s. "
                "Set VLLM_TURBOMIND_CACHE_BLOCK_SEQ_LEN to override.",
                vllm_config.cache_config.block_size,
                config.cache_block_seq_len,
            )
        config.cache_max_entry_count = vllm_config.cache_config.gpu_memory_utilization
        config.enable_prefix_caching = vllm_config.cache_config.enable_prefix_caching
        config.devices = list(range(config.tp))
        config.enable_metrics = False
        return config

    def _get_tokenizer_eos_token_ids(self) -> list[int]:
        tokenizer = self.input_processor.tokenizer
        eos_ids: list[int] = []
        if tokenizer is None:
            return eos_ids

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_ids.append(int(eos_token_id))
        elif isinstance(eos_token_id, (list, tuple, set)):
            eos_ids.extend(int(x) for x in eos_token_id if isinstance(x, (int, np.integer)))

        eos_token_ids = getattr(tokenizer, "eos_token_ids", None)
        if isinstance(eos_token_ids, (list, tuple, set)):
            eos_ids.extend(int(x) for x in eos_token_ids if isinstance(x, (int, np.integer)))
        elif isinstance(eos_token_ids, int):
            eos_ids.append(int(eos_token_ids))

        # Preserve order while deduplicating.
        unique: list[int] = []
        for tok_id in eos_ids:
            if tok_id not in unique:
                unique.append(tok_id)
        return unique

    def _sampling_params_to_tm_config(
        self, params: SamplingParams
    ) -> tuple[Any, list[int]]:
        if params.n > 1:
            raise ValueError("TurboMind backend does not support n > 1.")
        if params.presence_penalty or params.frequency_penalty:
            raise ValueError("presence_penalty/frequency_penalty are not supported by turbomind backend.")
        if params.prompt_logprobs is not None:
            raise ValueError("prompt_logprobs are not supported by turbomind backend.")
        if params.logits_processors is not None:
            raise ValueError("logits_processors are not supported by turbomind backend.")
        if params.logit_bias is not None or params.allowed_token_ids is not None:
            raise ValueError("logit_bias/allowed_token_ids are not supported by turbomind backend.")
        if params.structured_outputs is not None:
            raise ValueError("structured_outputs are not supported by turbomind backend.")
        if params.stop:
            raise ValueError("stop strings are not supported by turbomind backend.")

        cfg = _tm.GenerationConfig()
        cfg.max_new_tokens = params.max_tokens or 0
        cfg.min_new_tokens = params.min_tokens or 0
        cfg.top_k = 0 if params.top_k in (0, -1) else params.top_k
        cfg.top_p = params.top_p
        cfg.min_p = params.min_p
        cfg.temperature = params.temperature
        cfg.repetition_penalty = params.repetition_penalty

        # Debug isolation: force deterministic greedy decode to separate
        # sampling randomness from model/logit corruption.
        force_greedy = os.getenv("VLLM_TURBOMIND_FORCE_GREEDY", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if force_greedy:
            cfg.top_k = 1
            cfg.top_p = 1.0
            cfg.min_p = 0.0
            cfg.temperature = 1.0
            logger.warning(
                "TurboMind backend: forcing greedy sampling for debug "
                "(top_k=1, top_p=1.0, temperature=1.0)."
            )

        stop_token_ids = list(params.stop_token_ids or [])
        all_stop_token_ids = sorted(set(getattr(params, "all_stop_token_ids", set()) or set()))
        strict_lmdeploy_stop = os.getenv("VLLM_TURBOMIND_STRICT_LMDEPLOY_STOP", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if not strict_lmdeploy_stop:
            if not params.ignore_eos:
                stop_token_ids = sorted(set(all_stop_token_ids).union(stop_token_ids))

        # Always ensure EOS fallback exists when caller did not provide stop
        # token ids. This prevents unbounded generation in stream mode.
        if not stop_token_ids and not params.ignore_eos:
            stop_token_ids = self._get_tokenizer_eos_token_ids()

        if stop_token_ids:
            cfg.eos_ids = stop_token_ids
            if not params.ignore_eos:
                cfg.stop_ids = _construct_stop_or_bad_words(stop_token_ids)

        bad_token_ids: list[int] = []
        bad_words_token_ids = getattr(params, "bad_words_token_ids", None)
        if bad_words_token_ids:
            for token_seq in bad_words_token_ids:
                if not token_seq:
                    continue
                # TurboMind bad_ids currently supports token ids directly.
                # Keep single-token bad words and ignore multi-token phrases.
                if len(token_seq) == 1:
                    bad_token_ids.append(int(token_seq[0]))
        if bad_token_ids:
            cfg.bad_ids = _construct_stop_or_bad_words(bad_token_ids)
        debug_logprobs = os.getenv("VLLM_TURBOMIND_DEBUG_OUTPUT_LOGPROBS", "")
        if debug_logprobs:
            try:
                cfg.output_logprobs = max(1, min(int(debug_logprobs), 1024))
            except ValueError:
                cfg.output_logprobs = 20
        elif params.logprobs:
            cfg.output_logprobs = min(params.logprobs, 1024)
        if params.seed is not None:
            cfg.random_seed = params.seed
        else:
            disable_auto_seed = os.getenv("VLLM_TURBOMIND_NO_AUTO_RANDOM_SEED", "").lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            # In greedy debug mode, random_seed should be irrelevant.
            # Avoid writing random_seed to reduce chances of ABI field-width
            # mismatch affecting other generation fields.
            if (
                params.sampling_type == SamplingType.RANDOM
                and not force_greedy
                and not disable_auto_seed
            ):
                # Use 31-bit range for broad ABI compatibility.
                cfg.random_seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
        return cfg, stop_token_ids
