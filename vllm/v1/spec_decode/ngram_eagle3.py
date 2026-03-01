# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle import EagleSpeculator


def sampled_token_ids_list_to_padded_tensor(
    sampled_token_ids: list[list[int]],
    num_spec_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    padded_sampled_token_ids = torch.full(
        (len(sampled_token_ids), num_spec_tokens + 1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for i, token_ids in enumerate(sampled_token_ids):
        if not token_ids:
            continue
        valid_len = min(len(token_ids), num_spec_tokens + 1)
        padded_sampled_token_ids[i, :valid_len] = torch.tensor(
            token_ids[:valid_len], dtype=torch.int32, device=device
        )
    return padded_sampled_token_ids


def draft_token_ids_list_to_padded_tensor(
    draft_token_ids: list[list[int]],
    num_spec_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    padded_draft_token_ids = torch.full(
        (len(draft_token_ids), num_spec_tokens),
        0,
        dtype=torch.int32,
        device=device,
    )
    for i, token_ids in enumerate(draft_token_ids):
        if not token_ids:
            continue
        valid_len = min(len(token_ids), num_spec_tokens)
        padded_draft_token_ids[i, :valid_len] = torch.tensor(
            token_ids[:valid_len], dtype=torch.int32, device=device
        )
    return padded_draft_token_ids


def merge_ngram_first_drafts(
    sampled_token_ids: list[list[int]],
    ngram_draft_token_ids: list[list[int]],
    fallback_draft_token_ids: list[list[int]] | torch.Tensor,
) -> list[list[int]]:
    fallback_as_list = (
        fallback_draft_token_ids.tolist()
        if isinstance(fallback_draft_token_ids, torch.Tensor)
        else fallback_draft_token_ids
    )
    merged_draft_token_ids: list[list[int]] = []
    for sampled_ids, ngram_ids, fallback_ids in zip(
        sampled_token_ids, ngram_draft_token_ids, fallback_as_list
    ):
        if ngram_ids:
            merged_draft_token_ids.append(ngram_ids)
        elif sampled_ids:
            merged_draft_token_ids.append(fallback_ids)
        else:
            # Skip speculative drafting for requests without sampled tokens.
            merged_draft_token_ids.append([])
    return merged_draft_token_ids


class NgramEagle3Speculator:
    """Ngram-first + Eagle3-fallback speculator for V2 model runner."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.method == "eagle3"

        self.eagle_speculator = EagleSpeculator(vllm_config, device)
        self.ngram_proposer = NgramProposer(vllm_config)

        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.model_config.max_model_len
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.token_ids_cpu = np.zeros((max_num_reqs, max_model_len), dtype=np.int32)
        self.req_id_by_req_state_idx: list[str | None] = [None] * max_num_reqs

    def load_model(self, target_model: nn.Module) -> None:
        self.eagle_speculator.load_model(target_model)

    def set_attn(self, *args, **kwargs) -> None:
        self.eagle_speculator.set_attn(*args, **kwargs)

    def run_model(self, *args, **kwargs):
        return self.eagle_speculator.run_model(*args, **kwargs)

    def generate_draft(self, *args, **kwargs) -> None:
        self.eagle_speculator.generate_draft(*args, **kwargs)

    def capture_model(self) -> None:
        self.eagle_speculator.capture_model()

    @property
    def model(self):
        return getattr(self.eagle_speculator, "model", None)

    def _extract_sampled_token_ids(
        self, sampled_tokens: torch.Tensor, num_sampled: torch.Tensor
    ) -> list[list[int]]:
        sampled_token_ids: list[list[int]] = []
        sampled_tokens_cpu = sampled_tokens.to("cpu")
        num_sampled_cpu = num_sampled.to("cpu")
        for i in range(sampled_tokens_cpu.shape[0]):
            num_valid = int(num_sampled_cpu[i].item())
            if num_valid <= 0:
                sampled_token_ids.append([])
                continue
            sampled_token_ids.append(sampled_tokens_cpu[i, :num_valid].tolist())
        return sampled_token_ids

    def _sync_ngram_context(
        self,
        req_ids: Sequence[str],
        idx_mapping_np: np.ndarray,
        sampled_token_ids: list[list[int]],
        prefill_token_ids: torch.Tensor,
        prefill_len: np.ndarray,
    ) -> None:
        for i, req_id in enumerate(req_ids):
            req_state_idx = int(idx_mapping_np[i])
            if self.req_id_by_req_state_idx[req_state_idx] != req_id:
                # New request reusing this req_state slot.
                self.req_id_by_req_state_idx[req_state_idx] = req_id
                prefill_size = int(prefill_len[req_state_idx])
                if prefill_size > 0:
                    prompt_tokens = (
                        prefill_token_ids[req_state_idx, :prefill_size].to("cpu").numpy()
                    )
                    self.token_ids_cpu[req_state_idx, :prefill_size] = prompt_tokens
                self.num_tokens_no_spec[req_state_idx] = prefill_size

            sampled_ids = sampled_token_ids[i]
            if not sampled_ids:
                continue
            cur_size = int(self.num_tokens_no_spec[req_state_idx])
            max_write = min(len(sampled_ids), self.token_ids_cpu.shape[1] - cur_size)
            if max_write <= 0:
                continue
            self.token_ids_cpu[req_state_idx, cur_size : cur_size + max_write] = (
                sampled_ids[:max_write]
            )
            self.num_tokens_no_spec[req_state_idx] = cur_size + max_write

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        sampled_tokens: torch.Tensor | None = None,
        req_ids: Sequence[str] | None = None,
        idx_mapping_np: np.ndarray | None = None,
        prefill_token_ids: torch.Tensor | None = None,
        prefill_len: np.ndarray | None = None,
    ) -> torch.Tensor:
        draft_tokens = self.eagle_speculator.propose(
            input_batch=input_batch,
            last_hidden_states=last_hidden_states,
            aux_hidden_states=aux_hidden_states,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
            last_sampled=last_sampled,
            next_prefill_tokens=next_prefill_tokens,
            temperature=temperature,
            seeds=seeds,
        )

        if (
            sampled_tokens is None
            or req_ids is None
            or idx_mapping_np is None
            or prefill_token_ids is None
            or prefill_len is None
        ):
            return draft_tokens

        sampled_token_ids = self._extract_sampled_token_ids(sampled_tokens, num_sampled)
        self._sync_ngram_context(
            req_ids=req_ids,
            idx_mapping_np=idx_mapping_np,
            sampled_token_ids=sampled_token_ids,
            prefill_token_ids=prefill_token_ids,
            prefill_len=prefill_len,
        )
        req_state_indices = idx_mapping_np.astype(np.int32, copy=False)
        batch_num_tokens_no_spec = self.num_tokens_no_spec[req_state_indices]
        batch_token_ids_cpu = self.token_ids_cpu[req_state_indices]
        ngram_draft_tokens = self.ngram_proposer.propose(
            sampled_token_ids=sampled_token_ids,
            num_tokens_no_spec=batch_num_tokens_no_spec,
            token_ids_cpu=batch_token_ids_cpu,
            slot_mappings=None,
        )
        if not any(ngram_draft_tokens):
            return draft_tokens

        merged = draft_tokens.clone()
        for i, ngram_tokens in enumerate(ngram_draft_tokens):
            if not ngram_tokens:
                continue
            ngram_len = min(len(ngram_tokens), merged.shape[1])
            merged[i, :ngram_len] = torch.tensor(
                ngram_tokens[:ngram_len], dtype=merged.dtype, device=merged.device
            )
        return merged
