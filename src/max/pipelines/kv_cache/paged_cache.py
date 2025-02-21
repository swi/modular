# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""PagedAttention-enabled KV cache for the Transformer leveraging the mo.opaque pattern."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import reduce
from itertools import chain
from operator import mul
from typing import Any, Dict, Iterator, Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    _OpaqueType,
    _OpaqueValue,
    ops,
)

from ._utils import build_max_lengths_tensor
from .cache_params import KVCacheParams
from .manager import KVCacheInputSymbols, KVCacheManager
from .radix_trie import RadixTrie, TrieNode

PERCENTAGE_BLOCKS_TO_EVICT = 0.05


def ceildiv(n: int, d: int) -> int:
    """Compute ceil(n/d) using strictly integer arithmetic."""
    q, r = divmod(n, d)
    return q + bool(r)


@dataclass
class PagedCacheInputSymbols(KVCacheInputSymbols):
    kv_blocks: TensorType
    cache_lengths: TensorType
    lookup_table: TensorType
    max_lengths: TensorType


class PagedKVCacheType(_OpaqueType):
    """PagedAttention Mojo KV Cache graph type."""

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV Cache."""
        super().__init__("PagedKVCache")


class PagedKVCacheCollectionType(_OpaqueType):
    """The graph type for a "view" of the cache for the given sequences in the
    batch.

    This object does not own the underlying buffers in k_cache and v_cache,
    it's borrowing them from the BlockWrappers in our ContinuousKVCacheManager.
    It does own the Pointer[NDBuffer[type, 3]] and valid_lengths buffer
    """

    def __init__(self) -> None:
        """Creates an opaque type containing a paged KV cache collection."""
        super().__init__("PagedKVCacheCollection")


class PagedKVCache(_OpaqueValue):
    """PagedAttention Mojo KV cache graph value."""


class PagedKVCacheCollection(_OpaqueValue):
    """The graph value for a view of the KV cache."""


class FetchPagedKVCacheCollection:
    def __init__(self, kv_params: KVCacheParams) -> None:
        self.kv_params = kv_params

    def __call__(
        self,
        blocks: TensorValue,  # NDBuffer[type, 6, Self.blocks_shape]
        cache_lengths: TensorValue,  # NDBuffer[DType.uint32, 1],
        lookup_table: TensorValue,  # NDBuffer[DType.uint32, 2],
        is_cache_empty: TensorValue,
    ) -> PagedKVCacheCollection:
        """Constructs a PagedKVCacheCollection for use downstream."""

        # Explicit validation.
        if blocks.dtype != self.kv_params.dtype:
            msg = (
                f"expected blocks to be dtype: {self.kv_params.dtype}, got"
                f" {blocks.dtype}"
            )
            raise ValueError(msg)

        if blocks.rank != 6:
            msg = f"expected blocks to be of rank 6, got {blocks.rank}"
            raise ValueError(msg)

        # For all tensors other than the blocks tensor, the length should be equivalent
        # to batch size, which is unknown within the graph at this stage.
        if cache_lengths.dtype != DType.uint32:
            msg = f"expected cache lengths to be dtype: uint32, got {cache_lengths.dtype}"
            raise ValueError(msg)

        if cache_lengths.rank != 1:
            msg = f"expected cache lengths to be of rank 1, got {cache_lengths.rank}"
            raise ValueError(msg)

        if lookup_table.dtype != DType.uint32:
            msg = f"expected lookup_table to be dtype: uint32, got {lookup_table.dtype}"
            raise ValueError(msg)

        if lookup_table.rank != 2:
            msg = f"expected lookup_table to be of rank 2, got {lookup_table.rank}"
            raise ValueError(msg)

        return PagedKVCacheCollection(
            ops.custom(
                "mo.kv_collection_ctor.paged",
                values=[blocks, cache_lengths, lookup_table, is_cache_empty],
                out_types=[PagedKVCacheCollectionType()],
                parameters={
                    "num_heads": self.kv_params.n_kv_heads_per_device,
                    "head_dim": self.kv_params.head_dim,
                    "page_size": int(blocks.shape[3]),
                },
            )[0].opaque
        )


def construct_cow_strided_memcpy_graph(
    block_shape: list[int | str], dtype: DType, devices: list[Device]
) -> Graph:
    """
    Returns a graph for performing COW operations on the KV cache.
    """

    assert len(block_shape) == 6
    device_refs = [DeviceRef(device.label, device.id) for device in devices]
    blocks_ty = [
        BufferType(dtype, shape=block_shape, device=device_ref)
        for device_ref in device_refs
    ]
    block_src_idx_ty = TensorType(DType.uint32, shape=[])
    block_dst_idx_ty = TensorType(DType.uint32, shape=[])
    num_tokens_ty = TensorType(DType.uint32, shape=[])

    with Graph(
        "mo.kv_collection_cow_strided_memcpy.paged",
        input_types=[
            block_dst_idx_ty,
            block_src_idx_ty,
            num_tokens_ty,
            *blocks_ty,
        ],
        output_types=[],
    ) as graph:
        block_dst_idx, block_src_idx, num_tokens, *all_blocks = graph.inputs
        for blocks in all_blocks:
            ops.inplace_custom(
                "mo.kv_collection_cow_strided_memcpy.paged",
                values=[blocks, block_dst_idx, block_src_idx, num_tokens],
                out_types=[],
            )
        graph.output()

    return graph


@dataclass
class _PagedCacheMetadata:
    # Committed blocks are part of the radix trie and can be shared by many sequences.
    # They are used by the current sequence and possibly other sequences.
    committed_blocks: list[int] = field(default_factory=list)
    # Inflight blocks are not part of the radix trie and are not shared.
    # They are only used by the current sequence.
    inflight_blocks: list[int] = field(default_factory=list)

    # Leftover tokens from a prior call to step that were not committed because
    # they were in a partially filled block.
    previous_uncommitted_tokens: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )

    # This is a pointer into the radix trie indicating which prefix of the sequence
    # has been cached and committed into the radix trie.
    node: Optional[TrieNode] = None

    @property
    def all_assigned_blocks(self) -> Iterator[int]:
        return chain(self.committed_blocks, self.inflight_blocks)


class PagedKVCacheManager(KVCacheManager):
    def __init__(
        self,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        devices: list[Device],
        session: InferenceSession,
        cache_memory: int,
        page_size: int = 128,
    ):
        """
        Args:
            params: The KVCacheParams for the given pipeline.
            max_batch_size: The maximum number of active
                requests that the manager should support.
            max_seq_len: The maximum sequence length we will generate.
            num_layers: The number of layers in the model.
            devices: The devices on which the manager will allocate memory.
            session: The inference session to load ops from.
            cache_memory: The total amount of memory available for caching.
                This is aggregated across all devices.
            page_size: The number of tokens that will be stored in a single page.
        """
        # The number of tokens in a single page.
        self.page_size = page_size

        # The number of bytes that a single page will occupy.
        single_page_size_bytes = (
            2
            * num_layers
            * params.n_kv_heads_per_device
            * params.head_dim
            * page_size
            * params.dtype.size_in_bytes
        )

        # Normalize cache_memory across all devices.
        cache_memory_per_device = cache_memory // len(devices)

        # The total number of pages we'll have per-device.
        self.total_num_pages = int(
            cache_memory_per_device // single_page_size_bytes
        )

        if self.total_num_pages == 0:
            raise RuntimeError(
                f"Insufficient cache memory to allocate even a single page.\n"
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available."
            )

        if max_batch_size > self.total_num_pages:
            raise RuntimeError(
                f"Not enough cache memory to support a batch containing {max_batch_size} sequences.\n"
                f"Need to allocate at least {max_batch_size} blocks, but only have enough memory for {self.total_num_pages} blocks.\n"
                f"One page requires {single_page_size_bytes} bytes but only {cache_memory_per_device} bytes are available.\n"
                f"You must restart your process and set a smaller batch size."
            )

        # call our base class constructor
        super().__init__(
            params=params,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            devices=devices,
            session=session,
            is_ragged=True,
        )

        # Initialize the set of available blocks.
        self.available_blocks = set(range(self.total_num_pages))

        # Initialize the blocks for each device.
        self.blocks: list[Tensor] = []
        for device in self.devices:
            self.blocks.append(
                Tensor.zeros(
                    self.block_shape(),  # type: ignore
                    self.params.dtype,
                    device=device,
                )
            )

        self.active_requests: Dict[int, _PagedCacheMetadata] = {}

        # Initialize the radix trie if prefix caching is enabled.
        self.radix_trie: Optional[RadixTrie] = None
        self.cow_strided_memcpy_graph: Optional[Model] = None
        if params.enable_prefix_caching:
            self.radix_trie = RadixTrie(page_size=self.page_size)
            # Load single op graph for performing memory transfers needed for COW
            if self.page_size > 1:
                self.cow_strided_memcpy_graph = session.load(
                    construct_cow_strided_memcpy_graph(
                        self.block_shape(is_parameterized=True),
                        params.dtype,
                        devices,
                    ),
                )
        self.ce_all_tokens = 0
        self.ce_cache_hit_tokens = 0

    def cache_hit_rate(self) -> float:
        """Returns the prefix cache hit rate.

        This is the number of CE prompt tokens that were read from the prefix
        cache divided by the total number of CE prompt tokens.
        """
        if self.ce_all_tokens == 0:
            return 0.0
        return self.ce_cache_hit_tokens / self.ce_all_tokens

    def evict_blocks(self, percentage_to_evict: float = 1.0):
        if self.radix_trie is None:
            return

        # Evict a percentage of all blocks according to a LRU policy on the
        # trie leaves.
        evicted_blocks = self.radix_trie.evict_blocks(
            desired_num_evicted=int(
                max(1, self.total_num_pages * percentage_to_evict)
            )
        )

        for block in evicted_blocks:
            assert block not in self.available_blocks
            self.available_blocks.add(block)

    def alloc_block(self) -> int:
        if len(self.available_blocks) == 0:
            self.evict_blocks(percentage_to_evict=PERCENTAGE_BLOCKS_TO_EVICT)

        if len(self.available_blocks) == 0:
            raise RuntimeError(
                f"All {self.total_num_pages} KVCache pages have been exhausted! "
                "You must restart your process and set a smaller batch size or max seq len."
            )

        block = self.available_blocks.pop()
        return block

    def release_block(self, block: int, is_committed: bool = False) -> None:
        """We can release a block if prefix caching is disabled or if it is not committed.

        If it is committed, it may be in the radix tree and in use by other sequences.
        This means it can't be safely released without further checks.
        """
        if self.radix_trie is None or not is_committed:
            self.available_blocks.add(block)

    @classmethod
    def _block_size_per_token(
        cls, params: KVCacheParams, num_layers: int
    ) -> int:
        return (
            reduce(mul, cls._block_shape(params, 1, 1, num_layers), 1)
            * params.dtype.size_in_bytes
        )

    @classmethod
    def estimated_memory_size(
        cls,
        params: KVCacheParams,
        max_batch_size: int,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        # Determine how much size is necessary to store the full cache based on max_batch_size and max_seq_len.
        # If that's less than available_cache_memory, return that.
        # Otherwise, return available_cache_memory.
        # This is to prevent over-allocation on devices with a large amount of free memory (e.g. CPUs).
        assert params.page_size is not None
        block_size_per_token = cls._block_size_per_token(
            params, num_layers
        ) * len(devices)

        # round up our max_seq_len to the nearest page_size
        max_seq_len_round_up = (
            math.ceil(max_seq_len / params.page_size) * params.page_size
        )
        size_to_support_full_cache = (
            block_size_per_token * max_batch_size * max_seq_len_round_up
        )

        # return the minimum of the two
        return min(available_cache_memory, size_to_support_full_cache)

    @classmethod
    def infer_optimal_batch_size(
        cls,
        params: KVCacheParams,
        max_seq_len: int,
        num_layers: int,
        available_cache_memory: int,
        devices: list[Device],
        **kwargs: Any,
    ) -> int:
        # We just hard-code a default of 512 for paged attention.
        # The worst case scenario if this is too high is that we'll evict
        # requests at an elevated rate. We print warnings in that case so users
        # are aware of what needs to be tweaked/changed.
        return 512

    def block_shape(
        self,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        return self._block_shape(
            self.params,
            self.total_num_pages,
            self.page_size,
            self.num_layers,
            is_parameterized,
        )

    @classmethod
    def _block_shape(
        cls,
        params: KVCacheParams,
        total_num_pages: int,
        page_size: int,
        num_layers: int,
        is_parameterized: bool = False,
    ) -> list[int | str]:
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        kv_dim = 2
        return [
            num_layers,
            kv_dim,
            "total_num_pages" if is_parameterized else total_num_pages,
            page_size,
            params.n_kv_heads_per_device,
            params.head_dim,
        ]

    def get_num_free_blocks(self) -> int:
        if self.radix_trie is None:
            return len(self.available_blocks)
        else:
            return len(self.available_blocks) + len(
                self.radix_trie.get_evictable_blocks()
            )

    def can_fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> bool:
        """Checks if there are sufficient KV pages to run `fetch` on given batch.

        It is OK if some seq_id are not in the cache. We assume the cache lengths
        are zero in those cases.
        """

        total_blocks_to_allocate = 0
        all_cache_hit_blocks: set[int] = set()

        for seq_id, prompt in seq_ids_and_prompts.items():
            data = self.active_requests.get(seq_id, _PagedCacheMetadata())

            # Extend the kv cache for given request with any cached prefixes.
            cached_blocks: list[int] = []
            if self.radix_trie is not None:
                # Attempt to match all but the last token in the prompt. This is
                # because the model expects a prompt of length at least 1.
                _, cached_blocks = self.radix_trie.match_prefix(
                    prompt[:-1], node=data.node
                )

            cache_length = self.cache_lengths.get(seq_id, 0)

            # Compute the total sequence length and the number of pages required to store it.
            total_sequence_length = cache_length + len(prompt) + num_steps - 1
            num_pages_required = ceildiv(total_sequence_length, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            assert len(data.inflight_blocks) <= 1
            blocks_to_allocate = (
                num_pages_required
                - len(data.committed_blocks)
                - len(data.inflight_blocks)
                - len(cached_blocks)
            )

            total_blocks_to_allocate += blocks_to_allocate
            all_cache_hit_blocks.update(cached_blocks)

        num_evictable_blocks = 0
        if self.radix_trie is not None:
            # the blocks in the prefix cache that will be used by sequences in
            # this batch are no longer eligible for eviction / allocation.
            num_evictable_blocks = len(
                self.radix_trie.get_evictable_blocks() - all_cache_hit_blocks
            )

        num_free_blocks = len(self.available_blocks) + num_evictable_blocks

        return total_blocks_to_allocate <= num_free_blocks

    def get_num_cached_tokens(self, prompt: np.ndarray) -> int:
        """Returns the number of tokens in the CE prompt that are found in the
        prefix cache.
        """
        if self.radix_trie is None:
            return 0
        _, cached_blocks = self.radix_trie.match_prefix(prompt[:-1])
        return len(cached_blocks) * self.page_size

    def _fetch_prefix_caching(
        self,
        seq_id: int,
        data: _PagedCacheMetadata,
        prompt: np.ndarray,
    ) -> tuple[np.ndarray, list[int]]:
        """Extend the kv cache for given request with any cached prefixes."""
        assert self.radix_trie is not None
        uncommitted_tokens = np.concatenate(
            [
                data.previous_uncommitted_tokens,
                prompt,
            ]
        )
        if len(uncommitted_tokens) <= 1:
            return prompt, []
        assert self.cache_lengths[seq_id] == len(
            data.committed_blocks
        ) * self.page_size + len(data.previous_uncommitted_tokens)

        # Attempt to match all but the last token in the prompt. This is
        # because the model expects a prompt of length at least 1.
        data.node, prefix_blocks = self.radix_trie.match_prefix(
            uncommitted_tokens[:-1], node=data.node
        )
        # Mark the prefix blocks we retrieved from the radix trie cache as
        # in use by this sequence so they don't get evicted prematurely.
        assert data.node is not None
        self.radix_trie.mark_in_use_by(data.node, seq_id)

        # Update the cache hit rate metrics.
        num_cache_hit_tokens = len(prefix_blocks) * self.page_size
        self.ce_cache_hit_tokens += num_cache_hit_tokens
        self.ce_all_tokens += len(uncommitted_tokens) - 1

        # Add the prefix blocks to the request's cached blocks.
        data.committed_blocks.extend(prefix_blocks)
        self.cache_lengths[seq_id] += num_cache_hit_tokens

        # Shorten the previously uncommitted tokens if we got cache hits
        still_uncommitted_tokens = uncommitted_tokens[
            len(prefix_blocks) * self.page_size :
        ]
        num_prev_uncommitted_left = max(
            len(data.previous_uncommitted_tokens) - num_cache_hit_tokens,
            0,
        )
        data.previous_uncommitted_tokens = data.previous_uncommitted_tokens[
            :num_prev_uncommitted_left
        ]
        if data.inflight_blocks and len(data.previous_uncommitted_tokens) == 0:
            assert len(data.inflight_blocks) == 1
            partially_filled = data.inflight_blocks[0]
            self.release_block(partially_filled, is_committed=False)
            data.inflight_blocks.clear()

        # Shorten the prompt in place if we got cache hits
        prompt = still_uncommitted_tokens[-len(prompt) :]

        # Update the cache length to reflect the new cache hits.
        self.cache_lengths[seq_id] = len(
            data.committed_blocks
        ) * self.page_size + len(data.previous_uncommitted_tokens)

        return prompt, prefix_blocks

    def _fetch(
        self, seq_ids_and_prompts: dict[int, np.ndarray], num_steps: int = 1
    ) -> Sequence[tuple[Tensor, ...]]:
        """This method identifies available blocks to service the given requests and marks them as inflight.
        They're assigned to the request as "in-flight" until step is called.

        Generally the prompt length is n for prefill, and 1 for decode step. Additionally, there is not a
        kv entry associated with each token in the prompt.

        When prefix caching is enabled, and KV entries can be retrieved for some tokens in the prompt, the
        input `seq_ids_and_prompts` will be modified. Each prompt will be shortened to only include the tokens
        for which we do not have a cached KV entry. Note that we will never return a empty prompt.
        """

        batch_size = len(seq_ids_and_prompts)

        max_seq_len_in_batch = -1
        # before we start making any changes, validate that we won't over-write the cache
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            curr_seq_len = (
                self.cache_lengths[seq_id] + len(prompt) + num_steps - 1
            )
            if curr_seq_len > max_seq_len_in_batch:
                max_seq_len_in_batch = curr_seq_len

            assert curr_seq_len <= self.max_seq_len, (
                f"seq_id: {seq_id} would overrun the max cache length of {self.max_seq_len} "
                f"with {len(prompt)} new tokens. Existing length: {self.cache_lengths[seq_id]}"
            )

        max_num_pages = ceildiv(max_seq_len_in_batch, self.page_size)

        # Allocate the buffers containing metadata about the batch.
        # [0, total_num_pages) are the valid block ids and total_num_pages
        # denotes an unassigned block.
        lut_table_np = np.full(
            (batch_size, max_num_pages), self.total_num_pages, dtype=np.uint32
        )
        cache_lengths_np = np.zeros((batch_size,), dtype=np.uint32)

        max_seq_length = 0
        max_cache_length = 0

        all_cache_hit_blocks: set[int] = set()

        # Iterate over requests and query prefix cache, marking cached pages
        # as in use so they don't get evicted when we start allocating pages.
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            # Ensure we've called claim for this sequence id.
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            # Validate there aren't other inflight requests for this sequence.
            assert seq_id not in self.fetch_metadata

            # There can at most be one partially filled inflight block.
            data = self.active_requests[seq_id]
            if len(data.inflight_blocks) > 1:
                # TODO we need a way to invalidate "in-flight" blocks if something goes wrong during execution.
                # probably via a ``release_failed`` method.
                raise ValueError(
                    f"seq_id: {seq_id} already has {len(data.inflight_blocks)} inflight blocks."
                )

            if self.radix_trie is not None:
                trimmed_prompt, prefix_blocks = self._fetch_prefix_caching(
                    seq_id, data, prompt
                )
                seq_ids_and_prompts[seq_id] = trimmed_prompt
                all_cache_hit_blocks.update(prefix_blocks)

        # Determine the number of pages required for each sequence.
        total_sequence_length = 0
        total_blocks_to_allocate = 0
        blocks_to_allocate_by_seq = {}
        for batch_idx, (seq_id, prompt) in enumerate(
            seq_ids_and_prompts.items()
        ):
            data = self.active_requests[seq_id]

            # Get the existing cache length for this sequence.
            cache_length = self.cache_lengths[seq_id]
            cache_lengths_np[batch_idx] = cache_length

            # Update the maximum lengths seen so far.
            max_seq_length = max(max_seq_length, len(prompt))
            max_cache_length = max(max_cache_length, cache_length)

            # Compute the total sequence length and the number of pages required to store it.
            sequence_length = cache_length + len(prompt) + num_steps - 1
            total_sequence_length += sequence_length
            num_pages_required = ceildiv(sequence_length, self.page_size)

            # Compute the number of *new* pages we need to allocate.
            assert len(data.inflight_blocks) <= 1
            num_new_pages = (
                num_pages_required
                - len(data.committed_blocks)
                - len(data.inflight_blocks)
            )
            blocks_to_allocate_by_seq[seq_id] = num_new_pages
            total_blocks_to_allocate += num_new_pages

        # Check if we have enough free blocks to service all requests.
        num_evictable_blocks = 0
        if self.radix_trie is not None:
            # the blocks in the prefix cache that will be used by sequences in
            # this batch are no longer eligible for eviction / allocation.
            num_evictable_blocks = len(
                self.radix_trie.get_evictable_blocks() - all_cache_hit_blocks
            )
        num_free_blocks = len(self.available_blocks) + num_evictable_blocks
        if total_blocks_to_allocate > num_free_blocks:
            raise RuntimeError(
                f"Not enough free blocks to service all {len(seq_ids_and_prompts)} requests.\n"
                f"Need an additional {total_blocks_to_allocate} blocks to store KV projections for all {total_sequence_length} tokens.\n"
                f"But only {num_free_blocks} out of {self.total_num_pages} cache blocks are available to be allocated.\n"
                f"You must restart your process and set a smaller batch size or max sequence length.\n"
            )

        # Allocate additional pages for each request in the batch
        for batch_idx, (seq_id, num_new_pages) in enumerate(
            blocks_to_allocate_by_seq.items()
        ):
            data = self.active_requests[seq_id]

            # Assign some new pages to this request.
            for _ in range(num_new_pages):
                next_block = self.alloc_block()
                data.inflight_blocks.append(next_block)

            # Populate the lookup table with the new pages.
            for i, block_idx in enumerate(data.all_assigned_blocks):
                lut_table_np[batch_idx, i] = block_idx

        # Build a tensor of maximum lengths. Each step slices the first row to
        # advance to the values for the next row.
        max_lengths_host = build_max_lengths_tensor(
            num_steps, max_seq_length, max_cache_length
        )

        lut_table_host = Tensor.from_numpy(lut_table_np)
        cache_lengths_host = Tensor.from_numpy(cache_lengths_np)

        ret_list = []
        for i, device in enumerate(self.devices):
            ret_list.append(
                (
                    self.blocks[i],
                    cache_lengths_host.to(device=device),
                    lut_table_host.to(device=device),
                    max_lengths_host,
                )
            )

        return ret_list

    def input_symbols(
        self,
    ) -> list[PagedCacheInputSymbols]:
        return [
            PagedCacheInputSymbols(
                kv_blocks=TensorType(
                    self.params.dtype,
                    shape=self.block_shape(is_parameterized=True),
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=["batch_size"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=["batch_size", "max_num_pages"],
                    device=DeviceRef(self.devices[i].label, self.devices[i].id),
                ),
                max_lengths=TensorType(
                    DType.uint32, shape=["steps_remaining", 2]
                ),
            )
            for i in range(len(self.devices))
        ]

    def claim(self, n: int) -> list[int]:
        """Claims `n` blocks of memory in the cache for incoming requests.

        This returns a list of sequence ids, which identify a sequence's
        location within the cache. This sequence id can then be passed
        in the fetch function to return the ContinuousBatchingKVCacheCollection
        for those sequences.
        """
        seq_ids = super().claim(n)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = _PagedCacheMetadata()
        return seq_ids

    def external_claim(self, seq_ids: list[int]) -> None:
        """Variant of the above where sequence ids are reserved externally."""
        super().external_claim(seq_ids)
        for seq_id in seq_ids:
            self.active_requests[seq_id] = _PagedCacheMetadata()

    def _count_all_pages(self) -> int:
        available_blocks = self.available_blocks
        prefix_cache_blocks = set()
        if self.radix_trie is not None:
            prefix_cache_blocks = self.radix_trie.get_all_blocks()
        uncommitted_blocks = set()
        for seq_id in self.active_requests:
            uncommitted_blocks.update(
                self.active_requests[seq_id].inflight_blocks
            )
        return len(available_blocks | prefix_cache_blocks | uncommitted_blocks)

    def release(self, seq_id: int) -> None:
        """Release `seq_id` provided, marking this sequence as complete.
        This returns the seq_id back to the available pool of cache memory,
        allowing it to be reused when a new sequence is claimed.
        """
        super().release(seq_id)
        data = self.active_requests[seq_id]

        if self.radix_trie is not None:
            # mark the prefix blocks as not in use by this sequence so they can
            # potentially be evicted when we need more memory
            assert data.node is not None
            self.radix_trie.mark_not_in_use_by(data.node, seq_id)

        for block in data.committed_blocks:
            self.release_block(block, is_committed=True)
        for block in data.inflight_blocks:
            self.release_block(block, is_committed=False)
        del self.active_requests[seq_id]

    def _step_prefix_caching(
        self,
        seq_id: int,
        data: _PagedCacheMetadata,
        prompt: np.ndarray,
        new_tokens: np.ndarray,
    ) -> None:
        """Now that we have written to the inflight blocks, we will try to commit
        them to the radix trie.
        """

        assert self.radix_trie is not None
        assert self.cache_lengths[seq_id] == len(
            data.committed_blocks
        ) * self.page_size + len(data.previous_uncommitted_tokens)

        seq_len = self.cache_lengths[seq_id] + len(prompt) + len(new_tokens) - 1
        num_tokens_in_partially_filled_block = seq_len % self.page_size

        uncommitted_tokens = np.concatenate(
            [
                data.previous_uncommitted_tokens,
                prompt,
                new_tokens[:-1],
            ]
        )
        # Try to match the uncommitted tokens in the trie
        data.node, existing_blocks = self.radix_trie.match_prefix(
            uncommitted_tokens, node=data.node
        )

        # If we computed a kv entry for a token that was already cached,
        # we will just release that block we just computed.
        for b0, b1 in zip(existing_blocks, data.inflight_blocks):
            if b0 != b1:
                self.release_block(b1, is_committed=False)

        # Replace the inflight blocks with the existing prefix blocks.
        data.inflight_blocks[: len(existing_blocks)] = existing_blocks

        # Commit the rest of the tokens in the trie for use by future
        # sequences.
        uncommitted_blocks = data.inflight_blocks[len(existing_blocks) :]
        uncommitted_tokens = uncommitted_tokens[
            len(existing_blocks) * self.page_size :
        ]

        # round the number of uncommitted new tokens to the nearest
        # multiple of the page size if not aligned
        blocks_to_commit = uncommitted_blocks
        tokens_to_commit = uncommitted_tokens
        if num_tokens_in_partially_filled_block > 0:
            prefix, suffix = (
                tokens_to_commit[:-num_tokens_in_partially_filled_block],
                tokens_to_commit[-num_tokens_in_partially_filled_block:],
            )
            tokens_to_commit = prefix
            blocks_to_commit = blocks_to_commit[:-1]
            data.previous_uncommitted_tokens = suffix
        else:
            # Clear out the previous uncommitted tokens
            data.previous_uncommitted_tokens = np.array([], dtype=np.int64)

        assert len(tokens_to_commit) == len(blocks_to_commit) * self.page_size

        # If there are any tokens to commit, insert them into the radix
        # trie.
        data.node = self.radix_trie.insert(
            tokens_to_commit,
            blocks_to_commit,
            node=data.node,
        )

        # Mark the recently committed blocks as in use by this sequence
        # so they don't get evicted prematurely.
        assert data.node is not None
        self.radix_trie.mark_in_use_by(data.node, seq_id)

    def _step(
        self,
        seq_ids_and_new_tokens: dict[int, np.ndarray],
    ) -> None:
        """Update the `cache_lengths` objects to not that a new
        kv projection step has occurred, and that the underlying memory
        has been written to. This `cache_lengths` value is then used
        downstream in `fetch` to track what section of memory should
        be used in the kernels.
        """

        for seq_id, new_tokens in seq_ids_and_new_tokens.items():
            if seq_id not in self.active_requests:
                raise ValueError(f"seq_id: {seq_id} not in active requests.")

            data = self.active_requests[seq_id]
            fetch_metadata = self.fetch_metadata[seq_id]
            prompt = fetch_metadata.prompt
            num_steps = fetch_metadata.num_steps
            assert len(new_tokens) == num_steps

            seq_len = (
                self.cache_lengths[seq_id] + len(prompt) + len(new_tokens) - 1
            )
            num_tokens_in_partially_filled_block = seq_len % self.page_size

            if self.radix_trie is not None:
                self._step_prefix_caching(seq_id, data, prompt, new_tokens)

            expected_num_pages = ceildiv(seq_len, self.page_size)
            actual_num_pages = len(data.inflight_blocks) + len(
                data.committed_blocks
            )

            if expected_num_pages != actual_num_pages:
                raise ValueError(
                    f"Mismatch between expected and actual number of pages for seq_id: {seq_id}. Expected: {expected_num_pages}, Actual: {actual_num_pages}  "
                )

            if num_tokens_in_partially_filled_block > 0:
                # Leave one partially filled block in the inflight blocks
                # and finish committing the rest.
                partially_filled_block = data.inflight_blocks[-1]
                data.committed_blocks.extend(data.inflight_blocks[:-1])
                # This mutates the list in place.
                data.inflight_blocks.clear()
                data.inflight_blocks.append(partially_filled_block)
            else:
                # Commit all of the inflight blocks.
                data.committed_blocks.extend(data.inflight_blocks)
                data.inflight_blocks.clear()
