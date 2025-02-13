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
"""General interface for Attention."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)

from ..layer import Layer, LayerV2
from ..linear import Linear, LinearV2


@dataclass
class AttentionImpl(Layer, ABC):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:

    - AttentionWithRope
    - AttentionWithAlibi
    - VanillaAttentionWithCausalMask
    - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    .. code-block:: python

        @dataclass
        class AttentionWithRope(AttentionImpl):
            rope: OptimizedRotaryEmbedding
            ...

    We expect the ``__call__`` abstractmethod to remain relatively consistent, however the ``**kwargs``
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    .. code-block:: python

        @dataclass
        class VanillaAttentionWithCausalMask(AttentionImpl):
            ...

            def __call__(
                self,
                x: TensorValueLike,
                kv_collection: ContinuousBatchingKVCacheCollection,
                valid_lengths: TensorValueLike,
                **kwargs,
            ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

                if "attn_mask" not in kwargs:
                    raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

                # Which we can then use the attention mask downstream like so:
                op(
                    attn_mask = kwargs["attn_mask"]
                )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    layer_idx: TensorValue
    """The layer number associated with this Attention block."""

    wqkv: TensorValue
    """The concatenation of q, k, and v weight vectors."""

    wo: Linear
    """A linear layer for the output projection."""

    def __post_init__(self) -> None:
        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue: ...


class AttentionImplV2(LayerV2, ABC):
    """A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a separate subclass, articulating each variation of Attention:

    - AttentionWithRope
    - AttentionWithAlibi
    - VanillaAttentionWithCausalMask
    - ...

    `AttentionImplV2` will replace `AttentionImpl` as we roll out changes to the
    Layer API.

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    .. code-block:: python

        @dataclass
        class AttentionWithRope(AttentionImplV2):
            rope: OptimizedRotaryEmbedding
            ...

    We expect the ``__call__`` abstractmethod to remain relatively consistent, however the ``**kwargs``
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    .. code-block:: python

        class VanillaAttentionWithCausalMask(AttentionImplV2):
            ...

            def __call__(
                self,
                x: TensorValueLike,
                kv_collection: ContinuousBatchingKVCacheCollection,
                **kwargs,
            ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

                if "attn_mask" not in kwargs:
                    raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

                # Which we can then use the attention mask downstream like so:
                op(
                    attn_mask = kwargs["attn_mask"]
                )
    """

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.CPU(),
        linear_cls: type[LinearV2] = LinearV2,
    ):
        """Initializes the attention layer.

        Args:
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the
            device: Device to place the weights and run the computation.
            linear_cls: Linear class to use for the outputs dense layer.
        """
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        self.wq = Weight(
            name="wq", dtype=dtype, shape=[hidden_size, hidden_size]
        )
        self.wk = Weight(
            name="wk", dtype=dtype, shape=[kv_weight_dim, hidden_size]
        )
        self.wv = Weight(
            name="wv", dtype=dtype, shape=[kv_weight_dim, hidden_size]
        )
        self.wo = linear_cls(
            in_dim=hidden_size, out_dim=hidden_size, dtype=dtype, device=device
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        return ops.concat((self.wq, self.wk, self.wv))

    @abstractmethod
    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue: ...


@dataclass
class DistributedAttentionImpl(Layer, ABC):
    """
    A generalized Distributed attention interface.
    """

    @abstractmethod
    def __call__(
        self,
        x: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        **kwargs,
    ) -> list[TensorValue]: ...


@dataclass
class AttentionImplQKV(Layer, ABC):
    """
    A generalized attention interface, that will be used upstream by a general Transformer.
    We would expect a seperate subclass, articulating each variation of Attention:

    - AttentionWithRope
    - AttentionWithAlibi
    - VanillaAttentionWithCausalMask
    - ...

    There are a series of shared attributes, however, more may be needed for each individual variant.
    For example, we may introduce an OptimizedRotaryEmbedding class for the AttentionWithRope class:

    .. code-block:: python

        @dataclass
        class AttentionWithRope(AttentionImpl):
            rope: OptimizedRotaryEmbedding
            ...

    We expect the ``__call__`` abstractmethod to remain relatively consistent, however the ``**kwargs``
    argument is exposed, allowing you to leverage additional arguments for each particular variant.
    For example, we may introduce an VanillaAttentionWithCausalMask class, which includes an attention
    mask:

    .. code-block:: python

        @dataclass
        class VanillaAttentionWithCausalMask(AttentionImpl):
            ...

            def __call__(
                self,
                x: TensorValueLike,
                kv_collection: ContinuousBatchingKVCacheCollection,
                valid_lengths: TensorValueLike,
                **kwargs,
            ) -> tuple[TensorValue, ContinuousBatchingKVCacheCollection]: ...

                if "attn_mask" not in kwargs:
                    raise ValueError("attn_mask not provided to VanillaAttentionWithCausalMask")

                # Which we can then use the attention mask downstream like so:
                op(
                    attn_mask = kwargs["attn_mask"]
                )
    """

    n_heads: int
    """The number of attention heads."""

    kv_params: KVCacheParams
    """KV Cache Params, including the number of kv heads, the head dim, and data type."""

    layer_idx: int
    """The layer number associated with this Attention block."""

    wq: TensorValueLike
    """The q weight vector."""

    wk: TensorValueLike
    """The k weight vector."""

    wv: TensorValueLike
    """The v weight vector."""

    wo: Linear
    """A linear layer for the output projection."""

    def __post_init__(self) -> None:
        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

    @abstractmethod
    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue: ...
