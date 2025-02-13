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
"""Build a Llama3 model that uses continuous or paged kv-caching"""

from __future__ import annotations

from typing import Optional

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationEncoding
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from max.pipelines.nn import (
    AttentionWithRopeV2,
    EmbeddingV2,
    OptimizedRotaryEmbedding,
    RMSNormV2,
    Transformer,
    TransformerBlock,
    linear_class,
)

from .naive_llama3 import Llama3FeedForward


class Llama3(Transformer):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_hidden_layers: int,
        rope_theta: float,
        max_seq_len: int,
        rms_norm_eps: float,
        intermediate_size: int,
        interleaved_rope_weights: bool,
        rope_scaling: Optional[np.ndarray],
        vocab_size: int,
        dtype: DType,
        quantization_encoding: QuantizationEncoding,
        kv_params: KVCacheParams,
        all_logits: bool,
    ):
        rope = OptimizedRotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        linear_cls = linear_class(quantization_encoding)
        layers = [
            TransformerBlock(
                attention=AttentionWithRopeV2(
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    hidden_size=hidden_size,
                    kv_params=kv_params,
                    layer_idx=i,
                    dtype=dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                ),
                mlp=Llama3FeedForward(
                    dtype,
                    quantization_encoding,
                    hidden_size,
                    intermediate_size,
                    linear_cls,
                ),
                attention_norm=RMSNormV2(
                    hidden_size,
                    rms_norm_eps,
                ),
                mlp_norm=RMSNormV2(
                    hidden_size,
                    rms_norm_eps,
                ),
            )
            for i in range(num_hidden_layers)
        ]

        embedding_layer = EmbeddingV2(
            vocab_size,
            hidden_size,
            dtype,
            DeviceRef.CPU(),
            quantization_encoding=quantization_encoding,
        )

        output = linear_cls(
            vocab_size,
            hidden_size,
            dtype,
            DeviceRef.CPU(),
            quantization_encoding=quantization_encoding,
        )

        kv_collection_cls: (
            type[FetchContinuousBatchingKVCacheCollection]
            | type[FetchPagedKVCacheCollection]
        )
        if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
            kv_collection_cls = FetchContinuousBatchingKVCacheCollection
        elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                "Unsupported caching strategy " + str(kv_params.cache_strategy)
            )

        super().__init__(
            dim=hidden_size,
            n_heads=num_attention_heads,
            layers=layers,
            norm=RMSNormV2(
                hidden_size,
                rms_norm_eps,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=kv_collection_cls(kv_params),
            all_logits=all_logits,
        )
