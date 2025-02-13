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
"""Builds a Llama3 model that uses naive KV-caching."""

from typing import Optional, Union

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.quantization import QuantizationEncoding
from max.pipelines.kv_cache import KVCacheParams
from max.pipelines.nn import (
    MLPV2,
    EmbeddingV2,
    LinearV2,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    OptimizedRotaryEmbedding,
    RMSNormV2,
    RotaryEmbedding,
    linear_class,
)


class NaiveLlama3(NaiveTransformer):
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
    ):
        rope = RotaryEmbedding(
            dim=hidden_size,
            n_heads=num_attention_heads,
            theta=rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        linear_cls = linear_class(quantization_encoding)
        layers = [
            NaiveTransformerBlock(
                attention=NaiveLLama3Attention(
                    kv_params,
                    hidden_size,
                    num_attention_heads,
                    num_key_value_heads,
                    rope,
                    dtype,
                    quantization_encoding,
                    linear_cls,
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

        super().__init__(
            dim=hidden_size,
            n_heads=num_attention_heads,
            layers=layers,
            norm=RMSNormV2(
                hidden_size,
                rms_norm_eps,
            ),
            output=output,
            theta=rope_theta,
            embedding=embedding_layer,
        )


class NaiveLLama3Attention(NaiveAttentionWithRope):
    def __init__(
        self,
        kv_params: KVCacheParams,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope: Union[OptimizedRotaryEmbedding, RotaryEmbedding],
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        linear_cls: type[LinearV2],
    ):
        kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        super().__init__(
            n_heads=num_attention_heads,
            kv_params=kv_params,
            dim=hidden_size,
            wk=linear_cls(
                in_dim=kv_weight_dim,
                out_dim=hidden_size,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
            wv=linear_cls(
                in_dim=kv_weight_dim,
                out_dim=hidden_size,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
            wq=linear_cls(
                in_dim=hidden_size,
                out_dim=hidden_size,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
            wo=linear_cls(
                in_dim=hidden_size,
                out_dim=hidden_size,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
            rope=rope,
        )


class Llama3FeedForward(MLPV2):
    def __init__(
        self,
        dtype: DType,
        quantization_encoding: Optional[QuantizationEncoding],
        hidden_dim: int,
        feed_forward_length: int,
        linear_cls: type[LinearV2],
    ):
        super().__init__(
            gate_proj=linear_cls(
                in_dim=feed_forward_length,
                out_dim=hidden_dim,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
            down_proj=linear_cls(
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
            up_proj=linear_cls(
                in_dim=feed_forward_length,
                out_dim=hidden_dim,
                dtype=dtype,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            ),
        )
