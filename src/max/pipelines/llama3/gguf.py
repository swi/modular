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
"""Build a Llama3 model via Graph API from GGUF weights."""

from __future__ import annotations

from typing import List, Optional, Union, cast

import numpy as np
from max.dtype import DType
from max.graph import (
    DeviceRef,
    Graph,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.graph.weights import SafetensorWeights, Weights
from max.pipelines import PipelineConfig, RopeType, WeightsFormat
from max.pipelines.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    KVCacheStrategy,
)
from max.pipelines.nn import (
    MLP,
    AttentionWithRope,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    DistributedTransformer,
    DistributedTransformerBlock,
    Embedding,
    EmbeddingV2,
    Linear,
    LinearV2,
    NaiveAttentionWithRope,
    NaiveTransformer,
    NaiveTransformerBlock,
    OptimizedRotaryEmbedding,
    RMSNorm,
    RotaryEmbedding,
    Transformer,
    TransformerBlock,
)


def distribute_value(
    v: TensorValue, devices: List[DeviceRef]
) -> List[TensorValue]:
    return [v.to(device) for device in devices]


def shard_col_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    col_size = int(v.shape[1]) // n_devices
    return [
        v[:, i * col_size : (i + 1) * col_size].to(device)
        for i, device in enumerate(devices)
    ]


def shard_row_value(
    x: TensorValueLike, devices: List[DeviceRef]
) -> List[TensorValue]:
    n_devices = len(devices)
    v = TensorValue(x)
    row_size = int(v.shape[0]) // n_devices
    return [
        v[i * row_size : (i + 1) * row_size, :].to(device)
        for i, device in enumerate(devices)
    ]


def feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
    quantization_config: Optional[QuantizationConfig],
) -> MLP:
    return MLP(
        Linear.create(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.ffn_gate,
            quantization_config=quantization_config,
        ),
        Linear.create(
            dtype,
            quantization_encoding,
            hidden_dim,
            feed_forward_length,
            weights.ffn_down,
            quantization_config=quantization_config,
        ),
        Linear.create(
            dtype,
            quantization_encoding,
            feed_forward_length,
            hidden_dim,
            weights.ffn_up,
            quantization_config=quantization_config,
        ),
    )


def rms_norm(dims: int, eps: float, weights: Weights) -> RMSNorm:
    return RMSNorm(weights.weight.allocate(DType.float32, [dims]), eps)


def distributed_rms_norm(
    dims: int,
    eps: float,
    weights: Weights,
    devices: List[DeviceRef],
) -> DistributedRMSNorm:
    weights_ = TensorValue(weights.weight.allocate(DType.float32, [dims]))
    weights_devs = distribute_value(weights_, devices)

    rms_norms = [RMSNorm(weights_dev, eps) for weights_dev in weights_devs]

    return DistributedRMSNorm(rms_norms, devices)


def embedding(
    pipeline_config: PipelineConfig,
    vocab_size: int,
    hidden_dim: int,
    weights: Weights,
) -> Embedding:
    if pipeline_config.quantization_encoding == "gptq":
        return Embedding(
            weights.weight.allocate(
                DType.bfloat16,
                [vocab_size, hidden_dim],
                None,
            )
        )

    else:
        return Embedding(
            weights.weight.allocate(
                pipeline_config.dtype,
                [vocab_size, hidden_dim],
                pipeline_config.graph_quantization_encoding,
            )
        )


def distributed_feed_forward(
    dtype: DType,
    quantization_encoding: Optional[QuantizationEncoding],
    hidden_dim: int,
    feed_forward_length: int,
    weights: Weights,
    devices: List[DeviceRef],
) -> DistributedMLP:
    w_ffn_down_full = weights.ffn_down.weight.allocate(
        dtype, [hidden_dim, feed_forward_length], quantization_encoding
    )
    ffn_down_sharded = shard_col_value(w_ffn_down_full, devices)
    w_ffn_gate_full = weights.ffn_gate.weight.allocate(
        dtype, [feed_forward_length, hidden_dim], quantization_encoding
    )
    ffn_gate_sharded = shard_row_value(w_ffn_gate_full, devices)
    w_ffn_up_full = weights.ffn_up.weight.allocate(
        dtype, [feed_forward_length, hidden_dim], quantization_encoding
    )
    ffn_up_sharded = shard_row_value(w_ffn_up_full, devices)

    mlps = [
        MLP(
            Linear(ffn_gate_sharded[rank]),
            Linear(ffn_down_sharded[rank]),
            Linear(ffn_up_sharded[rank]),
        )
        for rank in range(len(devices))
    ]

    return DistributedMLP(mlps, len(devices))


def distributed_attention_opaque(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    layer_idx: TensorValue,
    devices: List[DeviceRef],
) -> DistributedAttentionWithRope:
    wq_full = ops.transpose(
        weights.attn_q.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
            pipeline_config.graph_quantization_encoding,
        ),
        0,
        1,
    )
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    wk_full = ops.transpose(
        weights.attn_k.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.graph_quantization_encoding,
        ),
        0,
        1,
    )
    wv_full = ops.transpose(
        weights.attn_v.weight.allocate(
            pipeline_config.dtype,
            [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
            pipeline_config.graph_quantization_encoding,
        ),
        0,
        1,
    )

    wo_full = weights.attn_output.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
        ],
        pipeline_config.graph_quantization_encoding,
    )
    wq_shards = shard_col_value(wq_full, devices)
    wk_shards = shard_col_value(wk_full, devices)
    wv_shards = shard_col_value(wv_full, devices)

    # Didn't transpose here since linear will transpose so shard on col instead
    # of row
    wo_shards = shard_col_value(wo_full, devices)
    attns = [
        AttentionWithRope(
            n_heads=pipeline_config.huggingface_config.num_attention_heads
            // len(devices),
            kv_params=kv_params,
            wqkv=ops.concat(
                (wq_shards[rank], wk_shards[rank], wv_shards[rank]), axis=1
            ).transpose(0, 1),
            wo=Linear(wo_shards[rank]),
            rope=rope,
            layer_idx=layer_idx,
        )
        for rank in range(len(devices))
    ]

    return DistributedAttentionWithRope(attns, devices)


def _attention_opaque(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: Weights,
    layer_idx: TensorValue,
) -> AttentionWithRope:
    if pipeline_config.quantization_encoding == "gptq":
        assert isinstance(weights, SafetensorWeights), (
            "Only support loading GPTQ weights from safetensor."
        )
        return quantized_attention_opaque(
            kv_params,
            pipeline_config,
            rope,
            weights,
            layer_idx,
        )
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads

    wq = weights.attn_q.weight.allocate(
        pipeline_config.dtype,
        [
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
        ],
        pipeline_config.graph_quantization_encoding,
    )
    wk = weights.attn_k.weight.allocate(
        pipeline_config.dtype,
        [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
        pipeline_config.graph_quantization_encoding,
    )
    wv = weights.attn_v.weight.allocate(
        pipeline_config.dtype,
        [kv_weight_dim, pipeline_config.huggingface_config.hidden_size],
        pipeline_config.graph_quantization_encoding,
    )

    wqkv = ops.concat((wq, wk, wv))

    return AttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=Linear.create(
            pipeline_config.dtype,
            pipeline_config.graph_quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_output,
            quantization_config=pipeline_config._quant_config,
        ),
        rope=rope,
        layer_idx=layer_idx,
    )


def _kv_collection_constructor(
    kv_params: KVCacheParams,
) -> FetchContinuousBatchingKVCacheCollection | FetchPagedKVCacheCollection:
    """Gets the fetch KV collection based on the KV cache strategy.

    Returns:
        Callable that stages an op to fetch a KV cache collection.

    Raises:
        ValueError: If the cache strategy is unsupported.
    """
    if kv_params.cache_strategy == KVCacheStrategy.CONTINUOUS:
        return FetchContinuousBatchingKVCacheCollection(kv_params)
    elif kv_params.cache_strategy == KVCacheStrategy.PAGED:
        return FetchPagedKVCacheCollection(kv_params)

    msg = f"Unsupported caching strategy {kv_params.cache_strategy}"
    raise ValueError(msg)


def quantized_attention_opaque(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: OptimizedRotaryEmbedding,
    weights: SafetensorWeights,
    layer_idx: TensorValue,
) -> AttentionWithRope:
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads

    hidden_size = pipeline_config.huggingface_config.hidden_size

    # fmt: off

    # the `qweight` tensor for a QuantLinear is of type uint32. When allocated as bytes, we reshape the
    # uint8 tensor to [cols, rows * 4] so concatenating the uint8 tensors along axis=1 is equivalent to
    # concatenating the original uint32 tensors along axis=1.
    wq_qweight = ops.reshape(weights.attn_q.qweight.allocate_as_bytes(), (-1, hidden_size * 4))
    wk_qweight = ops.reshape(weights.attn_k.qweight.allocate_as_bytes(), (-1, kv_weight_dim * 4))
    wv_qweight = ops.reshape(weights.attn_v.qweight.allocate_as_bytes(), (-1, kv_weight_dim * 4))

    wqkv_qweight = ops.reshape(
        ops.concat((wq_qweight, wk_qweight, wv_qweight), axis=1),
        (-1, hidden_size + 2 * kv_weight_dim),
    )

    # `scales` tensor is in f16/bf16 type, so we reshape the uint8 tensor to [cols, rows * 2].
    wq_scales = ops.reshape(weights.attn_q.scales.allocate_as_bytes(), (-1, hidden_size * 2))
    wk_scales = ops.reshape(weights.attn_k.scales.allocate_as_bytes(), (-1, kv_weight_dim * 2))
    wv_scales = ops.reshape(weights.attn_v.scales.allocate_as_bytes(), (-1, kv_weight_dim * 2))

    wqkv_scales = ops.reshape(
        ops.concat((wq_scales, wk_scales, wv_scales), axis=1),
        (-1, hidden_size + 2 * kv_weight_dim),
    )
    # fmt: on

    wqkv = ops.concat((wqkv_qweight, wqkv_scales))

    assert pipeline_config._quant_config, (
        "Missing `_quant_config` in `PipelineConfig`"
    )
    quant_config = pipeline_config._quant_config

    perm_idx = None
    if quant_config.desc_act:
        perm_idx = weights.attn_q.g_idx.allocate(
            DType.int32,
            [hidden_size],
        )
        # hack: argsort the perm_idx array
        weights._allocated[perm_idx.name] = np.argsort(
            weights._allocated[perm_idx.name]
        ).astype(np.int32)

    return AttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        wqkv=wqkv,
        wo=Linear.create(
            pipeline_config.dtype,
            pipeline_config.graph_quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_output,
            quantization_config=quant_config,
        ),
        rope=rope,
        layer_idx=layer_idx,
        perm_idx=perm_idx,
        quantization_config=quant_config,
    )


def distributed_transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> DistributedTransformer:
    devices = [
        DeviceRef(spec.device_type, spec.id)
        for spec in pipeline_config.device_specs
    ]
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        layers = [
            DistributedTransformerBlock(
                attention=distributed_attention_opaque(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.blk[i],
                    layer_idx=ops.constant(i, DType.uint32),
                    devices=devices,
                ),
                mlp=distributed_feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.graph_quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                    devices=devices,
                ),
                attention_norm=distributed_rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].attn_norm,
                    devices=devices,
                ),
                mlp_norm=distributed_rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.rms_norm_eps,
                    weights.blk[i].ffn_norm,
                    devices=devices,
                ),
                devices=devices,
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        # TODO(max.nn.Model): We still rely on the `Weights` mechanism for
        # constructing the Python-side weights registry.
        # So we have to "allocate" a spot in the weights registry for
        # output/embedding weights here.
        embedding_weight = weights.token_embd.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
        )
        weights.output.weight.allocate(
            pipeline_config.dtype,
            [
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
            ],
        )

        embedding_layer = EmbeddingV2(
            vocab_size=pipeline_config.huggingface_config.vocab_size,
            hidden_dim=pipeline_config.huggingface_config.hidden_size,
            dtype=pipeline_config.dtype,
            device=devices[0],
            # Use the embedding weight's name, which mismatches between
            # Safetensors and GGUF Llama 3.
            name=embedding_weight.name,
        )

        output = LinearV2(
            in_dim=pipeline_config.huggingface_config.hidden_size,
            out_dim=pipeline_config.huggingface_config.vocab_size,
            dtype=pipeline_config.dtype,
            # Only compute output embedding on device 0 for now.
            # TODO(MODELS-378): More optimal would be to:
            # - Shard embedding table across devices.
            # - Compute output on all devices for multistep.
            device=devices[0],
            # Smaller model variants lack dedicated weights for a final linear
            # layer, and share the embedding layer.
            name=(
                weights.output.name
                if weights.output.weight.exists()
                else embedding_layer.weight.name
            ),
        )

        return DistributedTransformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                pipeline_config.huggingface_config.rms_norm_eps,
                weights.output_norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=_kv_collection_constructor(kv_params),
            devices=devices,
            all_logits=pipeline_config.enable_echo,
        )


def _transformer_opaque(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> Transformer:
    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = OptimizedRotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        if pipeline_config.huggingface_config.model_type == "exaone":
            rms_norm_eps = pipeline_config.huggingface_config.layer_norm_epsilon
        else:
            rms_norm_eps = pipeline_config.huggingface_config.rms_norm_eps

        layers = [
            TransformerBlock(
                attention=_attention_opaque(
                    kv_params,
                    pipeline_config,
                    rope,
                    weights.blk[i],
                    layer_idx=ops.constant(i, DType.uint32),
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.graph_quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                    pipeline_config._quant_config,
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.token_embd,
        )

        # Smaller model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if weights.output.weight.exists():
            output = Linear.create(
                pipeline_config.dtype,
                pipeline_config.graph_quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                weights.output,
                quantization_config=pipeline_config._quant_config,
            )
        else:
            output = Linear.create(
                pipeline_config.dtype,
                pipeline_config.graph_quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                cast(Weight, embedding_layer.weights),
            )

        return Transformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                rms_norm_eps,
                weights.output_norm,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=kv_params,
            kv_collection_constructor=_kv_collection_constructor(kv_params),
            all_logits=pipeline_config.enable_echo,
        )


def attention(
    kv_params: KVCacheParams,
    pipeline_config: PipelineConfig,
    rope: Union[OptimizedRotaryEmbedding, RotaryEmbedding],
    weights: Weights,
) -> NaiveAttentionWithRope:
    kv_weight_dim = (
        pipeline_config.huggingface_config.hidden_size
        // pipeline_config.huggingface_config.num_attention_heads
    ) * pipeline_config.huggingface_config.num_key_value_heads
    return NaiveAttentionWithRope(
        n_heads=pipeline_config.huggingface_config.num_attention_heads,
        kv_params=kv_params,
        dim=pipeline_config.huggingface_config.hidden_size,
        wk=Linear.create(
            pipeline_config.dtype,
            pipeline_config.graph_quantization_encoding,
            kv_weight_dim,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_k,
            quantization_config=pipeline_config._quant_config,
        ),
        wv=Linear.create(
            pipeline_config.dtype,
            pipeline_config.graph_quantization_encoding,
            kv_weight_dim,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_v,
            quantization_config=pipeline_config._quant_config,
        ),
        wq=Linear.create(
            pipeline_config.dtype,
            pipeline_config.graph_quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_q,
            quantization_config=pipeline_config._quant_config,
        ),
        wo=Linear.create(
            pipeline_config.dtype,
            pipeline_config.graph_quantization_encoding,
            pipeline_config.huggingface_config.hidden_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.attn_output,
            quantization_config=pipeline_config._quant_config,
        ),
        rope=rope,
    )


def transformer(
    graph: Graph,
    pipeline_config: PipelineConfig,
    weights: Weights,
    max_seq_len: int,
    kv_params: KVCacheParams,
) -> Union[Transformer, NaiveTransformer]:
    if pipeline_config.cache_strategy.uses_opaque():
        return _transformer_opaque(
            graph=graph,
            pipeline_config=pipeline_config,
            weights=weights,
            max_seq_len=max_seq_len,
            kv_params=kv_params,
        )

    with graph:
        if weights.rope_freqs.weight.exists():
            rope_scaling = weights.rope_freqs.weight.raw_tensor()
        else:
            rope_scaling = None

        interleaved_rope_weights = (
            pipeline_config.weights_format == WeightsFormat.gguf
            and pipeline_config.rope_type == RopeType.normal
        )
        rope = RotaryEmbedding(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            theta=pipeline_config.huggingface_config.rope_theta,
            max_seq_len=max_seq_len,
            rope_scaling=rope_scaling,
            interleaved=interleaved_rope_weights,
        )

        if pipeline_config.huggingface_config.model_type == "exaone":
            rms_norm_eps = pipeline_config.huggingface_config.layer_norm_epsilon
        else:
            rms_norm_eps = pipeline_config.huggingface_config.rms_norm_eps
        layers = [
            NaiveTransformerBlock(
                attention=attention(
                    kv_params, pipeline_config, rope, weights.blk[i]
                ),
                mlp=feed_forward(
                    pipeline_config.dtype,
                    pipeline_config.graph_quantization_encoding,
                    pipeline_config.huggingface_config.hidden_size,
                    pipeline_config.huggingface_config.intermediate_size,
                    weights.blk[i],
                    pipeline_config._quant_config,
                ),
                attention_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
                    weights.blk[i].attn_norm,
                ),
                mlp_norm=rms_norm(
                    pipeline_config.huggingface_config.hidden_size,
                    rms_norm_eps,
                    weights.blk[i].ffn_norm,
                ),
            )
            for i in range(pipeline_config.huggingface_config.num_hidden_layers)
        ]

        embedding_layer = embedding(
            pipeline_config,
            pipeline_config.huggingface_config.vocab_size,
            pipeline_config.huggingface_config.hidden_size,
            weights.token_embd,
        )

        # Smaller model variants lack dedicated weights for a final linear
        # layer, and share the embedding layer.
        if weights.output.weight.exists():
            output = Linear.create(
                pipeline_config.dtype,
                pipeline_config.graph_quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                weights.output,
                quantization_config=pipeline_config._quant_config,
            )
        else:
            output = Linear.create(
                pipeline_config.dtype,
                pipeline_config.graph_quantization_encoding,
                pipeline_config.huggingface_config.vocab_size,
                pipeline_config.huggingface_config.hidden_size,
                cast(Weight, embedding_layer.weights),
            )

        return NaiveTransformer(
            dim=pipeline_config.huggingface_config.hidden_size,
            n_heads=pipeline_config.huggingface_config.num_attention_heads,
            layers=layers,
            norm=rms_norm(
                pipeline_config.huggingface_config.hidden_size,
                rms_norm_eps,
                weights.output_norm,
            ),
            output=output,
            theta=pipeline_config.huggingface_config.rope_theta,
            embedding=embedding_layer,
            output_type=DType.float32,
        )
