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

"""Mixture of Experts Gate Layer."""

from dataclasses import dataclass

from max.dtype import DType
from max.graph import TensorValue, Value, ops
from max.pipelines.nn import Linear
from max.pipelines.nn.layer import LayerV2


@dataclass
class MaxMoEGate(LayerV2):
    """Mixture of Experts Gate Layer.

    Args:
        gate_score: Linear layer that projects from hidden_size to intermediate_size.
    Input shape: (batch_size, seq_length, hidden_size)
    Output shapes: (num_experts_per_tok), num_experts_per_tok)
    """

    def __post_init__(self):
        super().__init__()

    gate_score: Linear
    num_experts_per_tok: int = 6
    n_routed_experts: int = 64
    routed_scaling_factor: float = 1.0
    aux_loss_alpha: float = 0.001
    n_group: int = 1
    topk_group: int = 1
    gating_dim: int = 2048  # equal to config.hidden_size

    def __call__(self, hidden_states: TensorValue) -> tuple[Value, Value]:
        # compute gating score
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.reshape([bsz * seq_len, h])

        logits = self.gate_score(hidden_states.cast(DType.float32))
        scores = ops.softmax(logits[-1].cast(DType.float32))

        # select top k experts
        topk_weight, topk_idx = ops.top_k(
            scores, self.num_experts_per_tok, -1, False
        )

        return topk_idx, topk_weight
