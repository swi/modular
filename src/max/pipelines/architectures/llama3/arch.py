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

from max.pipelines import (
    PipelineTask,
    RopeType,
    SupportedArchitecture,
    SupportedEncoding,
    TextTokenizer,
    WeightsFormat,
)
from max.pipelines.kv_cache import KVCacheStrategy

from . import weight_adapters
from .model import Llama3Model, OlmoModel, Phi3Model

llama_arch = SupportedArchitecture(
    name="LlamaForCausalLM",
    example_repo_ids=[
        "meta-llama/Llama-3.1-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "meta-llama/Llama-Guard-3-8B",
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "deepseek-ai/deepseek-coder-6.7b-instruct",
        "modularai/llama-3.1",
    ],
    default_encoding=SupportedEncoding.q4_k,
    supported_encodings={
        SupportedEncoding.gptq: [
            KVCacheStrategy.PAGED,
        ],
        SupportedEncoding.q4_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q4_0: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q6_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
)

exaone_arch = SupportedArchitecture(
    name="ExaoneForCausalLM",
    default_encoding=SupportedEncoding.float32,
    task=PipelineTask.TEXT_GENERATION,
    supported_encodings={
        SupportedEncoding.q4_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.q6_k: [KVCacheStrategy.NAIVE],
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    example_repo_ids=[
        "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
    ],
    pipeline_model=Llama3Model,
    tokenizer=TextTokenizer,
    rope_type=RopeType.neox,
    default_weights_format=WeightsFormat.gguf,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_exaone_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
)


phi3_arch = SupportedArchitecture(
    name="Phi3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["microsoft/phi-4", "microsoft/Phi-3.5-mini-instruct"],
    default_weights_format=WeightsFormat.gguf,
    default_encoding=SupportedEncoding.bfloat16,
    supported_encodings={
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            # KVCacheStrategy.NAIVE,  # TODO(kathywu): Support naive caching for phi models
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            # KVCacheStrategy.NAIVE,  # TODO(kathywu): Support naive caching for phi models
        ],
    },
    pipeline_model=Phi3Model,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
)


olmoe_arch = SupportedArchitecture(
    name="OlmoForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["allenai/OLMo-1B-hf", "allenai/OLMo-1B-0724-hf"],
    default_weights_format=WeightsFormat.gguf,
    default_encoding=SupportedEncoding.float32,
    supported_encodings={
        SupportedEncoding.float32: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
        SupportedEncoding.bfloat16: [
            KVCacheStrategy.PAGED,
            KVCacheStrategy.CONTINUOUS,
            KVCacheStrategy.NAIVE,
        ],
    },
    pipeline_model=OlmoModel,
    tokenizer=TextTokenizer,
    rope_type=RopeType.normal,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
)
