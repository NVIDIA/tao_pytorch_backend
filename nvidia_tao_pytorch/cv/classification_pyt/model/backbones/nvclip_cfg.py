# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classification CLIP backbone cfg"""

map_clip_model_cfg = {
    "ViT-H-14-SigLIP-CLIPA-224": {
        "embed_dim": 1024,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 32,
            "width": 1280,
            "head_width": 80,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 1024,
            "heads": 16,
            "layers": 24,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-L-14-SigLIP-CLIPA-336": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 336,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 768,
            "heads": 12,
            "layers": 12,
            "pool_type": "last",
            "no_causal_mask": True
        }
    },
    "ViT-L-14-SigLIP-CLIPA-224": {
        "embed_dim": 768,
        "init_logit_bias": -10,
        "vision_cfg": {
            "image_size": 224,
            "layers": 24,
            "width": 1024,
            "head_width": 64,
            "patch_size": 14,
            "no_ln_pre": True,
            "pool_type": "avg",
            "final_ln_after_pool": True,
            "pos_embed_type": "sin_cos_2d",
            "patch_dropout": 0.0
        },
        "text_cfg": {
            "context_length": 77,
            "vocab_size": 32000,
            "hf_tokenizer_name": "bert-base-uncased",
            "tokenizer_kwargs": {
                "strip_sep_token": True
            },
            "width": 768,
            "heads": 12,
            "layers": 12,
            "pool_type": "last",
            "no_causal_mask": True
        }
    }
}
