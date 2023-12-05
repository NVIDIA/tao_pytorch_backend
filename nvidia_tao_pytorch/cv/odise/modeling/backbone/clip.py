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

"""ODISE ConvNext based CLIP backbone."""
import open_clip
import torch
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.utils import comm


@BACKBONE_REGISTRY.register()
class CLIP(Backbone):
    def __init__(self, model_name, pretrained="laion2b_s29b_b131k_ft_soup", precision='fp32'):
        super().__init__()
        self.precision = precision
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                device='cuda',
                precision=precision)
        comm.synchronize()

        # checked, the same as openai original CLIP
        self.clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device='cuda',
            precision=precision)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.clip_normalize = preprocess.transforms[-1]

        model_name = model_name.lower()
        self.model_name = model_name
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]
        else:
            raise NotImplementedError

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            "clip_embedding": -1
        }
        self._out_feature_channels = {
            "stem": self.output_channels[0],
            "res2": self.output_channels[1],
            "res3": self.output_channels[2],
            "res4": self.output_channels[3],
            "res5": self.output_channels[4],
            "clip_embedding": self.dim_latent
        }

        self.eval()
        self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def encode_text(self, text, normalize: bool=False):
        cast_dtype = self.clip_model.transformer.get_cast_dtype()

        x = self.clip_model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x, attn_mask=self.clip_model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def tokenize_text(self, text):
        return self.text_tokenizer(text)

    def extract_features(self, x):
        return {
            'convnext': self.extract_features_convnext,
        }[self.model_type](x)
    
    def visual_prediction_forward(self, x, masks=None):
        return {
            'convnext': self.visual_prediction_forward_convnext,
        }[self.model_type](x, masks)

    # @torch.compile
    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)

        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out

    def visual_prediction_forward_convnext(self, x, mask):
        batch, num_query, channel = x.shape
        x = x.reshape(batch*num_query, channel, 1, 1) # fake 2D input (torch.Size([100, 1536, 1, 1]))
        x = self.clip_model.visual.trunk.head(x) # torch.Size([100, 1536])
        x = self.clip_model.visual.head(x) # torch.Size([100, 768])
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640 (torch.Size([1, 100, 768]))

    def get_text_classifier(self, text_list, device):
        self.eval()
        with torch.no_grad():
            # reference for templates: https://github.com/mlfoundations/open_clip/blob/91f6cce16b7bee90b3b5d38ca305b5b3b67cc200/src/training/imagenet_zeroshot_data.py
            text_tokens = self.tokenize_text(text_list)
            text_tokens = text_tokens.to(device)
            # we return un-normalized text feature.
            text_features = self.encode_text(text_tokens, normalize=False)
            return text_features

    @property
    def size_divisibility(self) -> int:
        return -1

    @property
    def dim_latent(self):
        return self.clip_model.text_projection.shape[-1]

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in ["stem", "res2", "res3", "res4", "res5", "clip_embedding"]
        }

    def forward(self, x):
        # self.eval()
        with torch.no_grad():
            return self.extract_features(x)
