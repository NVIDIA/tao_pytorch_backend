"""NVDINOv2 legacy model."""
from functools import partial

from torch import nn
from timm.models.vision_transformer import (
    VisionTransformer,
    build_model_with_cfg,
    checkpoint_filter_fn,
)
from timm.layers import (PatchEmbed, SwiGLUPacked)
from nvidia_tao_pytorch.cv.classification_pyt.model.backbones.dinov2_vit import DinoV2ViT


def vit_large_patch14_dinov2_swiglu_legacy(**kwargs) -> VisionTransformer:
    """ViT-L/14 for DINOv2"""
    model_args = {
        'patch_size': 14,
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'init_values': 1e-5,
        'mlp_layer': SwiGLUPacked,
        'act_layer': nn.SiLU,
        'mlp_ratio': 5472 / 1024,
        'embed_layer': partial(PatchEmbed, strict_img_size=False),
        'num_classes': 0,
    }

    model = build_model_with_cfg(
        DinoV2ViT,
        "vit_large_patch14_dinov2_swiglu_legacy",
        pretrained=False,
        pretrained_filter_fn=checkpoint_filter_fn,
        **dict(model_args, **kwargs),
    )

    return model
