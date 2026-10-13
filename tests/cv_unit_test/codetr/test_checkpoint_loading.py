# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for loading TAO and reference Co-DETR checkpoints into CoDETRPlModel."""

import re

import pytest
import torch
from omegaconf import OmegaConf

from nvidia_tao_pytorch.config.codetr.default_config import ExperimentConfig
from nvidia_tao_pytorch.cv.codetr.model.pl_codetr_model import CoDETRPlModel
from nvidia_tao_pytorch.cv.codetr.model.utils import (
    is_reference_codetr_checkpoint,
    load_codetr_checkpoint,
    load_codetr_state_dict,
)

DINO = "model.model.model."
# Decoder heads are shared with the top-level class_embed / bbox_embed modules.
TIED = (DINO + "transformer.decoder.bbox_embed.", DINO + "transformer.decoder.class_embed.")
_ENC_INV = [("self_attn.", "attentions.0."), ("linear1.", "ffns.0.layers.0.0."),
            ("linear2.", "ffns.0.layers.1."), ("norm1.", "norms.0."), ("norm2.", "norms.1.")]
_DEC_INV = [("self_attn.", "attentions.0.attn."), ("cross_attn.", "attentions.1."),
            ("linear1.", "ffns.0.layers.0.0."), ("linear2.", "ffns.0.layers.1."),
            ("norm2.", "norms.0."), ("norm1.", "norms.1."), ("norm3.", "norms.2.")]
_ATSS_INV = {"cls_pred": "atss_cls", "reg_pred": "atss_reg", "centerness_pred": "atss_centerness"}


def _build_model(num_classes=4, num_co_heads=2, num_feature_levels=2):
    cfg = OmegaConf.structured(ExperimentConfig())
    cfg.model.backbone = "resnet_50"
    cfg.model.num_feature_levels = num_feature_levels
    cfg.model.return_interm_indices = [1, 2, 3][:num_feature_levels]
    cfg.model.num_queries = 100
    cfg.model.num_co_heads = num_co_heads
    cfg.model.co_head_num_convs = 1
    cfg.dataset.num_classes = num_classes
    torch.manual_seed(num_classes * 10 + num_co_heads)
    return CoDETRPlModel(experiment_spec=cfg)


def _swap(suffix, table):
    for src, dst in table:
        if suffix.startswith(src):
            return dst + suffix[len(src):]
    return suffix


def _to_reference_key(key, dec_layers):  # pylint: disable=R0911
    """Invert the TAO -> reference mapping for a ResNet/FPN CoDETR model."""
    m = re.match(r"model\.collab_heads\.0\.(cls_convs|reg_convs)\.(\d+)\.([01])\.(.*)", key)
    if m:
        tower, j, sub, param = m.groups()
        return f"bbox_head.0.{tower}.{j}.{'conv' if sub == '0' else 'gn'}.{param}"
    m = re.match(r"model\.collab_heads\.0\.(cls_pred|reg_pred|centerness_pred|scales)\.(.*)", key)
    if m:
        return f"bbox_head.0.{_ATSS_INV.get(m.group(1), m.group(1))}.{m.group(2)}"
    if key.startswith("model.downsample."):
        return "query_head." + key[len("model."):]
    if not key.startswith(DINO) or key.startswith(TIED):
        return None
    k = key[len(DINO):]
    if k.startswith("backbone.0.body."):
        return "backbone." + k[len("backbone.0.body."):]
    m = re.match(r"input_proj\.(\d+)\.([01])\.(.*)", k)
    if m:
        return f"neck.convs.{m.group(1)}.{'conv' if m.group(2) == '0' else 'gn'}.{m.group(3)}"
    if k == "label_enc.weight":
        return "query_head.label_embedding.weight"
    m = re.match(r"(class_embed|bbox_embed)\.(\d+)\.(.*)", k) or \
        re.match(r"transformer\.enc_out_(class_embed|bbox_embed)\.()(.*)", k)
    if m:
        head, idx, rest = m.groups()
        branch = "cls_branches" if head == "class_embed" else "reg_branches"
        rest = re.sub(r"^layers\.(\d+)\.", lambda x: f"{int(x.group(1)) * 2}.", rest)
        return f"query_head.{branch}.{idx or dec_layers}.{rest}"
    t = k[len("transformer."):]
    if t == "level_embed":
        return "query_head.transformer.level_embeds"
    if t.startswith("tgt_embed."):
        return "query_head.transformer.query_embed." + t[len("tgt_embed."):]
    m = re.match(r"(encoder|decoder)\.layers\.(\d+)\.(.*)", t)
    if m:
        table = _ENC_INV if m.group(1) == "encoder" else _DEC_INV
        return f"query_head.transformer.{m.group(1)}.layers.{m.group(2)}.{_swap(m.group(3), table)}"
    m = re.match(r"decoder\.ref_point_head\.layers\.(\d+)\.(.*)", t)
    if m:
        return f"query_head.transformer.decoder.ref_point_head.{int(m.group(1)) * 2}.{m.group(2)}"
    return "query_head.transformer." + t


def _reference_state_dict(model, dec_layers):
    """Build a reference-layout checkpoint (one ATSS head, as in the released Co-DINO)."""
    ref = {}
    for k, v in model.state_dict().items():
        ref_key = _to_reference_key(k, dec_layers)
        if ref_key is not None:
            ref[ref_key] = v.clone()
    # The reference label embedding has no background row.
    ref["query_head.label_embedding.weight"] = ref["query_head.label_embedding.weight"][:-1]
    # Keys with no TAO equivalent must be dropped, not reported as unexpected.
    ref["rpn_head.rpn_conv.weight"] = torch.zeros(1)
    ref["roi_head.0.bbox_head.fc_cls.weight"] = torch.zeros(1)
    ref["query_head.transformer.aux_pos_trans.0.weight"] = torch.zeros(1)
    return ref


@pytest.mark.cv_unit
def test_load_reference_checkpoint():
    """A reference checkpoint loads fully; only the head it lacks is missing."""
    src = _build_model(num_co_heads=1)
    dst = _build_model(num_co_heads=2)
    dec_layers = dst.experiment_spec.model.dec_layers
    ref = _reference_state_dict(src, dec_layers)
    assert is_reference_codetr_checkpoint(ref)

    result = load_codetr_state_dict(dst, ref, dec_layers=dec_layers, strict=True)

    assert result.unexpected_keys == []
    unexplained = [k for k in result.missing_keys
                   if not k.startswith(("model.collab_heads.1.",) + TIED)]
    assert unexplained == []
    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    for k in src_sd:
        if k.startswith(DINO + "label_enc."):
            assert torch.equal(dst_sd[k][:-1], src_sd[k][:-1])
        elif not k.startswith(TIED) and "num_batches_tracked" not in k:
            assert torch.equal(dst_sd[k], src_sd[k]), k


@pytest.mark.cv_unit
def test_load_tao_checkpoint(tmp_path):
    """A checkpoint saved by a TAO Co-DETR run round-trips without missing keys."""
    src = _build_model()
    dst = _build_model()
    path = tmp_path / "codetr.pth"
    checkpoint = {"state_dict": src.state_dict()}
    src.on_save_checkpoint(checkpoint)
    assert checkpoint["tao_model"] == "codetr"
    torch.save(checkpoint, path)

    state_dict = load_codetr_checkpoint(str(path))
    assert not is_reference_codetr_checkpoint(state_dict)
    keys_before = list(state_dict)
    result = load_codetr_state_dict(dst, state_dict, dec_layers=dst.experiment_spec.model.dec_layers, strict=True)

    assert list(state_dict) == keys_before
    assert result.missing_keys == [] and result.unexpected_keys == []
    for k, v in dst.state_dict().items():
        assert torch.equal(v, state_dict[k]), k


@pytest.mark.cv_unit
def test_load_reference_checkpoint_new_num_classes():
    """Changing num_classes re-initializes only the class-sized layers."""
    src = _build_model(num_classes=4, num_co_heads=1)
    dst = _build_model(num_classes=7, num_co_heads=1)
    dec_layers = dst.experiment_spec.model.dec_layers
    init_sd = {k: v.clone() for k, v in dst.state_dict().items()}

    result = load_codetr_state_dict(dst, _reference_state_dict(src, dec_layers), dec_layers=dec_layers,
                                    strict=True)

    assert result.unexpected_keys == []
    dst_sd, src_sd = dst.state_dict(), src.state_dict()
    for k, v in dst_sd.items():
        if v.shape != src_sd[k].shape:
            assert torch.equal(v, init_sd[k]), k
    backbone_key = next(k for k in dst_sd if k.startswith(DINO + "backbone.0.body."))
    assert torch.equal(dst_sd[backbone_key], src_sd[backbone_key])


@pytest.mark.cv_unit
def test_strict_rejects_mismatched_spec():
    """A spec the checkpoint does not cover fails instead of training from random init."""
    src = _build_model(num_feature_levels=2)
    dst = _build_model(num_feature_levels=3)
    dec_layers = dst.experiment_spec.model.dec_layers

    with pytest.raises(ValueError, match="not found in the checkpoint"):
        load_codetr_state_dict(dst, src.state_dict(), dec_layers=dec_layers, strict=True)
    result = load_codetr_state_dict(dst, src.state_dict(), dec_layers=dec_layers)
    assert any(k.startswith(DINO + "input_proj.2.") for k in result.missing_keys)
