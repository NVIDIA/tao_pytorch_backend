# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for CoDETR weight loading."""

import re

import torch

from nvidia_tao_pytorch.core.tlt_logging import logging as logger
from nvidia_tao_pytorch.core.utils.ptm_utils import load_pretrained_weights

# Key prefixes that identify a reference (MMDetection) Co-DETR checkpoint.
REFERENCE_KEY_PREFIXES = ("backbone.", "query_head.", "neck.", "bbox_head.")

# Missing keys that are expected after loading a checkpoint: collab heads and
# the downsample conv that the checkpoint does not provide, buffers, and
# decoder heads that are tied to the loaded top-level heads.
_EXPECTED_MISSING_PREFIXES = (
    "model.collab_heads.",
    "model.downsample.",
)
_EXPECTED_MISSING_SUBSTR = (
    "rope_win.freqs",
    "rope_glb.freqs",
    "label_enc.weight",
    "num_batches_tracked",
    "transformer.decoder.bbox_embed",
    "transformer.decoder.class_embed",
)
# Layers sized by num_classes; skipping them on a num_classes change is expected.
_CLASS_SIZED_SUBSTR = ("class_embed", "label_enc", "cls_pred")


# ---------------------------------------------------------------------------
# Reference Co-DETR (MMDetection) → TAO CoDETR key mapping
# ---------------------------------------------------------------------------

# Encoder layer: MMDet names → TAO names
_ENC_LAYER_MAP = [
    ("attentions.0.", "self_attn."),
    ("ffns.0.layers.0.0.", "linear1."),
    ("ffns.0.layers.1.", "linear2."),
    ("norms.0.", "norm1."),
    ("norms.1.", "norm2."),
]

# Decoder layer: MMDet names → TAO names
# In MMDet: attentions.0 = self-attn (norms.0), attentions.1 = cross-attn (norms.1), FFN (norms.2)
# In TAO:  cross_attn (norm1), self_attn (norm2), FFN (norm3)
_DEC_LAYER_MAP = [
    ("attentions.0.attn.", "self_attn."),
    ("attentions.1.", "cross_attn."),
    ("ffns.0.layers.0.0.", "linear1."),
    ("ffns.0.layers.1.", "linear2."),
    ("norms.0.", "norm2."),       # self-attn norm
    ("norms.1.", "norm1."),       # cross-attn norm
    ("norms.2.", "norm3."),       # FFN norm
]

# MLP layer index remap: MMDet Sequential(Linear,ReLU,Linear,ReLU,Linear)
# stores layers at indices 0, 2, 4 → TAO MLP uses layers.0, layers.1, layers.2
_MLP_IDX = {"0": "0", "2": "1", "4": "2"}


def _map_transformer_layer(suffix, layer_map):
    """Apply encoder/decoder layer key mapping."""
    for src, dst in layer_map:
        if suffix.startswith(src):
            return dst + suffix[len(src):]
    return suffix


def _map_reg_branch(suffix):
    """Map reg_branches.{layer}.{idx}.{param} → bbox_embed.{layer}.layers.{mapped_idx}.{param}."""
    m = re.match(r"(\d+)\.(\d+)\.(.*)", suffix)
    if m:
        layer, idx, rest = m.group(1), m.group(2), m.group(3)
        mapped_idx = _MLP_IDX.get(idx, idx)
        return f"{layer}.layers.{mapped_idx}.{rest}"
    return suffix


def map_codetr_checkpoint(state_dict, num_backbone_stages=4, dec_layers=6):
    """Map a reference Co-DETR checkpoint to TAO CoDETR model keys.

    The reference checkpoint (from the MMDetection-based Co-DETR repo) uses
    a different naming convention than the TAO implementation.  This function
    converts the key names so that the state dict can be loaded into a
    CoDETRModel (after stripping the ``model.`` PL prefix).

    Keys that have no TAO equivalent (rpn_head, roi_head, aux_pos_trans, etc.)
    are silently dropped.

    Args:
        state_dict (dict): raw reference checkpoint state dict.
        num_backbone_stages (int): number of backbone stages (for neck mapping).
        dec_layers (int): number of decoder layers (to identify enc_out heads).

    Returns:
        dict: state dict with TAO-compatible key names.
    """
    mapped = {}
    skipped = []

    for key, value in state_dict.items():
        new_key = _map_single_key(key, num_backbone_stages, dec_layers)
        if new_key is None:
            skipped.append(key)
            continue
        mapped[new_key] = value

    if skipped:
        logger.info("Skipped %d checkpoint keys with no TAO equivalent "
                    "(rpn_head, roi_head, aux_pos_trans, ...)", len(skipped))
    return mapped


def _map_single_key(key, num_backbone_stages, dec_layers):  # pylint: disable=R0911
    """Map a single reference key to its TAO equivalent, or None to skip.

    The TAO PL model hierarchy is:
        CoDETRPlModel.model (CoDETRModel)
            .model (DINOModel)
                .model (DINO)  ← backbone, transformer, heads live here
            .collab_heads      ← collaborative ATSS heads

    So DINO internals need prefix ``model.model.model.`` and collab heads
    need prefix ``model.collab_heads.``.
    """
    # Prefix constants matching the PL model hierarchy
    DINO = "model.model.model."   # CoDETRPlModel → CoDETRModel → DINOModel → DINO
    COLLAB = "model.collab_heads."  # CoDETRPlModel → CoDETRModel → collab_heads

    # ------------------------------------------------------------------
    # 1. Backbone:  backbone.X → DINO.backbone.0.body.X
    # ------------------------------------------------------------------
    if key.startswith("backbone."):
        return DINO + "backbone.0.body." + key[len("backbone."):]

    # ------------------------------------------------------------------
    # 2. Neck → input_proj (FPN-style) or backbone.sfp (SFP-style)
    # ------------------------------------------------------------------
    # SFP neck keys (ViT backbone): neck.p2.* / p3 / p4 / p5 / p6
    if key.startswith("neck.") and key.split(".")[1] in ("p2", "p3", "p4", "p5", "p6"):
        rest = key[len("neck."):]
        return DINO + "backbone.0.body.sfp." + rest

    # FPN neck keys (Swin/ResNet backbone): neck.convs.* / extra_convs.*
    if key.startswith("neck.convs."):
        rest = key[len("neck.convs."):]
        m = re.match(r"(\d+)\.(conv|gn)\.(.*)", rest)
        if m:
            idx, mod, param = m.group(1), m.group(2), m.group(3)
            sub = "0" if mod == "conv" else "1"
            return f"{DINO}input_proj.{idx}.{sub}.{param}"
    if key.startswith("neck.extra_convs."):
        rest = key[len("neck.extra_convs."):]
        m = re.match(r"(\d+)\.(conv|gn)\.(.*)", rest)
        if m:
            idx, mod, param = m.group(1), m.group(2), m.group(3)
            proj_idx = num_backbone_stages + int(idx)
            sub = "0" if mod == "conv" else "1"
            return f"{DINO}input_proj.{proj_idx}.{sub}.{param}"

    # ------------------------------------------------------------------
    # 3. query_head.transformer → DINO.transformer
    # ------------------------------------------------------------------
    if key.startswith("query_head.transformer."):
        suffix = key[len("query_head.transformer."):]

        # 3a. level_embeds → level_embed
        if suffix == "level_embeds":
            return DINO + "transformer.level_embed"

        # 3b. query_embed → tgt_embed
        if suffix.startswith("query_embed."):
            return DINO + "transformer.tgt_embed." + suffix[len("query_embed."):]

        # 3c. enc_output, enc_output_norm (direct)
        if suffix.startswith("enc_output.") or suffix.startswith("enc_output_norm."):
            return DINO + "transformer." + suffix

        # 3d. Encoder layers
        enc_m = re.match(r"encoder\.layers\.(\d+)\.(.*)", suffix)
        if enc_m:
            layer_idx, layer_suffix = enc_m.group(1), enc_m.group(2)
            mapped_suffix = _map_transformer_layer(layer_suffix, _ENC_LAYER_MAP)
            return f"{DINO}transformer.encoder.layers.{layer_idx}.{mapped_suffix}"

        # 3e. Decoder layers
        dec_m = re.match(r"decoder\.layers\.(\d+)\.(.*)", suffix)
        if dec_m:
            layer_idx, layer_suffix = dec_m.group(1), dec_m.group(2)
            mapped_suffix = _map_transformer_layer(layer_suffix, _DEC_LAYER_MAP)
            return f"{DINO}transformer.decoder.layers.{layer_idx}.{mapped_suffix}"

        # 3f. Decoder ref_point_head (MLP)
        rph_m = re.match(r"decoder\.ref_point_head\.(\d+)\.(.*)", suffix)
        if rph_m:
            idx, param = rph_m.group(1), rph_m.group(2)
            mapped_idx = _MLP_IDX.get(idx, idx)
            return f"{DINO}transformer.decoder.ref_point_head.layers.{mapped_idx}.{param}"

        # 3g. Decoder norm
        if suffix.startswith("decoder.norm."):
            return DINO + "transformer.decoder.norm." + suffix[len("decoder.norm."):]

        # 3h. aux_pos_trans — skip (Co-DETR specific, not in TAO)
        if "aux_pos_trans" in suffix:
            return None

        # Fallback for other transformer keys
        return DINO + "transformer." + suffix

    # ------------------------------------------------------------------
    # 4. query_head.cls_branches → class_embed / enc_out_class_embed
    # ------------------------------------------------------------------
    if key.startswith("query_head.cls_branches."):
        rest = key[len("query_head.cls_branches."):]
        m = re.match(r"(\d+)\.(.*)", rest)
        if m:
            idx, param = int(m.group(1)), m.group(2)
            if idx == dec_layers:
                return f"{DINO}transformer.enc_out_class_embed.{param}"
            return f"{DINO}class_embed.{idx}.{param}"

    # ------------------------------------------------------------------
    # 5. query_head.reg_branches → bbox_embed / enc_out_bbox_embed
    # ------------------------------------------------------------------
    if key.startswith("query_head.reg_branches."):
        rest = key[len("query_head.reg_branches."):]
        m = re.match(r"(\d+)\.(.*)", rest)
        if m:
            branch_idx = int(m.group(1))
            branch_rest = m.group(2)
            mlp_m = re.match(r"(\d+)\.(.*)", branch_rest)
            if mlp_m:
                seq_idx, param = mlp_m.group(1), mlp_m.group(2)
                mapped_idx = _MLP_IDX.get(seq_idx, seq_idx)
                if branch_idx == dec_layers:
                    return f"{DINO}transformer.enc_out_bbox_embed.layers.{mapped_idx}.{param}"
                return f"{DINO}bbox_embed.{branch_idx}.layers.{mapped_idx}.{param}"

    # ------------------------------------------------------------------
    # 6. query_head.label_embedding → label_enc
    # ------------------------------------------------------------------
    if key.startswith("query_head.label_embedding."):
        rest = key[len("query_head.label_embedding."):]
        return DINO + "label_enc." + rest

    # ------------------------------------------------------------------
    # 7. query_head.downsample → CoDETRModel.downsample
    # ------------------------------------------------------------------
    if key.startswith("query_head.downsample."):
        rest = key[len("query_head.downsample."):]
        return "model.downsample." + rest

    # ------------------------------------------------------------------
    # 8. bbox_head → collab_heads
    # ------------------------------------------------------------------
    if key.startswith("bbox_head."):
        rest = key[len("bbox_head."):]
        m = re.match(r"(\d+)\.(cls_convs|reg_convs)\.(\d+)\.(conv|gn)\.(.*)", rest)
        if m:
            head_i, tower, j, mod, param = m.groups()
            sub = "0" if mod == "conv" else "1"
            return f"{COLLAB}{head_i}.{tower}.{j}.{sub}.{param}"
        m = re.match(r"(\d+)\.atss_cls\.(.*)", rest)
        if m:
            return f"{COLLAB}{m.group(1)}.cls_pred.{m.group(2)}"
        m = re.match(r"(\d+)\.atss_reg\.(.*)", rest)
        if m:
            return f"{COLLAB}{m.group(1)}.reg_pred.{m.group(2)}"
        m = re.match(r"(\d+)\.atss_centerness\.(.*)", rest)
        if m:
            return f"{COLLAB}{m.group(1)}.centerness_pred.{m.group(2)}"
        m = re.match(r"(\d+)\.scales\.(.*)", rest)
        if m:
            return f"{COLLAB}{m.group(1)}.scales.{m.group(2)}"

    # ------------------------------------------------------------------
    # 9. rpn_head, roi_head — skip (not used in TAO CoDETR)
    # ------------------------------------------------------------------
    if key.startswith("rpn_head.") or key.startswith("roi_head."):
        return None

    # Unmapped — return as-is (will be caught by strict=False)
    return key


def is_reference_codetr_checkpoint(state_dict):
    """Return True if ``state_dict`` uses the reference (MMDetection) Co-DETR layout."""
    return any(k.startswith(REFERENCE_KEY_PREFIXES) for k in state_dict)


def load_codetr_checkpoint(path):
    """Read a TAO or reference Co-DETR checkpoint file and return its state dict.

    TAO checkpoints carry ``tao_model: codetr`` and are returned unchanged;
    key mapping happens in :func:`load_codetr_state_dict`.

    Args:
        path (str): path to a ``.pth``, ``.tlt`` or ``.safetensors`` checkpoint.

    Returns:
        dict: the checkpoint's state dict.
    """
    return load_pretrained_weights(path, ptm_adapter=lambda _name, state_dict, model_type=None: state_dict)


def load_codetr_state_dict(model, state_dict, dec_layers=6, strict=False):
    """Load a TAO or reference Co-DETR state dict into a ``CoDETRPlModel``.

    Reference checkpoints (e.g. the Co-DETR authors' release on Hugging Face)
    are detected by key prefix and mapped to TAO names. EMA keys are dropped,
    ``label_enc`` is padded for the background token, and size-mismatched keys
    (e.g. classification layers when ``num_classes`` changes) are skipped so
    they keep their initialization.

    Args:
        model (CoDETRPlModel): model to load the weights into.
        state_dict (dict): TAO (``model.model.model.*``) or reference
            (``backbone.*``, ``query_head.*``, ...) state dict.
        dec_layers (int): number of decoder layers, used to map reference heads.
        strict (bool): raise if any weight other than collab heads, tied
            aliases, buffers or class-sized layers is left uninitialized.

    Returns:
        torch.nn.modules.module._IncompatibleKeys: missing and unexpected keys.

    Raises:
        ValueError: if ``strict`` and the checkpoint does not cover the model.
    """
    state_dict = dict(state_dict)
    # EMA keys are flattened with underscores and are not usable directly.
    ema_keys = [k for k in state_dict if k.startswith("ema_")]
    if ema_keys:
        logger.info("Dropping %d EMA keys from checkpoint.", len(ema_keys))
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ema_")}

    if is_reference_codetr_checkpoint(state_dict):
        logger.info("Detected reference Co-DETR checkpoint - mapping keys to TAO format.")
        # num_backbone_stages: for FPN-style necks derive from neck.convs;
        # for SFP-style necks (ViT) this is unused (SFP maps to backbone).
        num_backbone_stages = len({
            k.split('.')[2] for k in state_dict if k.startswith('neck.convs.')
        })
        state_dict = map_codetr_checkpoint(
            state_dict,
            num_backbone_stages=num_backbone_stages,
            dec_layers=dec_layers,
        )

    model_sd = model.state_dict()

    # Pad label_enc from num_classes to num_classes+1 (background token)
    label_key = "model.model.model.label_enc.weight"
    if label_key in state_dict and label_key in model_sd:
        ckpt_w = state_dict[label_key]
        if ckpt_w.shape[0] == model_sd[label_key].shape[0] - 1:
            state_dict[label_key] = torch.cat(
                [ckpt_w, torch.zeros(1, ckpt_w.shape[1], dtype=ckpt_w.dtype, device=ckpt_w.device)], dim=0
            )

    # Skip size-mismatched keys (e.g. class heads for a different num_classes)
    mismatched = [k for k, v in state_dict.items() if k in model_sd and v.shape != model_sd[k].shape]
    for k in mismatched:
        log = logger.info if any(s in k for s in _CLASS_SIZED_SUBSTR) else logger.warning
        log("Skipping size-mismatched key %s: ckpt %s vs model %s",
            k, list(state_dict[k].shape), list(model_sd[k].shape))
    state_dict = {k: v for k, v in state_dict.items() if k not in mismatched}

    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        # Collab/aux heads, tied aliases, buffers and resized class layers are
        # expected to be missing; any other parameter indicates a spec mismatch.
        class_sized = {k for k in mismatched if any(s in k for s in _CLASS_SIZED_SUBSTR)}
        unexpected_missing = [
            k for k in result.missing_keys
            if not k.startswith(_EXPECTED_MISSING_PREFIXES) and
            not any(s in k for s in _EXPECTED_MISSING_SUBSTR) and
            k not in class_sized
        ]
        logger.info(
            "Missing keys: %d total (%d expected for collab heads / buffers / class layers, %d unexpected)",
            len(result.missing_keys), len(result.missing_keys) - len(unexpected_missing),
            len(unexpected_missing),
        )
        if unexpected_missing:
            msg = (f"{len(unexpected_missing)} model weights were not found in the checkpoint; check that "
                   f"the spec (backbone, feature levels, layers) matches it: {unexpected_missing[:20]}")
            if strict:
                raise ValueError(msg)
            logger.error(msg)
    if result.unexpected_keys:
        logger.warning("Unexpected keys (in checkpoint, not in model): %s", result.unexpected_keys[:10])
    logger.info("Matched %d of %d model keys from checkpoint (%d size-mismatched keys skipped).",
                len(state_dict) - len(result.unexpected_keys), len(model_sd), len(mismatched))
    return result
