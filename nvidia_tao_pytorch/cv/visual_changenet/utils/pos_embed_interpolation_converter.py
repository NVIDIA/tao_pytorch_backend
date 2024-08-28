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

"""script to interpolate n x n ViT to m x m patches"""

import argparse
import torch


def interpolate_pos_embed(checkpoint_model,
                          orig_resolution=None,
                          orig_patch_size=14,
                          new_resolution=None,
                          new_patch_size=16):
    """Interpolate Positional Embedding from ViT.

    Args:
        cheeckpoint_model (dict): ViT state_dict
        orig_patch_size (int): original patch size
        new_patch_size (int): new patch size

    Return:
        checkpoint_model: updated state dict
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1

        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)  # 16

        if orig_resolution:
            assert orig_resolution == orig_size * orig_patch_size, \
                f"Resolution {orig_resolution} and patch size {orig_patch_size} do not match"
        else:
            orig_resolution = orig_size * orig_patch_size

        # height (== width) for the new position embedding
        if new_resolution:
            new_size = new_resolution // new_patch_size
        else:
            new_size = orig_resolution // new_patch_size

        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]

        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens.float(), size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model['pos_embed'] = new_pos_embed

        print(f"Resolution: {orig_resolution}")
        print(f"Old Pos Embed: {pos_embed_checkpoint.shape}")
        print(f"New Pos Embed: {new_pos_embed.shape}")

    return checkpoint_model


def interpolate_patch_embed(checkpoint, new_patch_size=16):
    """Interpolate patch_embed.

    Args:
        cheeckpoint (dict): ViT state_dict
        new_patch_size (int): new patch size

    Return:
        checkpoint: updated state dict

    """
    patch_embed = checkpoint['patch_embed.proj.weight']
    patch_embed = torch.nn.functional.interpolate(
        patch_embed.float(), size=(new_patch_size, new_patch_size), mode='bicubic', align_corners=False)
    checkpoint['patch_embed.proj.weight'] = patch_embed
    return checkpoint


if __name__ == '__main__':
    """main function."""
    parser = argparse.ArgumentParser(description='interpolate patch_embed kernel')
    parser.add_argument('-i', '--input', default='/path/to/eva_psz14.pt', type=str, metavar='PATH', required=True,
                        help='path to input checkpoint')
    parser.add_argument('-o', '--output', default='/path/to/eva_psz14to16.pt', type=str, metavar='PATH', required=True,
                        help='path to output checkpoint')
    parser.add_argument('-op', '--orig_patch_size', type=int, default=14,
                        help='original patch size. (default: 14)')
    parser.add_argument('-or', '--orig_resolution', type=int, default=None,
                        help='original image resolution. (default: None)')
    parser.add_argument('-np', '--new_patch_size', type=int, default=16,
                        help='new patch size. (default: 14)')
    parser.add_argument('-nr', '--new_resolution', type=int, default=None,
                        help='new image resolution. (default: None)')
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location=torch.device("cpu"))
    checkpoint = interpolate_patch_embed(checkpoint,
                                         new_patch_size=args.new_patch_size)

    # interpolate pos_embed too
    checkpoint = interpolate_pos_embed(checkpoint,
                                       orig_resolution=args.orig_resolution,
                                       orig_patch_size=args.orig_patch_size,
                                       new_resolution=args.new_resolution,
                                       new_patch_size=args.new_patch_size)

    print('======== new state_dict ========')
    for k, v in list(checkpoint.items()):
        print(k, '\t', v.shape)

    torch.save(checkpoint, args.output)
