# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

""" VL utility functions. """

import random
from typing import List

import torch


def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j

    Args:
        tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(tokens_positive), max_text_len), dtype=torch.float)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except Exception:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except Exception:
                end_pos = None

        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        # assert beg_pos is not None and end_pos is not None
        positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map


def create_positive_map_from_span(tokenized, token_span, max_text_len=256):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j
    Args:
        tokenized:
            - input_ids: Tensor[1, ntokens]
            - attention_mask: Tensor[1, ntokens]
        token_span: list with length num_boxes.
            - each item: [start_idx, end_idx]
    """
    positive_map = torch.zeros((len(token_span), max_text_len), dtype=torch.float)
    for j, tok_list in enumerate(token_span):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except Exception:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except Exception:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)

    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def build_captions_and_token_span(cat_list, force_lowercase):
    """
    Return:
        captions: str
        cat2tokenspan: dict
            {
                'dog': [[0, 2]],
                ...
            }
    """
    cat2tokenspan = {}
    captions = ""
    for catname in cat_list:
        class_name = catname
        if force_lowercase:
            class_name = class_name.lower()
        if "/" in class_name:
            class_name_list: List = class_name.strip().split("/")
            class_name_list.append(class_name)
            class_name: str = random.choice(class_name_list)

        tokens_positive_i = []
        subnamelist = [i.strip() for i in class_name.strip().split(" ")]
        for subname in subnamelist:
            if len(subname) == 0:
                continue
            if len(captions) > 0:
                captions = captions + " "
            strat_idx = len(captions)
            end_idx = strat_idx + len(subname)
            tokens_positive_i.append([strat_idx, end_idx])
            captions = captions + subname

        if len(tokens_positive_i) > 0:
            captions = captions + " ."
            cat2tokenspan[class_name] = tokens_positive_i

    return captions, cat2tokenspan


def build_id2posspan_and_caption(category_dict: dict):
    """Build id2pos_span and caption from category_dict

    Args:
        category_dict (dict): category_dict
    """
    cat_list = [item["name"].lower() for item in category_dict]
    id2catname = {item["id"]: item["name"].lower() for item in category_dict}
    caption, cat2posspan = build_captions_and_token_span(cat_list, force_lowercase=True)
    id2posspan = {catid: cat2posspan[catname] for catid, catname in id2catname.items()}
    return id2posspan, caption
