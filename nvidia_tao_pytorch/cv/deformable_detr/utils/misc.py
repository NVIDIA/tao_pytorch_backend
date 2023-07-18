
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

""" Misc functions. """

import os
import pickle  # nosec B403

import torch
import torch.distributed as dist

from nvidia_tao_pytorch.core.cookbooks.tlt_pytorch_cookbook import TLTPyTorchCookbook
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import patch_decrypt_checkpoint


DEFAULT_TARGET_CLASS_MAPPING = {
    "Person": "person",
    "Person Group": "person",
    "Rider": "person",
    "backpack": "bag",
    "face": "face",
    "large_bag": "bag",
    "person": "person",
    "person group": "person",
    "person_group": "person",
    "personal_bag": "bag",
    "rider": "person",
    "rolling_bag": "bag",
    "rollingbag": "bag",
    "largebag": "bag",
    "personalbag": "bag"
}

DEFAULT_CLASS_LOOKUP_TABLE = {'person': 1, 'face': 2, 'bag': 3}


def get_categories(cat_map):
    """
    Function to convert the category map to COCO annotation format

    Args:
        cat_map (dict): Category map

    Returns:
        categories_info (list): COCO annotation format of the category map
        categories_dict (dict): In a format of {"class_name": "id"}
    """
    categories_info = []
    categories_dict = {}
    for i, class_name in enumerate(sorted(set(cat_map.values()), reverse=True)):
        category = {
            'id': i + 1,
            'name': class_name
        }
        categories_info.append(category)
        categories_dict[class_name] = i + 1
    return categories_info, categories_dict


def check_and_create(d):
    """Create a directory."""
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def load_pretrained_weights(pretrained_path):
    """To get over pytorch lightning module in the checkpoint state_dict."""
    temp = torch.load(pretrained_path,
                      map_location="cpu")

    if temp.get("state_dict_encrypted", False):
        # Retrieve encryption key from TLTPyTorchCookbook.
        key = TLTPyTorchCookbook.get_passphrase()
        if key is None:
            raise PermissionError("Cannot access model state dict without the encryption key")
        temp = patch_decrypt_checkpoint(temp, key)

    # for loading pretrained I3D weights released on
    # https://github.com/piergiaj/pytorch-i3d
    if "state_dict" not in temp:
        return temp

    state_dict = {}
    for key, value in list(temp["state_dict"].items()):
        if "module" in key:
            new_key = ".".join(key.split(".")[1:])
            state_dict[new_key] = value
        elif key.startswith("backbone."):
            # MMLab compatible weight loading
            new_key = key[9:]
            state_dict[new_key] = value
        elif key.startswith("ema_"):
            # Do not include ema params from MMLab
            continue
        else:
            state_dict[key] = value

    return state_dict


def match_name_keywords(n, name_keywords):
    """match_name_keywords"""
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object.
    Returns:
        list[data]: list of data gathered from each rank.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to('cuda')

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device='cuda')
    size_list = [torch.tensor([0], device='cuda') for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device='cuda'))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device='cuda')
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))  # nosec B301

    return data_list


def collate_fn(batch):
    """Custom collate function for DETR-like models.

    DETR models use mutli-scale resize and random cropping which results in the varying input resolution of a single image.
    Hence, we need a custom collate_fn to pad additional regions and pass that as mask to transformer.

    Args:
        batch (tuple): tuple of a single batch. Contains image and label tensors

    Returns:
        batch (tuple): tuple of a single batch with uniform image resolution after padding.
    """
    batch = list(zip(*batch))
    batch[0] = tensor_from_tensor_list(batch[0], batch[1])
    return tuple(batch)


def _max_by_axis(the_list):
    """Get maximum image shape for padding."""
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def tensor_from_tensor_list(tensor_list, targets):
    """Convert list of tensors with different size to fixed resolution.

    The final size is determined by largest height and width.
    In theory, the batch could become [3, 1333, 1333] on dataset with different aspect ratio, e.g. COCO
    A fourth channel dimension is the mask region in which 0 represents the actual image and 1 means the padded region.
    This is to give size information to the transformer archicture. If transform-padding is applied,
    then only the pre-padded regions gets mask value of 1.

    Args:
        tensor_list (List[Tensor]): list of image tensors
        targets (List[dict]): list of labels that contain the size information

    Returns:
        tensors (torch.Tensor): list of image tensors in shape of (B, 4, H, W)
    """
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        temp_tensors = torch.zeros((b, c, h, w), dtype=dtype, device=device)
        mask = torch.ones((b, 1, h, w), dtype=dtype, device=device)
        tensors = torch.concat((temp_tensors, mask), 1)
        for img, target, pad_img in zip(tensor_list, targets, tensors):
            # Get original image size before transform-padding
            # If no transform-padding has been applied,
            # then height == img.shape[1] and width == img.shape[2]
            actual_height, actual_width = target['size']
            pad_img[:img.shape[0], :actual_height, :actual_width].copy_(img[:, :actual_height, :actual_width])
            pad_img[c, :actual_height, :actual_width] = 0  # set zeros for mask in non-padded area
    else:
        raise ValueError('Channel size other than 3 is not supported')
    return tensors


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_total_grad_norm(parameters, norm_type=2):
    """Get toal gradient norm."""
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def is_dist_avail_and_initialized():
    """Check if DDP is initialized."""
    is_dist = True
    if not dist.is_available():
        is_dist = False
    else:
        is_dist = dist.is_initialized() or False
    return is_dist


def get_world_size():
    """Get world size."""
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_global_rank():
    """Get global rank."""
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()
