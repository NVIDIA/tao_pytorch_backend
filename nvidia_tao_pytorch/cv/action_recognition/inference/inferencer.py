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

"""Inferencer"""
import torch
from nvidia_tao_pytorch.cv.action_recognition.utils.common_utils import data_to_device


class Inferencer():
    """PyTorch model inferencer.

    This class takes a PyTorch model and a boolean flag indicating whether to return probabilities as input.
    It initializes the model, sets it to evaluation mode, and moves it to the GPU. It also provides a method for
    doing inference on input data and returning predicted class IDs or probabilities.
    """

    def __init__(self, model, ret_prob=False):
        """Initialize the inferencer with a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model.
            ret_prob (bool, optional): Whether to return probabilities. Defaults to False.
        """
        self.model = model
        self.model.eval()
        self.model.cuda()
        self.ret_prob = ret_prob

    def inference(self, data):
        """Do inference on input data and return predicted class IDs or probabilities.

        Args:
            data (torch.Tensor): The input data.

        Returns:
            numpy.ndarray or int: The predicted class IDs or probabilities.
        """
        cuda_data = data_to_device(data)
        cls_scores = self.model(cuda_data)

        if self.ret_prob:
            prob = torch.softmax(cls_scores, dim=1)
            prob = prob.detach().cpu().numpy()
            return prob

        pred_id = torch.argmax(cls_scores, dim=1)
        pred_id = pred_id.cpu().numpy()

        return pred_id


# @TODO(tylerz): TRT inference
class TRTInferencer():
    """TRT engine inferencer."""

    pass
