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

"""Inferencer."""
import torch
from nvidia_tao_pytorch.cv.pose_classification.utils.common_utils import data_to_device


class Inferencer():
    """
    Pytorch model inferencer.

    This class takes a PyTorch model, moves it to GPU for execution, and provides a method for making
    inferences with the model on given data. The class optionally returns probabilities if 'ret_prob' is set to True.

    Attributes:
        model (torch.nn.Module): The PyTorch model to use for inference.
        ret_prob (bool): If True, the 'inference' method will return probabilities instead of class IDs.
    """

    def __init__(self, model, ret_prob=False):
        """
        Initialize Inferencer with a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model to use for inference.
            ret_prob (bool, optional): If True, the 'inference' method will return probabilities instead of
                                       class IDs. Defaults to False.
        """
        self.model = model
        self.model.eval()
        self.model.cuda()
        self.ret_prob = ret_prob

    def inference(self, data):
        """
        Perform inference on the given data.

        The data is moved to GPU and passed through the model for inference. If 'ret_prob' is True, softmax
        is applied to the model output to get probabilities, and these probabilities are returned. Otherwise,
        the IDs of the classes with the highest scores are returned.

        Args:
            data (torch.Tensor): A tensor containing the data to perform inference on.

        Returns:
            numpy.ndarray: If 'ret_prob' is True, an array of probabilities. Otherwise, an array of class IDs.
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
