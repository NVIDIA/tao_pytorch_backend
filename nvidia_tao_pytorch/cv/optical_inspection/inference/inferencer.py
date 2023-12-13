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
from nvidia_tao_pytorch.cv.optical_inspection.utils.common_utils import data_to_device
import torch.nn.functional as F
from torch.autograd import Variable


class Inferencer():
    """Pytorch model inferencer."""

    def __init__(self, model, ret_prob=False):
        """Initialize the Inferencer with a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model.
        """
        self.model = model
        self.model.eval()
        self.model.cuda()

    def inference(self, data):
        """
        Perform inference using the model.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): The input data tuple.

        Returns:
            torch.Tensor: The Siamese score tensor.
        """
        cuda_data = []
        cuda_data0 = data_to_device(Variable(data[0]))
        cuda_data1 = data_to_device(Variable(data[1]))
        cuda_data.append(cuda_data0)
        cuda_data.append(cuda_data1)

        # output1, output2 = self.model(cuda_data0, cuda_data1)
        output1, output2 = self.model(cuda_data)
        siam_score = F.pairwise_distance(output1, output2)
        # score = euclidean_distance.detach().cpu().numpy()

        return siam_score


# @TODO(tylerz): TRT inference
class TRTInferencer():
    """TRT engine inferencer."""

    pass
