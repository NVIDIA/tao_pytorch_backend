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

"""Visual ChangeNet Classification Inferencer"""
from nvidia_tao_pytorch.cv.optical_inspection.utils.common_utils import data_to_device
from torch.autograd import Variable
import torch


class Inferencer():
    """Pytorch model inferencer."""

    def __init__(self, model, difference_module='learnable'):
        """Initialize the Inferencer with a PyTorch model.

        Args:
            model (torch.nn.Module): The PyTorch model.
        """
        self.model = model
        self.model.eval()
        self.model.cuda()
        self.difference_module = difference_module

    def inference(self, data):
        """
        Perform inference using the model.

        Args:
            data (Tuple[torch.Tensor, torch.Tensor]): The input data tuple.

        Returns:
            torch.Tensor: The Siamese score tensor.
        """
        cuda_data = [data_to_device(Variable(item)) for item in data]

        output = self.model(cuda_data)
        if self.difference_module == 'learnable':
            output = torch.softmax(output, dim=1)[:, 1]
        return output
