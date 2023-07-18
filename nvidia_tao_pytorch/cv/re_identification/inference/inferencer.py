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
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import data_to_device


class Inferencer():
    """A class to perform inference with a PyTorch model.

    This class is designed to facilitate the inference process for a given PyTorch model. The model
    and data used for inference are assumed to be compatible with GPU processing.

    Args:
        model (torch.nn.Module): The PyTorch model to be used for inference. The model should already
        be in a state ready for inference (i.e., it should already be trained).

    Attributes:
        model (torch.nn.Module): The PyTorch model for inference.

    Methods:
        inference(data): Perform inference on the provided data.
    """

    def __init__(self, model):
        """Initialize the inferencer with a PyTorch model.

        This function prepares the model for inference by setting it to evaluation mode and moving it to GPU.

        Args:
            model (torch.nn.Module): The PyTorch model to be used for inference. The model should be in a state ready
            for inference (i.e., it should already be trained).
        """
        self.model = model
        self.model.eval()
        self.model.cuda()

    def inference(self, data):
        """Perform inference on the provided data and return the model's output.

        The data is first converted to a float tensor and moved to the device where the model resides (assumed to be a GPU).
        Then it is passed through the model, and the output is returned.

        Args:
            data (torch.Tensor): The input data for the model. The data should be compatible with the model's expected input format.

        Returns:
            torch.Tensor: The output of the model. For a model trained for re-identification, this would typically be the feature
            embeddings for the input images.
        """
        data = data.float()
        cuda_data = data_to_device(data)
        feat = self.model(cuda_data)
        return feat
