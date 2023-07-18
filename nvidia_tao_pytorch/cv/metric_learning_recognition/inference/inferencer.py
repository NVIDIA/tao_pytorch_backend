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

import os

import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from pytorch_metric_learning.utils.inference import InferenceModel

from nvidia_tao_pytorch.cv.metric_learning_recognition.utils.match_finder import EmbeddingKNN
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.build_nn_model import build_model
from nvidia_tao_pytorch.cv.metric_learning_recognition.model.pl_ml_recog_model import MLRecogModel
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.transforms import build_transforms
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.build_data_loader import build_inference_dataloader
from nvidia_tao_pytorch.cv.metric_learning_recognition.dataloader.build_data_loader import build_dataloader
from nvidia_tao_pytorch.cv.metric_learning_recognition.utils.common_utils import no_folders_in
from nvidia_tao_pytorch.cv.re_identification.utils.common_utils import read_image
import nvidia_tao_pytorch.core.loggers.api_logging as status_logging


class Inferencer():
    """Pytorch model inferencer."""

    def __init__(self, cfg, results_dir):
        """Inferencer for Metric Learning Recognition model. The inferencer would
        load the model and process the dataset, and run inference on the inputs.
        Three formats of inputs are supported: a single image, a folder of images,
        a folder of classification dataset. The inferencer would return the predicted
        class if the input is a single image, and return a csv table of predicted
        classes if the input is a folder of images or a folder of classification
        dataset.

        During inference, the model would be loaded from the checkpoint specified
        in the config. If no checkpoint is specified, the model would be initialized
        randomly. The model would be loaded to the device specified in the config.
        The model would be set to eval mode during inference. The reference dataset
        would be loaded from the `dataset.val_dataset.reference` specified in the
        config. The query dataset would be loaded from `inference.input_path` specified
        in the config. The dataset would be processed with the non-train mode
        transforms returned from the
        `cv.metric_learning_recognition.dataloader.transforms.build_transforms`
        function.

        The reference and query embeddings would be generated from the model. The
        K nearest neighbors of the query embeddings would be found from the reference
        embeddings. The classes of the K nearest neighbors would be the returned.
        If the input is a folder of images or a folder of classification dataset,
        a csv table would be generated and it would also include the distances
        of the query embeddings from the reference neighbors.

        Args:
            cfg (DictConfig): Hydra config object for inference task
            results_dir (String): path to save the results
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            gpu_id = cfg["inference"]["gpu_id"]
            torch.cuda.set_device(gpu_id)
        self.experiment_spec = cfg
        self.results_dir = results_dir

        self.__load_model()
        self.model.to(device)

        _, _, _, self.dataset_dict = build_dataloader(cfg, mode="inference")
        self.transforms = build_transforms(cfg, is_train=False)
        self.__load_inferencer()
        self.class_dict = self.dataset_dict["gallery"].class_dict
        self.topk = cfg["inference"]["topk"]

    def __load_model(self):
        if not self.experiment_spec["inference"]["checkpoint"]:
            self.model = build_model(self.experiment_spec, checkpoint_to_load=False)
            status_logging.get_status_logger().write(
                message="No weights loaded, model initialized randomly.",
                status_level=status_logging.Status.SKIPPED)
        else:
            status_logging.get_status_logger().write(
                message=f"Loading checkpoint: {self.experiment_spec['inference']['checkpoint']}",
                status_level=status_logging.Status.STARTED
            )
            self.model = MLRecogModel.load_from_checkpoint(
                self.experiment_spec["inference"]["checkpoint"],
                map_location="cpu",
                experiment_spec=self.experiment_spec,
                results_dir=self.results_dir,
                subtask="inference")
        self.model.eval()

    def __load_inferencer(self):
        # TODO: reset before and after for better mem control?
        infernce_knn_func = EmbeddingKNN(reset_before=False,
                                         reset_after=False)
        self.inference_model = InferenceModel(self.model,
                                              knn_func=infernce_knn_func)
        self.inference_model.train_knn(self.dataset_dict["gallery"])

    def preprocess(self, image_path):
        """Preprocesses a single image file to inferencer.

        Args:
            image_path (str): path of an image file.

        Returns:
            image_tensor (torch.Tensor): image tensor with shape (1, C, W, H).
        """
        image = read_image(image_path)
        image_tensor = self.transforms(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def infer_image(self):
        """Infers the class of a single image tensor.

        Returns:
            class_idx (int): the index of predicted class
        """
        image = self.experiment_spec['inference']['input_path']
        device = "cuda" if torch.cuda.is_available() else "cpu"
        img = self.preprocess(image).to(device)
        _, indices = self.inference_model.get_nearest_neighbors(img, k=self.topk)
        class_indices = [self.dataset_dict["gallery"][i][1] for i in indices[0]]
        # class_idx = Counter(class_indices).most_common(1)[0][0]
        class_ids = [self.class_dict[i] for i in class_indices]
        status_logging.get_status_logger().write(
            message=f"input image: {image}, predicted top {self.topk} class names: {class_ids}",
            status_level=status_logging.Status.SUCCESS)
        return class_ids

    def infer_image_dir(self):
        """Infers all images in an image folder or a classification folder.

        Returns:
            final_df (pd.DataFrame): a table displaying image file path,
                top k predicted classes, topk distances
        """
        inference_dataloader = build_inference_dataloader(
            self.experiment_spec)
        dfs = []
        for img_batch in tqdm(inference_dataloader):
            distances, indices = self.inference_model.get_nearest_neighbors(
                img_batch[0], k=self.topk)
            class_indices = [self.dataset_dict["gallery"][i][1] for i in indices.flatten()]
            class_labels = np.array([self.class_dict[idx] for idx in
                                     class_indices]).reshape(len(img_batch[0]), -1)
            df = pd.DataFrame(zip(list(img_batch[1]), class_labels.tolist(), distances.tolist()))
            dfs.append(df)
        csv_f = os.path.join(self.results_dir, 'result.csv')
        final_df = pd.concat(dfs)
        final_df.to_csv(csv_f, header=False, index=False)
        status_logging.get_status_logger().write(
            message=f"result saved at {csv_f}",
            status_level=status_logging.Status.SUCCESS)
        return final_df

    def infer(self):
        """Runs inference for files at `cfg.inference.input_path`.

        Returns:
            output (int / pd.DataFrame): If self.experiment_spec.inference.inference_input_type
                is `image`,the output is a integer of the predicted class index
                If self.experiment_spec.inference.inference_input_type is `xx_folder`
                the output is a table displaying `file_name, topk predicted
                classes, topk distances` of the examples in the folder line by line
        """
        # check input
        input_file = self.experiment_spec["inference"]["input_path"]
        inference_input_type = self.experiment_spec["inference"]["inference_input_type"]
        if os.path.isdir(input_file):
            if (not no_folders_in(input_file)) and inference_input_type != "classification_folder":
                raise ValueError("Folders detected in the dataset.query_dataset, The inference_input_type should be classification_folder")
            if no_folders_in(input_file) and inference_input_type != "image_folder":
                raise ValueError("No folders detected in the dataset.query_dataset, The inference_input_type should be image_folder")
        elif inference_input_type != "image":
            raise ValueError("The input is not a folder, try 'image' as the inference_input_type")
        if inference_input_type == "image":
            output = self.infer_image()
        else:
            output = self.infer_image_dir()
        return output
