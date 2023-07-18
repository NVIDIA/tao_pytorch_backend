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

"""Build torch data loader."""

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from nvidia_tao_pytorch.core.tlt_logging import logging
from nvidia_tao_pytorch.cv.optical_inspection.dataloader.oi_dataset import SiameseNetworkTRIDataset, get_sampler

# START SIAMESE DATALOADER


def build_dataloader(df, weightedsampling, split, data_config):
    """Build torch dataloader.

    Args:
        df (pd.DataFrame): The input data frame.
        weightedsampling (bool): Flag indicating whether to use weighted sampling.
        split (str): The split type ('train', 'valid', 'test', 'infer').
        data_config (OmegaConf.DictConf): Configuration spec for data loading.

    Returns:
        DataLoader: The built torch DataLoader object.
    """
    workers = data_config["workers"]
    batch_size = data_config["batch_size"]
    output_shape = data_config["output_shape"]
    rgb_mean = data_config["augmentation_config"]["rgb_input_mean"]
    rgb_std = data_config["augmentation_config"]["rgb_input_std"]

    train_transforms = transforms.Compose(
        [
            transforms.Resize((output_shape[0], output_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((output_shape[0], output_shape[1])),
            transforms.ToTensor(),
            transforms.Normalize(rgb_mean, rgb_std)
        ]
    )

    dataloader_kwargs = {
        "pin_memory": True,
        "batch_size": batch_size,
        "num_workers": workers
    }

    if split == 'train':
        input_data_path = data_config["train_dataset"]["images_dir"]
        dataset = SiameseNetworkTRIDataset(data_frame=df,
                                           train=True,
                                           input_data_path=input_data_path,
                                           transform=train_transforms,
                                           data_config=data_config)

        if weightedsampling:
            fpratio_sampling = data_config['fpratio_sampling']
            wt_sampler = get_sampler(dataset, fpratio_sampling)
            dataloader_kwargs["sampler"] = wt_sampler
        else:
            dataloader_kwargs["shuffle"] = True
        assert batch_size > 1, "Training batch size must be greater than 1."
        dataloader_kwargs["drop_last"] = True

    elif split == 'valid':
        input_data_path = data_config["validation_dataset"]["images_dir"]
        dataset = SiameseNetworkTRIDataset(data_frame=df,
                                           train=False,
                                           input_data_path=input_data_path,
                                           transform=test_transforms,
                                           data_config=data_config)
        dataloader_kwargs["shuffle"] = False

    elif split == 'test':
        input_data_path = data_config["test_dataset"]["images_dir"]
        dataset = SiameseNetworkTRIDataset(data_frame=df,
                                           train=False,
                                           input_data_path=input_data_path,
                                           transform=test_transforms,
                                           data_config=data_config)

        dataloader_kwargs["shuffle"] = False

    elif split == 'infer':
        input_data_path = data_config["infer_dataset"]["images_dir"]
        dataset = SiameseNetworkTRIDataset(data_frame=df,
                                           train=False,
                                           input_data_path=input_data_path,
                                           transform=test_transforms,
                                           data_config=data_config)

        dataloader_kwargs["shuffle"] = False

    # Build dataloader
    dataloader = DataLoader(
        dataset,
        **dataloader_kwargs
    )
    return dataloader
