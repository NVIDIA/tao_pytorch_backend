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

"""Utilities using the NVML library for GPU devices."""

import json
import pynvml

BRAND_NAMES = {
    pynvml.NVML_BRAND_UNKNOWN: "Unknown",
    pynvml.NVML_BRAND_QUADRO: "Quadro",
    pynvml.NVML_BRAND_TESLA: "Tesla",
    pynvml.NVML_BRAND_NVS: "NVS",
    pynvml.NVML_BRAND_GRID: "Grid",
    pynvml.NVML_BRAND_TITAN: "Titan",
    pynvml.NVML_BRAND_GEFORCE: "GeForce",
    pynvml.NVML_BRAND_NVIDIA_VAPPS: "NVIDIA Virtual Applications",
    pynvml.NVML_BRAND_NVIDIA_VPC: "NVIDIA Virtual PC",
    pynvml.NVML_BRAND_NVIDIA_VCS: "NVIDIA Virtual Compute Server",
    pynvml.NVML_BRAND_NVIDIA_VWS: "NVIDIA RTX Virtual Workstation",
    pynvml.NVML_BRAND_NVIDIA_VGAMING: "NVIDIA Cloud Gaming",
    pynvml.NVML_BRAND_QUADRO_RTX: "Quadro RTX",
    pynvml.NVML_BRAND_NVIDIA_RTX: "NVIDIA RTX",
    pynvml.NVML_BRAND_NVIDIA: "NVIDIA",
    pynvml.NVML_BRAND_GEFORCE_RTX: "GeForce RTX",
    pynvml.NVML_BRAND_TITAN_RTX: "TITAN RTX",
}


class GPUDevice:
    """Data structure to represent a GPU device."""

    def __init__(self, pci_bus_id,
                 device_name,
                 device_brand,
                 memory,
                 cuda_compute_capability):
        """Data structure representing a GPU device.

        Args:
            pci_bus_id (hex): PCI bus ID of the GPU.
            device_name (str): Name of the device GPU.
            device_branch (int): Brand of the GPU.
        """
        self.name = device_name
        self.pci_bus_id = pci_bus_id
        if device_brand in BRAND_NAMES.keys():
            self.brand = BRAND_NAMES[device_brand]
        else:
            self.brand = None
        self.defined = True
        self.memory = memory
        self.cuda_compute_capability = cuda_compute_capability

    def get_config(self):
        """Get json config of the device.

        Returns
            device_dict (dict): Dictionary containing data about the device.
        """
        assert self.defined, "Device wasn't defined."
        config_dict = {}
        config_dict["name"] = self.name.decode().replace(" ", "-")
        config_dict["pci_bus_id"] = self.pci_bus_id
        config_dict["brand"] = self.brand
        config_dict["memory"] = self.memory
        config_dict["cuda_compute_capability"] = self.cuda_compute_capability
        return config_dict

    def __str__(self):
        """Generate a printable representation of the device."""
        config = self.get_config()
        data_string = json.dumps(config, indent=2)
        return data_string


def pynvml_context(fn):
    """Simple decorator to setup python nvml context.

    Args:
        f: Function pointer.

    Returns:
        output of f.
    """
    def _fn_wrapper(*args, **kwargs):
        """Wrapper setting up nvml context."""
        try:
            pynvml.nvmlInit()
            return fn(*args, **kwargs)
        finally:
            pynvml.nvmlShutdown()
    return _fn_wrapper


@pynvml_context
def get_number_gpus_available():
    """Get the number of GPU's attached to the machine.

    Returns:
        num_gpus (int): Number of GPUs in the machine.
    """
    num_gpus = pynvml.nvmlDeviceGetCount()
    return num_gpus


@pynvml_context
def get_device_details():
    """Get details about each device.

    Returns:
        device_list (list): List of GPUDevice objects.
    """
    num_gpus = pynvml.nvmlDeviceGetCount()
    device_list = list()
    assert num_gpus > 0, "Atleast 1 GPU is required for TAO Toolkit to run."
    for idx in range(num_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
        device_name = pynvml.nvmlDeviceGetName(handle)
        brand_name = pynvml.nvmlDeviceGetBrand(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        cuda_compute_capability = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        device_list.append(
            GPUDevice(
                pci_info.busId,
                device_name,
                brand_name,
                memory.total,
                cuda_compute_capability
            )
        )
    return device_list
