# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Utils for Segformer Segmentation"""

import os
import struct
from eff.core.codec import encrypt_stream
import shutil


def check_and_create(d):
    """Create a directory."""
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)


def check_and_delete(d):
    """Delete a directory."""
    if os.path.isdir(d):
        shutil.rmtree(d)


def encrypt_onnx(tmp_file_name, output_file_name, key):
    """Encrypt the onnx model."""
    with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                           "wb") as open_encoded_file:
        # set the input name magic number
        open_encoded_file.write(struct.pack("<i", 0))

        encrypt_stream(
            input_stream=open_temp_file, output_stream=open_encoded_file,
            passphrase=key, encryption=True
        )
