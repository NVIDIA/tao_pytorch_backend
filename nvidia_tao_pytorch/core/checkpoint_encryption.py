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

"""File containing functions used to encrypt and decrypt encrypted checkpoints."""

import pickle
from io import BytesIO

from eff.codec import decrypt_stream, encrypt_stream, get_random_encryption


def decrypt_checkpoint(checkpoint, key):
    """Function to decrypt checkpoint using the supplied key.

    Args:
        checkpoint(dict): Pytorch checkpoint file.
        key (str): String key to decrypt the checkpoint file.

    Returns:
        checkpoint (dict): Checkpoint containing decrypted checkpoint file.
    """
    # Get the encrypted model state dict.
    encrypted_state_stream = BytesIO(checkpoint["state_dict"])
    # Get encryption_type
    encryption_type = checkpoint["state_dict_encrypted"]

    # Decrypt it to binary stream.
    decrypted_state_stream = BytesIO()
    decrypt_stream(
        input_stream=encrypted_state_stream, output_stream=decrypted_state_stream,
        passphrase=key, encryption=encryption_type
    )

    # Restore state dict from binary stream.
    deserialized_state_dict = pickle.loads(decrypted_state_stream.getvalue())

    # Overwrite the state.
    checkpoint["state_dict"] = deserialized_state_dict
    return checkpoint


def encrypt_checkpoint(checkpoint, key):
    """Function to encrypt checkpoint with supplied key.

    Args:
        checkpoint (dict): Dictionary checkpoint for pytorch.
        key (str): Key to encode the checkpoint state dict.

    Returns:
        checkpoint (dict): Encrypted checkpoint file.
    """
    # Get model state dict.
    state_dict = checkpoint["state_dict"]

    # Get a random encryption type
    encryption_type = get_random_encryption()
    checkpoint["state_dict_encrypted"] = encryption_type

    # Serialize it.
    serialized_state_stream = BytesIO(pickle.dumps(state_dict))

    # Encrypt it to binary steam.
    encrypted_state_stream = BytesIO()
    encrypt_stream(
        input_stream=serialized_state_stream, output_stream=encrypted_state_stream,
        passphrase=key, encryption=encryption_type
    )

    # Overwrite the state.
    checkpoint["state_dict"] = encrypted_state_stream.getvalue()

    return checkpoint
