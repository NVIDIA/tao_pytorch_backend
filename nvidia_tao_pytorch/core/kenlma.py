#!/usr/bin/env python3

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

"""

Abstraction over EFF/cookbooks stuff
We only deal in file handles through cli.
We want to be able to open an archive and pass around its interal file handles pointing to the
pieces in /tmp without reading and rewriting the contents in this process.

"""
import tempfile
from nvidia_tao_pytorch.core.cookbooks.kenlm_cookbook import KenLMCookbook
from eff.core import Archive
import shutil
import os

INTERMEDIATE = "intermediate.kenlm_intermediate"
VOCABULARY = "vocabulary.txt"
BINARY = "kenlm.arpa"


class KenLMArchive():
    """Implementation of a KenLM EFF archive model file."""

    @classmethod
    def dumpLMA(cls, archive_path, key, binary=None, vocabulary=None, intermediate=None):
        """Create eff archive for language model.

        Args:
            archive_path (str): Path to the eff archive.
            binary (str): Path to the binary file.
            vocabulary (str): Path to the vocabulary file.
            intermediate (str): Path to the intermediate file.
        """
        cb = KenLMCookbook()
        cb.set_passphrase(key)

        if intermediate:
            with open(intermediate, "rb") as f:
                c = f.read()
            cb.add_class_file_content(name=INTERMEDIATE,
                                      content=c,
                                      description="KenLM intermediate format")

        if vocabulary:
            with open(vocabulary, "r") as f:
                c = f.read()
            cb.add_class_file_content(name=VOCABULARY,
                                      content=c,
                                      description="KenLM vocabulary file")
        with open(binary, "rb") as f:
            c = f.read()

        cb.add_class_file_content(name=BINARY,
                                  content=c,
                                  description="KenLM binary .arpa",
                                  binary=True)

        cb.save(archive_path)

        return cls(archive_path, key)

    def __init__(self, archive_path, key):
        """Setup archive_path and key.

        Args:
            archive_path (str): Path to the KenLM archive object.
            key (str): Passphrase to encrypt the model file.
        """
        self._archive_path = archive_path
        self._key = key

    def open(self):
        """
        Restore a kenLM archive model file.

        Underlying EFF drops bits into tmpdirs.
        this moves those into a tmpdir our object controls
        """
        self._tmpdir = tempfile.TemporaryDirectory()
        current_dir = self._tmpdir.name

        with Archive.restore_from(
                restore_path=self._archive_path,
                passphrase=self._key
        ) as archive:
            for artifact in archive['artifacts'].keys():
                fname = archive['artifacts'][artifact].get_handle()
                shutil.copyfile(fname, os.path.join(current_dir, artifact))

    def get_tmpdir(self):
        """Return path to the temp directory.

        Args:
            self._tmpdir.name (str): Path to the temp directory.
        """
        return self._tmpdir.name

    def get_intermediate(self):
        """Return path of INTERMEDIATE file."""
        return os.path.join(self._tmpdir.name, INTERMEDIATE)

    def get_vocabulary(self):
        """Return path of VOCABULARY file."""
        return os.path.join(self._tmpdir.name, VOCABULARY)

    def get_binary(self):
        """Return path of BINARY file."""
        return os.path.join(self._tmpdir.name, BINARY)
