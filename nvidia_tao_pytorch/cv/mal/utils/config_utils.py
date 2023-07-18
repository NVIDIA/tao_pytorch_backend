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

"""Utils for configuration."""
import logging
import os
logger = logging.getLogger(__name__)


def update_config(cfg, task):
    """Update config parameters.

    This function should be called at the beginning of a pipeline script.
    Global results_dir will be updated based on task.results_dir

    Args:
        cfg (Hydra config): Config object loaded by Hydra
        task (str): TAO pipeline name
    Return:
        Updated cfg
    """
    # mask threshold
    if len(cfg.train.mask_thres) == 1:
        # this means to repeat the same threshold three times
        # all scale objects are sharing the same threshold
        cfg.train.mask_thres = [cfg.train.mask_thres[0] for _ in range(3)]
    assert len(cfg.train.mask_thres) == 3, "Length of mask thresholds must be 1 or 3."

    # frozen_stages
    if len(cfg.model.frozen_stages) == 1:
        cfg.model.frozen_stages = [0, cfg.model.frozen_stages[0]]
    assert len(cfg.model.frozen_stages) == 2, "Length of frozen stages must be 1 or 2."
    assert len(cfg.train.margin_rate) == 2, "Length of margin rate must be 2."

    if cfg[task]['results_dir']:
        cfg.results_dir = cfg[task]['results_dir']
    else:
        cfg.results_dir = os.path.join(cfg.results_dir, task)
        cfg[task]['results_dir'] = cfg.results_dir
    logger.info(f"{task.capitalize()} results will be saved at: %s", cfg.results_dir)
    return cfg
