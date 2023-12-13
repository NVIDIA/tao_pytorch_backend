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
Evaluate OCRNet script.
"""
import os
import argparse

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig


def test(opt):
    """Test the OCRNet according to option."""
    # @TODO(tylerz): Lazy import for correctly setting CUDA_VISIBLE_DEVICES
    import torch
    import torch.utils.data

    from nvidia_tao_pytorch.cv.ocrnet.utils.utils import (CTCLabelConverter,
                                                          AttnLabelConverter,
                                                          validation, load_checkpoint)
    from nvidia_tao_pytorch.cv.ocrnet.dataloader.ocr_dataset import LmdbDataset, RawGTDataset, AlignCollateVal
    from nvidia_tao_pytorch.cv.ocrnet.model.model import Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    ckpt = load_checkpoint(opt.saved_model, key=opt.encryption_key, to_cpu=True)
    if not isinstance(ckpt, Model):
        model = Model(opt)
        model = torch.nn.DataParallel(model).to(device)
        state_dict = ckpt
        model.load_state_dict(state_dict, strict=True)
    else:
        model = torch.nn.DataParallel(ckpt).to(device)
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0

    """ evaluation """
    model.eval()
    with torch.no_grad():
        log = open(f'{opt.exp_name}/log_evaluation.txt', 'a')
        AlignCollate_evaluation = AlignCollateVal(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        if opt.eval_gt_file is None:
            eval_data = LmdbDataset(opt.eval_data, opt)
        else:
            eval_data = RawGTDataset(gt_file=opt.eval_gt_file, img_dir=opt.eval_data, opt=opt)
        eval_data_log = f"data directory:\t{opt.eval_data}\t num samples: {len(eval_data)}"
        print(eval_data_log)
        evaluation_loader = torch.utils.data.DataLoader(
            eval_data, batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_evaluation, pin_memory=True)
        _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
            model, criterion, evaluation_loader, converter, opt)
        log.write(eval_data_log)
        print(f'Accuracy: {accuracy_by_best_model:0.3f}')
        log.write(f'{accuracy_by_best_model:0.3f}\n')
        log.close()


def init_configs(experiment_spec: ExperimentConfig):
    """Pass the yaml config to argparse.Namespace"""
    parser = argparse.ArgumentParser()

    opt, _ = parser.parse_known_args()
    if experiment_spec.evaluate.results_dir is not None:
        results_dir = experiment_spec.evaluate.results_dir
    else:
        results_dir = os.path.join(experiment_spec.results_dir, "evaluate")
        experiment_spec.evaluate.results_dir = results_dir

    opt.exp_name = results_dir
    opt.encryption_key = experiment_spec.encryption_key
    opt.eval_data = experiment_spec.evaluate.test_dataset_dir
    opt.eval_gt_file = experiment_spec.evaluate.test_dataset_gt_file

    # 1. Init dataset params
    dataset_config = experiment_spec.dataset
    model_config = experiment_spec.model
    opt.batch_max_length = dataset_config.max_label_length
    opt.imgH = model_config.input_height
    opt.imgW = model_config.input_width
    opt.input_channel = model_config.input_channel
    if model_config.input_channel == 3:
        opt.rgb = True
    else:
        opt.rgb = False

    if dataset_config.augmentation.keep_aspect_ratio:
        opt.PAD = True
    else:
        opt.PAD = False

    # load character list:
    # Don't convert the characters to lower case
    with open(dataset_config.character_list_file, "r") as f:
        characters = "".join([ch.strip() for ch in f.readlines()])
    opt.character = characters

    # TODO(tylerz): hardcode the data_filtering_off to be True.
    # And there will be KeyError when encoding the labels
    opt.data_filtering_off = True

    opt.workers = dataset_config.workers
    opt.batch_size = experiment_spec.evaluate.batch_size

    # 2. Init Model params
    opt.saved_model = experiment_spec.evaluate.checkpoint
    if model_config.TPS:
        opt.Transformation = "TPS"
    else:
        opt.Transformation = "None"

    opt.FeatureExtraction = model_config.backbone
    opt.SequenceModeling = model_config.sequence
    opt.Prediction = model_config.prediction
    opt.num_fiducial = model_config.num_fiducial
    opt.output_channel = model_config.feature_channel
    opt.hidden_size = model_config.hidden_size

    if model_config.quantize:
        opt.quantize = True
    else:
        opt.quantize = False
    opt.baiduCTC = False

    # 4. Init for Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(experiment_spec.evaluate.gpu_id)
    import torch
    opt.num_gpu = torch.cuda.device_count()

    return opt


def run_experiment(experiment_spec):
    """run experiment."""
    opt = init_configs(experiment_spec)
    os.makedirs(f'{opt.exp_name}', exist_ok=True)

    # Set status logging
    status_file = os.path.join(opt.exp_name, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file,
                                                                 append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting OCRNet evaluation"
    )

    test(opt)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Load experiment specification, additially using schema for validation/retrieving the default values.
# --config_path and --config_name will be provided by the entrypoint script.


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"), config_name="experiment", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Run the training process."""
    try:
        run_experiment(experiment_spec=cfg)
        status_logging.get_status_logger().write(
            status_level=status_logging.Status.SUCCESS,
            message="Evaluation finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Evaluation was interrupted",
            verbosity_level=status_logging.Verbosity.INFO,
            status_level=status_logging.Status.FAILURE
        )
    except Exception as e:
        status_logging.get_status_logger().write(
            message=str(e),
            status_level=status_logging.Status.FAILURE
        )
        raise e


if __name__ == '__main__':
    main()
