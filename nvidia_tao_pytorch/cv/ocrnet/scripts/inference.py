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
Inference OCRNet script.
"""
import argparse
import os
from tabulate import tabulate

import nvidia_tao_pytorch.core.loggers.api_logging as status_logging
from nvidia_tao_pytorch.core.hydra.hydra_runner import hydra_runner
from nvidia_tao_pytorch.cv.ocrnet.config.default_config import ExperimentConfig


def inference(opt):
    """Inference on the OCRNet according to option"""
    # @TODO(tylerz): Lazy import for correctly setting CUDA_VISIBLE_DEVICES
    import torch
    import torch.utils.data
    import torch.nn.functional as F

    from nvidia_tao_pytorch.cv.ocrnet.utils.utils import CTCLabelConverter, AttnLabelConverter, load_checkpoint
    from nvidia_tao_pytorch.cv.ocrnet.dataloader.ocr_dataset import RawDataset, AlignCollateVal
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
        state_dict = ckpt
        model = Model(opt)
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(state_dict)
    else:
        model = torch.nn.DataParallel(ckpt).to(device)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollateVal(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    table_header = ["image_path", "predicted_labels", "confidence score"]
    table_data = []
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                table_data.append((img_name, pred, f"{confidence_score:0.4f}"))
        print(tabulate(table_data, headers=table_header, tablefmt="psql"))


def init_configs(experiment_spec: ExperimentConfig):
    """Pass the yaml config to argparse.Namespace"""
    parser = argparse.ArgumentParser()

    opt, _ = parser.parse_known_args()
    opt.encryption_key = experiment_spec.encryption_key
    opt.image_folder = experiment_spec.inference.inference_dataset_dir

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

    opt.workers = dataset_config.workers
    opt.batch_size = experiment_spec.inference.batch_size

    # 2. Init Model params
    opt.saved_model = experiment_spec.inference.checkpoint
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

    opt.baiduCTC = False

    # 4. Init for Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(experiment_spec.inference.gpu_id)
    import torch
    opt.num_gpu = torch.cuda.device_count()

    return opt


def run_experiment(experiment_spec):
    """run experiment."""
    opt = init_configs(experiment_spec)
    # Set status logging
    if experiment_spec.inference.results_dir is not None:
        results_dir = experiment_spec.inference.results_dir
    else:
        results_dir = os.path.join(experiment_spec.results_dir, "inference")
        experiment_spec.inference.results_dir = results_dir
    os.makedirs(results_dir, exist_ok=True)
    status_file = os.path.join(results_dir, "status.json")
    status_logging.set_status_logger(status_logging.StatusLogger(filename=status_file,
                                                                 append=True))
    status_logging.get_status_logger().write(
        status_level=status_logging.Status.STARTED,
        message="Starting OCRNet inference"
    )

    inference(opt)


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
            message="Inference finished successfully."
        )
    except (KeyboardInterrupt, SystemExit):
        status_logging.get_status_logger().write(
            message="Inference was interrupted",
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
