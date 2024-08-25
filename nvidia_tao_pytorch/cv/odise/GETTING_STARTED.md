## Getting Started with TAO ODISE1.1

This document provides a brief introduction on how to run the training, evaluation and inference pipelines with TAO ODISE1.1.

**Important Note**: Similar to other TAO models, we recommend to use the model entrypoint to launch training, evaluation and inference. 
```
# cd nvidia_tao_pytorch/cv/odise/entrypoint
python odise.py [train/evaluate/inference] -e $YOUR_EXPERIMENT_SPEC
```
Alternatively, each standalone pipeline script is available under `odise/scripts`

### Experiment configuration file

TAO ODISE1.1 supports two types of supervision methods, category based or caption based. The common configurations are stored in `config/common/category_odise.py` and `config/common/caption_odise.py` respectively following Detectron2's [Lazy Configs](https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html) format.

To further ease the experiment preparation and better bookkeeping, TAO ODISE1.1 introduces a [Hydra](https://hydra.cc/docs/intro/) configuration layer on top of the common settings. Most of the essential hyperparameters are extracted and listed in `config/default_config.py`. We can use a YAML spec file to update the common configurations. Here is an example:

```
results_dir: '/workspace/odise/'
model:
  type: "category"
  name: "convnext_large_d_320"
  pretrained_weights: "laion2b_s29b_b131k_ft_soup"
  num_classes: 133
train:
  use_amp: True
  checkpoint_interval: 5000
evaluate:
  checkpoint: "/workspace/odise/model_0089999.pth"
inference:
  precision: 'fp16'
  checkpoint: "/workspace/odise/model_0089999.pth"
  image_dir: "/tmp/odise_test_images"
  overlap_threshold: 0
  object_mask_threshold: 0.0
dataset:
  train:
    name: "coco_train_panoptic"
    root_dir: "/workspace/datasets"
    panoptic_json: "coco/annotations/panoptic_train2017.json"
    instance_json: "/workspace/datasets/coco/annotations/instances_train2017.json"
    instance_root: "/workspace/datasets/coco/train2017"
    panoptic_root: "coco/panoptic_train2017"
    semantic_root:  "coco/panoptic_semseg_train2017"
    prompt_eng_file: "/workspace/odise/coco_labels.txt"
  val:
    name: "coco_val_panoptic"
    root_dir: "/workspace/datasets"
    panoptic_json: "coco/annotations/panoptic_val2017.json"
    instance_json: "/workspace/datasets/coco/annotations/instances_val2017.json"
    instance_root: "/workspace/datasets/coco/val2017"
    panoptic_root: "coco/panoptic_val2017"
    semantic_root:  "coco/panoptic_semseg_val2017"
    prompt_eng_file: "/workspace/odise/coco_labels.txt"

```

### Training

To train a model with ODISE1.1 entrypoint, first prepare the datasets following the instructions in [datasets/README.md](./datasets/README.md) and update the `dataset` configs in the YAML spec file.

**Important note**: To use the Hydra config to register a custom dataset, the processed dataset must follow the COCO instance segmentation and panoptic segmentation formats. The `prompt_eng_file` contains mapping info between each class id and the prompt-engineered labels. See examples at [./data/datasets/openseg_labels](./data/datasets/openseg_labels/coco_panoptic_with_prompt_eng.txt)

```sh
python odise.py train -e $YOUR_EXPERIMENT_SPEC --gpus 8
```
By default, the checkpoints and training logs will be saved at `$results_dir/train`.


### Evaluation

To run the standalone evaluation with `dataset.val`.
```sh
python odise.py evaluate -e $YOUR_EXPERIMENT_SPEC --gpus 8 evaluate.checkpoint=$YOUR_CHECKPOINT
```
By default, the evaluation logs will be saved at `$results_dir/eval`.


### Inference

The standalone inference pipeline supports both category and caption ODISE models. Given a directory of images, `inference.label_set` or `inference.vocab`/`inference.caption`, the pipeline will save the annotated images at `inference.results_dir`. If `inference.results_dir` is not set, it will save the output images at `$results_dir/inference`

```sh
python odise.py inference -e $YOUR_EXPERIMENT_SPEC --gpus 8 inference.checkpoint=$YOUR_CHECKPOINT inference.vocab='sky;trees;road'
```
