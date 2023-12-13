# TAO ODISE1.1 (Developer Preview)

This repository is the optmized implementation of [ODISE](https://arxiv.org/abs/2303.04803) introduced in the paper with some architectural improvement:

  - [x] Replacing the Stable Diffusion backbone with a ConvNext based CLIP model 
  - [x] Optimizing the standalone inference pipeline
  - [x] Enabling the Hydra based configuration
  - [x] Supporting both caption and label supervision


[**Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models**](https://arxiv.org/abs/2303.04803)
[*Jiarui Xu*](https://jerryxu.net),
[*Sifei Liu**](https://research.nvidia.com/person/sifei-liu),
[*Arash Vahdat**](http://latentspace.cc/),
[*Wonmin Byeon*](https://wonmin-byeon.github.io/),
[*Xiaolong Wang*](https://xiaolonw.github.io/),
[*Shalini De Mello*](https://research.nvidia.com/person/shalini-de-mello)
CVPR 2023 Highlight. (*equal contribution)

**ODISE**: **O**pen-vocabulary **DI**ffusion-based panoptic **SE**gmentation exploits pre-trained text-image diffusion and discriminative models to perform open-vocabulary panoptic segmentation.
It leverages the frozen representation of both these models to perform panoptic segmentation of any category in the wild. 


## Environment Setup

Set up and launch an interative TAO development environment
(please check [Instantiating the development container](../../../README.md) for more details):
```bash
git clone https://github.com/NVIDIA/tao_pytorch_backend.git
source scripts/envsetup.sh
tao_pt --gpus all --volume /path/to/data/on/host:/path/to/data/on/container
```

Once inside the container, build the multi-scale deformable attention kernel:
```bash
cd /tao-pt/nvidia_tao_pytorch/cv/odise/modeling/pixel_decoder/ops/
sh make.sh
```

## Model Zoo

We provide the pre-trained model for TAO ODISE1.1 trained with label supervision on [COCO's](https://cocodataset.org/#home) entire training set.
ODISE's pre-trained models are subject to the [Creative Commons — Attribution-NonCommercial-ShareAlike 4.0 International — CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) terms.
The model contains 21M trainable parameters.
The download links for these models are provided in the table below.

<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" colspan="3">ADE20K(A-150)</th>
    <th align="center" style="text-align:center" colspan="3">COCO</th>
    <th align="center" style="text-align:center">download </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"></td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
  </tr>
  <tr>
    <td align="center"><a href="config/common/category_odise.py"> ODISE1.1 (label) </a></td>
    <td align="center">23.3</td>
    <td align="center">16.1</td>
    <td align="center">30.4</td>
    <td align="center">55.5</td>
    <td align="center">46.3</td>
    <td align="center">64.6</td>
    <td align="center"><a href="https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/odise">checkpoint</a></td>
  </tr>
</tbody>
</table>

## Get Started
See [Preparing Datasets for ODISE1.1](datasets/README.md).

See [Getting Started with ODISE1.1](GETTING_STARTED.md) for detailed instructions on training and inference with ODISE 1.1.

## Demo

* To run the ODISE1.1 [Gradio](https://github.com/gradio-app/gradio) demo locally:
    ```shell
    cd /tao-pt/nvidia_tao_pytorch/cv/odise/gradio
    python app.py -p $ODISE_CHECKPOINT
    ```
    By default, the Gradio app uses the hyperparameters from `config/common/category_odise.py` and the provided checkpoint file to run inference with the images uploaded via the web UI.

* To run ODISE1.1's demo from the command line:

    ```shell
    cd /tao-pt/nvidia_tao_pytorch/cv/odise/entrypoint
    python odise.py inference -e $YOUR_EXPERIMENT_SPEC
    ```
    Please refer to the `Experiment configuration file` in [Getting Started with ODISE1.1](GETTING_STARTED.md) to compose your own spec file. The inference pipeline takes a directory of images and a text prompt as inputs and outputs the annotated images.

## Visual Results

<div align="center">
<img src="gradio/examples/coco.jpg" width="32%">
<img src="gradio/examples/coco_result.jpg" width="32%">
</div>

## Citation

If you find our work useful in your research, please cite:

```BiBTeX
@article{xu2023odise,
  title={{Open-Vocabulary Panoptic Segmentation with Text-to-Image Diffusion Models}},
  author={Xu, Jiarui and Liu, Sifei and Vahdat, Arash and Byeon, Wonmin and Wang, Xiaolong and De Mello, Shalini},
  journal={arXiv preprint arXiv:2303.04803},
  year={2023}
}
```

## Acknowledgement

The code is largely based on [Detectron2](https://github.com/facebookresearch/detectron2), [Mask2Former](https://github.com/facebookresearch/Mask2Former) and [OpenCLIP](https://github.com/mlfoundations/open_clip).
