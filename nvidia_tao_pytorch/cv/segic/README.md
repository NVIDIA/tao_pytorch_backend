# TAO SegIC 1.0 (Developer Preview)

This repository supports gradio demo of [SegIC](https://arxiv.org/abs/2311.14671) introduced, optimized by TensorRT.

[**SEGIC: Unleashing the Emergent Correspondence for In-Context Segmentation**](https://arxiv.org/abs/2311.14671)
*Lingchen Meng*,
*Shiyi Lan*,
*Hengduo Li*,
*Jose M. Alvarez*,
*Zuxuan Wu*,
*Yu-Gang Jiang*

**SEGIC**: The SegIC (**Seg**ment-**I**n-**C**ontext) model is designed for in-context segmentation, which aims to segment novel images using a few labeled example images (in-context examples). The SegIC shows powerful performance as a generalist model for segmenting everything in context. It achieves SOTA results in [COCO-20i](https://arxiv.org/pdf/1909.13140), [FSS-1000](https://github.com/HKUSTCV/FSS-1000) and recent [LVIS-92i](https://docs.ultralytics.com/datasets/detect/lvis/). 


## Environment Setup

Set up and launch an interative TAO development environment
(please check [Instantiating the development container](../../../README.md) for more details):
```bash
git clone https://github.com/NVIDIA/tao_pytorch_backend.git
source scripts/envsetup.sh
tao_pt --gpus all --volume /path/to/data/on/host:/path/to/data/on/container
```

## Demo

To run the NVIDIA SegIC 1.0 [Gradio](https://github.com/gradio-app/gradio) demo locally:

1. Download `segic_deployable_v1.0.onnx` and `prompt_feature_extract_deployable_v1.0.onnx` from [NGC](TODO: model NGC address) to `$MODEL_DIR`

2. Convert the deployable model files to TensorRT (TRT) engines and launch the Gradio demo.
    ```shell
    cd /tao-pt/nvidia_tao_pytorch/cv/segic/gradio
    python app.py -o $MODEL_DIR/segic_deployable_v1.0.onnx -op $MODEL_DIR/prompt_feature_extract_deployable_v1.0.onnx
    ```
    This script would generate the TRT engines `segic.trt` and `prompt_feature_extract.trt` before launching the Gradio demo. By default, the Gradio app runs inference with the images uploaded via the web UI.

3. (Optiontal) If you already have the TRT engines, you can launch the Gradio demo with TRT engine files
    ```shell
    cd /tao-pt/nvidia_tao_pytorch/cv/segic/gradio
    python app.py -t segic.trt -tp prompt_feature_extract.trt
    ```

## Visual Results

<div align="center">
<img src="gradio/examples/segic_demo.drawio.png" width="100%">
</div>

## Citation

```BiBTeX
@article{meng2023segic,
  title={SegIC: Unleashing the Emergent Correspondence for In-Context Segmentation},
  author={Meng, Lingchen and Lan, Shiyi and Li, Hengduo and Alvarez, Jose M and Wu, Zuxuan and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2311.14671},
  year={2023},
  url={https://arxiv.org/abs/2311.14671}
}

```

## Acknowledgement

Thanks to the help from SegIC paper author [Lingchen Meng](https://github.com/MengLcool).
