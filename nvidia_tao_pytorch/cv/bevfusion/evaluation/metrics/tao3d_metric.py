# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from mmmdet3d. https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/

"""BEVFusion TAO3DMetric modules"""

from typing import Dict, List, Optional, Sequence, Tuple, Union
from os import path as osp
import tempfile
import numpy as np
import torch

import mmengine
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmdet3d.registry import METRICS
from mmdet3d.structures import Box3DMode

from nvidia_tao_pytorch.cv.bevfusion.evaluation import tao3d_eval
from nvidia_tao_pytorch.cv.bevfusion.structures import TAOLiDARInstance3DBoxes, TAOCameraInstance3DBoxes, project_cam2img
from nvidia_tao_pytorch.core import TAO_PYT_CACHE


@METRICS.register_module(force=True)
class TAO3DMetric(BaseMetric):
    """Tao3d evaluation metric."""

    def __init__(self,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 prefix: Optional[str] = None,
                 pklfile_prefix: Optional[str] = None,
                 default_cam_key: str = 'CAM0',
                 gt_box_type: str = 'camera',
                 yaw_dim: int = -1,
                 origin: Tuple[float, float, float] = (0.5, 1.0, 0.5),
                 is_synthetic: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 format_only: bool = False,
                 submission_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None,
                 collect_dir: str = osp.join(TAO_PYT_CACHE, 'metric')) -> None:
        """
        Initialization of TAO3DMetric
        Args:
        ann_file (str): Annotation file path.
        metric (str or List[str]): Metrics to be evaluated. Defaults to 'bbox'.

        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
        pklfile_prefix (str, optional): The prefix of pkl files, including the
            file path and the prefix of filename, e.g., "a/b/prefix". If not
            specified, a temp file will be created. Defaults to None.
        default_cam_key (str): The default camera for lidar to camera
            conversion. By default, tao3d: 'CAM2', Waymo: 'CAM_FRONT'.
            Defaults to 'CAM2'.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result to a
            specific format and submit it to the test server.
            Defaults to False.
        submission_prefix (str, optional): The prefix of submission data. If
            not specified, the submission data will not be generated.
            Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        """
        self.default_prefix = 'TAO3D metric'
        super().__init__(
            collect_device=collect_device, prefix=prefix, collect_dir=collect_dir)
        self.ann_file = ann_file
        self.pklfile_prefix = pklfile_prefix
        self.format_only = format_only
        self.gt_box_type = gt_box_type
        self.origin = origin
        self.yaw_dim = yaw_dim
        self.is_synthetic = is_synthetic
        self.pcd_limit_range = pcd_limit_range

        if self.gt_box_type not in ['camera', 'lidar']:
            raise KeyError("gt_box_type should be one of 'camera', 'lidar', "
                           f'but got {self.gt_box_type}.')

        if self.format_only:
            assert submission_prefix is not None, 'submission_prefix must be '
            'not None when format_only is True, otherwise the result files '
            'will be saved to a temp directory which will be cleaned up at '
            'the end.'

        self.submission_prefix = submission_prefix
        self.default_cam_key = default_cam_key
        self.backend_args = backend_args

        allowed_metrics = ['bbox', 'img_bbox', 'mAP', 'LET_mAP']
        self.metrics = metric if isinstance(metric, list) else [metric]
        for local_metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError("metric should be one of 'bbox', 'img_bbox', "
                               f'but got {local_metric}.')

    def convert_annos_to_tao3d_annos(self, data_infos: dict) -> List[dict]:
        """Convert loading annotations to tao3d annotations.

        Args:
            data_infos (dict): Data infos including metainfo and annotations
                loaded from ann_file.

        Returns:
            List[dict]: List of tao3d annotations.
        """
        data_annos = data_infos['data_list']
        if not self.format_only:
            cat2label = data_infos['metainfo']['categories']
            label2cat = {v: k for (k, v) in cat2label.items()}
            assert 'instances' in data_annos[0], 'data_annos must have instances'
            for i, annos in enumerate(data_annos):
                if len(annos['instances']) == 0:
                    tao3d_annos = {
                        'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation': np.zeros([0, 3]),
                        'score': np.array([]),
                    }
                else:
                    tao3d_annos = {
                        'name': [],
                        'truncated': [],
                        'occluded': [],
                        'alpha': [],
                        'bbox': [],
                        'location': [],
                        'dimensions': [],
                        'rotation': [],
                        'score': []
                    }

                    lidar2cam = np.array(annos['images'][self.default_cam_key]['lidar2cam']).astype(np.float32)

                    for instance in annos['instances']:
                        label = instance['bbox_label']
                        tao3d_annos['name'].append(label2cat[label])
                        tao3d_annos['truncated'].append(instance['truncated'])
                        tao3d_annos['occluded'].append(instance['occluded'])
                        tao3d_annos['alpha'].append(instance['alpha'])
                        tao3d_annos['bbox'].append(instance['bbox'])

                        if self.gt_box_type.lower() == 'lidar':
                            converted_bbox_3d = TAOLiDARInstance3DBoxes(np.array(instance['bbox_3d']).reshape(1, -1), origin=self.origin).convert_to(Box3DMode.CAM, np.transpose(lidar2cam), yaw_dim=self.yaw_dim).tensor.tolist()[0]
                        else:
                            converted_bbox_3d = TAOCameraInstance3DBoxes(np.array(instance['bbox_3d']).reshape(1, -1), origin=self.origin, is_synthetic=self.is_synthetic).tensor.tolist()[0]

                        tao3d_annos['location'].append(converted_bbox_3d[:3])
                        tao3d_annos['dimensions'].append(
                            converted_bbox_3d[3:6])
                        tao3d_annos['rotation'].append(
                            converted_bbox_3d[6:9])
                        tao3d_annos['score'].append(instance['score'])
                    for name in tao3d_annos:
                        tao3d_annos[name] = np.array(tao3d_annos[name])
                data_annos[i]['tao3d_annos'] = tao3d_annos
        return data_annos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = {}
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of the whole dataset.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self.classes = self.dataset_meta['classes']

        # load annotations
        pkl_infos = load(self.ann_file, backend_args=self.backend_args)

        self.data_infos = self.convert_annos_to_tao3d_annos(pkl_infos)
        result_dict, tmp_dir = self.format_results(
            results,
            pklfile_prefix=self.pklfile_prefix,
            submission_prefix=self.submission_prefix,
            classes=self.classes)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.dirname(self.submission_prefix)}')
            return metric_dict

        gt_annos = [
            self.data_infos[result['sample_idx']]['tao3d_annos']
            for result in results
        ]
        for metric in self.metrics:
            ap_dict = self.tao3d_evaluate(
                result_dict,
                gt_annos,
                metric=metric,
                logger=logger,
                classes=self.classes)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def tao3d_evaluate(self,
                       results_dict: dict,
                       gt_annos: List[dict],
                       metric: Optional[str] = None,
                       classes: Optional[List[str]] = None,
                       logger: Optional[MMLogger] = None) -> Dict[str, float]:
        """Evaluation in tao3d protocol.

        Args:
            results_dict (dict): Formatted results of the dataset.
            gt_annos (List[dict]): Contain gt information of each sample.
            metric (str, optional): Metrics to be evaluated. Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        ap_dict = {}
        for name in results_dict:
            if name == 'pred_instances' or metric == 'img_bbox':
                eval_types = ['bbox']
            else:
                eval_types = ['bbox', 'bev', '3d']

            ap_result_str, ap_dict_ = tao3d_eval(
                gt_annos, results_dict[name], classes, eval_types=eval_types)
            for ap_type, ap in ap_dict_.items():
                ap_dict[f'{name}/{ap_type}'] = float(f'{ap:.4f}')

            print_log(f'Results of {name}:\n' + ap_result_str, logger=logger)

        return ap_dict

    def format_results(
        self,
        results: List[dict],
        pklfile_prefix: Optional[str] = None,
        submission_prefix: Optional[str] = None,
        classes: Optional[List[str]] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the results to pkl file.

        Args:
            results (List[dict]): Testing results of the dataset.
            pklfile_prefix (str, optional): The prefix of pkl files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submitted files.
                It includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.
            classes (List[str], optional): A list of class name.
                Defaults to None.

        Returns:
            tuple: (result_dict, tmp_dir), result_dict is a dict containing the
            formatted result, tmp_dir is the temporal directory created for
            saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = {}
        sample_idx_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if submission_prefix is not None:
                submission_prefix_ = osp.join(submission_prefix, name)
            else:
                submission_prefix_ = None
            if pklfile_prefix is not None:
                pklfile_prefix_ = osp.join(pklfile_prefix, name) + '.pkl'
            else:
                pklfile_prefix_ = None
            if 'pred_instances' in name and '3d' in name and name[
                    0] != '_' and results[0][name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_tao3d(net_outputs,
                                                      sample_idx_list, classes,
                                                      pklfile_prefix_,
                                                      submission_prefix_)
                result_dict[name] = result_list_
            elif name == 'pred_instances' and name[0] != '_' and results[0][
                    name]:
                net_outputs = [result[name] for result in results]
                result_list_ = self.bbox2result_tao3d2d(
                    net_outputs, sample_idx_list, classes, pklfile_prefix_,
                    submission_prefix_)
                result_dict[name] = result_list_
        return result_dict, tmp_dir

    def bbox2result_tao3d(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 3D detection results to tao3d format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the tao3d format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmengine.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting 3D prediction to tao3d format')
        for idx, pred_dicts in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            sample_idx = sample_idx_list[idx]
            info = self.data_infos[sample_idx]
            # Here default used 'CAM2' to compute metric. If you want to
            # use another camera, please modify it.
            image_shape = (info['images'][self.default_cam_key]['height'],
                           info['images'][self.default_cam_key]['width'])
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation': [],
                'score': []
            }
            if len(box_dict['bbox']) > 0:
                box_2d_preds = box_dict['bbox']
                box_preds = box_dict['box3d_camera']
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']
                for box_camera, _, bbox, score, label in zip(box_preds, box_preds_lidar, box_2d_preds, scores, label_preds):
                    bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                    bbox[:2] = np.maximum(bbox[:2], [0, 0])
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(0.0)
                    anno['bbox'].append(bbox)

                    anno['location'].append(box_camera[:3])
                    anno['dimensions'].append(box_camera[3:6])
                    anno['rotation'].append(box_camera[6:9])
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation': np.zeros([0, 3]),
                    'score': np.array([]),
                }

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w', encoding="utf-8") as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']
                    rot = anno['rotation']
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], rot[idx][0], rot[idx][1], rot[idx][2],
                                anno['score'][idx]),
                            file=f)

            anno['sample_idx'] = np.array(
                [sample_idx] * len(anno['score']), dtype=np.int64)

            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def bbox2result_tao3d2d(
            self,
            net_outputs: List[dict],
            sample_idx_list: List[int],
            class_names: List[str],
            pklfile_prefix: Optional[str] = None,
            submission_prefix: Optional[str] = None) -> List[dict]:
        """Convert 2D detection results to tao3d format for evaluation and test
        submission.

        Args:
            net_outputs (List[dict]): List of dict storing the inferenced
                bounding boxes and scores.
            sample_idx_list (List[int]): List of input sample idx.
            class_names (List[str]): A list of class names.
            pklfile_prefix (str, optional): The prefix of pkl file.
                Defaults to None.
            submission_prefix (str, optional): The prefix of submission file.
                Defaults to None.

        Returns:
            List[dict]: A list of dictionaries with the tao3d format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        det_annos = []
        print('\nConverting 2D prediction to tao3d format')
        for i, bboxes_per_sample in enumerate(
                mmengine.track_iter_progress(net_outputs)):
            anno = {'name': [],
                    'truncated': [],
                    'occluded': [],
                    'alpha': [],
                    'bbox': [],
                    'dimensions': [],
                    'location': [],
                    'rotation': [],
                    'score': []}
            sample_idx = sample_idx_list[i]

            num_example = 0
            bbox = bboxes_per_sample['bboxes']
            for i in range(bbox.shape[0]):
                anno['name'].append(class_names[int(
                    bboxes_per_sample['labels'][i])])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(0.0)
                anno['bbox'].append(bbox[i, :4])
                # set dimensions (height, width, length) to zero
                anno['dimensions'].append(
                    np.zeros(shape=[3], dtype=np.float32))
                # set the 3D translation to (-1000, -1000, -1000)
                anno['location'].append(
                    np.ones(shape=[3], dtype=np.float32) * (-1000.0))
                # set rotation (Rx, Ry, Rz) to zero
                anno['rotation'].append(np.zeros(shape=[3], dtype=np.float32))
                anno['score'].append(bboxes_per_sample['scores'][i])
                num_example += 1

            if num_example == 0:
                anno = {'name': np.array([]),
                        'truncated': np.array([]),
                        'occluded': np.array([]),
                        'alpha': np.array([]),
                        'bbox': np.zeros([0, 4]),
                        'dimensions': np.zeros([0, 3]),
                        'location': np.zeros([0, 3]),
                        'rotation': np.zeros([0, 3]),
                        'score': np.array([]),
                        }
            else:
                anno = {k: np.stack(v) for k, v in anno.items()}

            anno['sample_idx'] = np.array(
                [sample_idx] * num_example, dtype=np.int64)
            det_annos.append(anno)

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            else:
                out = pklfile_prefix
            mmengine.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        if submission_prefix is not None:
            # save file in submission format
            mmengine.mkdir_or_exist(submission_prefix)
            print(f'Saving tao3d submission to {submission_prefix}')
            for i, anno in enumerate(det_annos):
                sample_idx = sample_idx_list[i]
                cur_det_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(cur_det_file, 'w', encoding="utf-8") as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']
                    rot = anno['rotation']
                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} '
                            '{:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f} {:4f}'.format(
                                anno['name'][idx],
                                anno['alpha'][idx],
                                *bbox[idx],  # 4 float
                                *dims[idx],  # 3 float
                                *loc[idx],  # 3 float
                                *rot[idx],  # 3 float
                                anno['score'][idx]),
                            file=f,
                        )
            print(f'Result is saved to {submission_prefix}')

        return det_annos

    def convert_valid_bboxes(self, box_dict: dict, info: dict) -> dict:
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - bboxes_3d (:obj:`BaseInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (Tensor): Scores of boxes.
                - labels_3d (Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

            - bbox (np.ndarray): 2D bounding boxes.
            - box3d_camera (np.ndarray): 3D bounding boxes in
              camera coordinate.
            - box3d_lidar (np.ndarray): 3D bounding boxes in
              LiDAR coordinate.
            - scores (np.ndarray): Scores of boxes.
            - label_preds (np.ndarray): Class label predictions.
            - sample_idx (int): Sample index.
        """
        box_preds = box_dict['bboxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['sample_idx']

        if len(box_preds) == 0:
            return {'bbox': np.zeros([0, 4]),
                    'box3d_camera': np.zeros([0, 9]),
                    'box3d_lidar': np.zeros([0, 9]),
                    'scores': np.zeros([0]),
                    'label_preds': np.zeros([0, 1]),
                    'sample_idx': sample_idx}
        # Used self.default_cam_key
        lidar2cam = np.array(
            info['images'][self.default_cam_key]['lidar2cam']).astype(
                np.float32)
        cam2img = np.array(info['images'][self.default_cam_key]['cam2img']).astype(
            np.float32)
        img_shape = (info['images'][self.default_cam_key]['height'],
                     info['images'][self.default_cam_key]['width'])
        cam2img = box_preds.tensor.new_tensor(cam2img)

        if isinstance(box_preds, TAOLiDARInstance3DBoxes):
            if self.yaw_dim != -1:
                box_preds_camera = box_preds.convert_to(Box3DMode.CAM, np.transpose(lidar2cam), yaw_dim=2)
            else:
                box_preds_camera = box_preds.convert_to(Box3DMode.CAM, np.transpose(lidar2cam), yaw_dim=self.yaw_dim)
            box_preds_lidar = box_preds
        elif isinstance(box_preds, TAOCameraInstance3DBoxes):
            if self.yaw_dim != -1:
                box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                                       np.transpose(np.linalg.inv(lidar2cam)), yaw_dim=1)
            else:
                box_preds_lidar = box_preds.convert_to(Box3DMode.LIDAR,
                                                       np.transpose(np.linalg.inv(lidar2cam)), yaw_dim=self.yaw_dim)
            box_preds_camera = box_preds

        box_corners = box_preds_camera.corners
        num_bbox = box_corners.shape[0]
        points_3d = box_corners.reshape(-1, 3)

        box_corners_in_image = project_cam2img(points_3d, cam2img.t())  # (num_bbox*8, 3) -> (num_bbox*8, 2)
        box_corners_in_image = box_corners_in_image.reshape(num_bbox, 8, 2)

        # box_corners_in_image: [N, 8, 2]
        minxy = torch.min(box_corners_in_image, dim=1)[0]
        maxxy = torch.max(box_corners_in_image, dim=1)[0]
        box_2d_preds = torch.cat([minxy, maxxy], dim=1)

        # Post-processing
        # check box_preds_camera
        image_shape = box_preds.tensor.new_tensor(img_shape)
        valid_cam_inds = ((box_2d_preds[:, 0] < image_shape[1]) &
                          (box_2d_preds[:, 1] < image_shape[0]) &
                          (box_2d_preds[:, 2] > 0) & (box_2d_preds[:, 3] > 0))

        if isinstance(box_preds, TAOCameraInstance3DBoxes):
            limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
            valid_pcd_inds = ((box_preds_lidar.center > limit_range[:3]) &
                              (box_preds_lidar.center < limit_range[3:]))
            valid_inds = valid_cam_inds & valid_pcd_inds.all(-1)
        else:
            valid_inds = valid_cam_inds

        # noqa pylint: disable=R1705
        if valid_inds.sum() > 0:
            return {
                'bbox': box_2d_preds[valid_inds, :].numpy(),
                'pred_box_type_3d': type(box_preds),
                'box3d_camera': box_preds_camera[valid_inds].numpy(),
                'box3d_lidar': box_preds_lidar[valid_inds].numpy(),
                'scores': scores[valid_inds].numpy(),
                'label_preds': labels[valid_inds].numpy(),
                'sample_idx': sample_idx
            }
        else:
            return {
                'bbox': np.zeros([0, 4]),
                'pred_box_type_3d': type(box_preds),
                'box3d_camera': np.zeros([0, 9]),
                'box3d_lidar': np.zeros([0, 9]),
                'scores': np.zeros([0]),
                'label_preds': np.zeros([0]),
                'sample_idx': sample_idx
            }
