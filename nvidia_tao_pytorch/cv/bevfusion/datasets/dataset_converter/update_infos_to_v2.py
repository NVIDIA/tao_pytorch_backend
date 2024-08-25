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

""" BEVFusion Data Convert the annotation pkl to the standard format in OpenMMLab V2.0."""

import copy
import time
from os import path as osp
from pathlib import Path
import numpy as np

import mmengine
from mmdet3d.datasets.convert_utils import get_kitti_style_2d_boxes
from mmdet3d.structures import points_cam2img


def generate_kitti_camera_instances(ori_info_dict):
    """ generate_kitti_camera_instances """
    cam_key = 'CAM2'
    empty_camera_instances = get_empty_multicamera_instances([cam_key])
    annos = copy.deepcopy(ori_info_dict['annos'])
    ann_infos = get_kitti_style_2d_boxes(
        ori_info_dict, occluded=[0, 1, 2, 3], annos=annos)
    empty_camera_instances[cam_key] = ann_infos

    return empty_camera_instances


def update_tao3d_infos(pkl_path, out_dir):
    """ update tao3d into mmlab v2 version """
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    # TODO update to full label
    METAINFO = {
        'classes': ('person',),
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []

    for i, ori_info_dict in enumerate(mmengine.track_iter_progress(data_list)):
        temp_data_info = get_empty_standard_data_info()
        temp_data_info['sample_idx'] = i

        image_name = Path(ori_info_dict['image']['image_path'])

        temp_data_info['images']['CAM0']['img_path'] = osp.join(image_name.parents[0].stem, image_name.name)
        h, w = ori_info_dict['image']['image_shape']
        temp_data_info['images']['CAM0']['height'] = h
        temp_data_info['images']['CAM0']['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        lidar_name = Path(ori_info_dict['point_cloud']['lidar_path'])
        temp_data_info['lidar_points']['lidar_path'] = osp.join(lidar_name.parents[0].stem, lidar_name.name)

        # meta info has right-hand matrices for code compatibility
        cam2img = ori_info_dict['calib']['cam2img'].astype(np.float32)

        lidar2cam = ori_info_dict['calib']['lidar2cam'].astype(np.float32)
        lidar2img = lidar2cam @ cam2img
        temp_data_info['images']['CAM0']['cam2img'] = np.transpose(cam2img).tolist()
        temp_data_info['images']['CAM0']['lidar2cam'] = np.transpose(lidar2cam).tolist()
        temp_data_info['images']['CAM0']['lidar2img'] = np.transpose(lidar2img).tolist()

        temp_data_info['lidar_points']['lidar2cam'] = temp_data_info['images']['CAM0']['lidar2cam']

        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        empty_flag = False
        if anns is not None:
            num_instances = len(anns['name'])
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                name = (anns['name'][instance_id]).lower()
                if name in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(name)
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    continue

                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                loc = anns['location'][instance_id]
                dims = anns['dimensions'][instance_id]
                rots = anns['rotation'][instance_id]
                gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()
                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                empty_instance['truncated'] = anns['truncated'][
                    instance_id].tolist()
                empty_instance['occluded'] = anns['occluded'][
                    instance_id].tolist()
                empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
                empty_instance['score'] = anns['score'][instance_id].tolist()
                empty_instance['index'] = anns['index'][instance_id].tolist()
                empty_instance['group_id'] = anns['group_ids'][
                    instance_id].tolist()
                empty_instance['difficulty'] = anns['difficulty'][
                    instance_id].tolist()
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            if len(instance_list) == 0:
                empty_flag = True
                continue
            temp_data_info['instances'] = instance_list

        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        # converted_list.append(temp_data_info)
        if not empty_flag:
            converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = {}
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}

    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'tao3d'
    metainfo['info_version'] = '1.0'
    converted_data_info = {'metainfo': metainfo, 'data_list': converted_list}

    mmengine.dump(converted_data_info, out_path, 'pkl')


def update_kitti_infos(pkl_path, out_dir):
    """ update kitti into mmlab v2 version """
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting '
              f'the original data {pkl_path}.')
        time.sleep(5)
    METAINFO = {
        'classes': ('person',),
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    for ori_info_dict in mmengine.track_iter_progress(data_list):
        temp_data_info = get_empty_standard_data_info(camera_types=['CAM2'])

        if 'plane' in ori_info_dict:
            temp_data_info['plane'] = ori_info_dict['plane']

        temp_data_info['sample_idx'] = ori_info_dict['image']['image_idx']
        temp_data_info['images']['CAM2']['cam2img'] = ori_info_dict['calib'][
            'P2'].tolist()

        temp_data_info['images']['CAM2']['img_path'] = Path(
            ori_info_dict['image']['image_path']).name
        h, w = ori_info_dict['image']['image_shape']
        temp_data_info['images']['CAM2']['height'] = h
        temp_data_info['images']['CAM2']['width'] = w
        temp_data_info['lidar_points']['num_pts_feats'] = ori_info_dict[
            'point_cloud']['num_features']
        temp_data_info['lidar_points']['lidar_path'] = Path(
            ori_info_dict['point_cloud']['velodyne_path']).name

        rect = ori_info_dict['calib']['R0_rect'].astype(np.float32)
        Trv2c = ori_info_dict['calib']['Tr_velo_to_cam'].astype(np.float32)
        lidar2cam = rect @ Trv2c
        temp_data_info['images']['CAM2']['lidar2cam'] = lidar2cam.tolist()
        temp_data_info['images']['CAM2']['lidar2img'] = (
            ori_info_dict['calib']['P2'] @ lidar2cam).tolist()

        temp_data_info['lidar_points']['Tr_velo_to_cam'] = Trv2c.tolist()

        # for potential usage
        temp_data_info['images']['R0_rect'] = ori_info_dict['calib'][
            'R0_rect'].astype(np.float32).tolist()
        temp_data_info['lidar_points']['Tr_imu_to_velo'] = ori_info_dict[
            'calib']['Tr_imu_to_velo'].astype(np.float32).tolist()

        cam2img = ori_info_dict['calib']['P2']

        anns = ori_info_dict.get('annos', None)
        ignore_class_name = set()
        empty_flag = False
        if anns is not None:
            num_instances = len(anns['name'])
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()

                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label'] = METAINFO['classes'].index(
                        anns['name'][instance_id])
                elif anns['name'][instance_id] in ('Pedestrian'):
                    empty_instance['bbox_label'] = METAINFO['classes'].index('person')
                else:
                    ignore_class_name.add(anns['name'][instance_id])
                    continue

                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()

                loc = anns['location'][instance_id]
                dims = anns['dimensions'][instance_id]
                rots = anns['rotation_y'][:, None][instance_id]

                dst = np.array([0.5, 0.5, 0.5])
                src = np.array([0.5, 1.0, 0.5])

                center_3d = loc + dims * (dst - src)
                center_2d = points_cam2img(
                    center_3d.reshape([1, 3]), cam2img, with_depth=True)
                center_2d = center_2d.squeeze().tolist()
                empty_instance['center_2d'] = center_2d[:2]
                empty_instance['depth'] = center_2d[2]

                gt_bboxes_3d = np.concatenate([loc, dims, [0.0], rots, [0.0]]).tolist()
                # gt_bboxes_3d = np.concatenate([loc, dims, rots]).tolist()

                empty_instance['bbox_3d'] = gt_bboxes_3d
                empty_instance['bbox_label_3d'] = copy.deepcopy(
                    empty_instance['bbox_label'])
                empty_instance['bbox'] = anns['bbox'][instance_id].tolist()
                empty_instance['truncated'] = anns['truncated'][
                    instance_id].tolist()
                empty_instance['occluded'] = anns['occluded'][
                    instance_id].tolist()
                empty_instance['alpha'] = anns['alpha'][instance_id].tolist()
                empty_instance['score'] = anns['score'][instance_id].tolist()
                empty_instance['index'] = anns['index'][instance_id].tolist()
                empty_instance['group_id'] = anns['group_ids'][
                    instance_id].tolist()
                empty_instance['difficulty'] = anns['difficulty'][
                    instance_id].tolist()
                empty_instance['num_lidar_pts'] = anns['num_points_in_gt'][
                    instance_id].tolist()
                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            if len(instance_list) == 0:
                empty_flag = True
                continue
            temp_data_info['instances'] = instance_list
            cam_instances = generate_kitti_camera_instances(ori_info_dict)
            temp_data_info['cam_instances'] = cam_instances
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        if not empty_flag:
            converted_list.append(temp_data_info)
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')

    # dataset metainfo
    metainfo = {}
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    if ignore_class_name:
        for ignore_class in ignore_class_name:
            metainfo['categories'][ignore_class] = -1
    metainfo['dataset'] = 'kitti'
    metainfo['info_version'] = '1.1'
    converted_data_info = {'metainfo': metainfo, 'data_list': converted_list}

    mmengine.dump(converted_data_info, out_path, 'pkl')


def get_empty_instance():
    """Empty annotation for single instance."""
    instance = {
        # (list[float], required): list of 4 numbers representing
        # the bounding box of the instance, in (x1, y1, x2, y2) order.
        'bbox': None,
        # (int, required): an integer in the range
        # [0, num_categories-1] representing the category label.
        'bbox_label': None,
        #  (list[float], optional): list of  9 numbers representing
        #  the 3D bounding box of the instance,
        #  in [x, y, z, w, h, l, rx, ry,rz]
        'bbox_3d': None,
        # (bool, optional): Whether to use the
        # 3D bounding box during training.
        'bbox_3d_isvalid': None,
        # (int, optional): 3D category label
        # (typically the same as label).
        'bbox_label_3d': None,
        # (float, optional): Projected center depth of the
        # 3D bounding box compared to the image plane.
        'depth': None,
        #  (list[float], optional): Projected
        #  2D center of the 3D bounding box.
        'center_2d': None,
        # (int, optional): Attribute labels
        # (fine-grained labels such as stopping, moving, ignore, crowd).
        'attr_label': None,
        # (int, optional): The number of LiDAR
        # points in the 3D bounding box.
        'num_lidar_pts': None,
        # (int, optional): The number of Radar
        # points in the 3D bounding box.
        'num_radar_pts': None,
        # (int, optional): Difficulty level of
        # detecting the 3D bounding box.
        'difficulty': None,
        'unaligned_bbox_3d': None}
    return instance


def get_empty_multicamera_instances(camera_types):
    """ get_empty_multicamera_instances """
    cam_instance = {}
    for cam_type in camera_types:
        cam_instance[cam_type] = None
    return cam_instance


def get_empty_lidar_points():
    """ get_empty_lidar_points """
    lidar_points = {
        # (int, optional) : Number of features for each point.
        'num_pts_feats': None,
        # (str, optional): Path of LiDAR data file.
        'lidar_path': None,
        # (list[list[float]], optional): Transformation matrix
        # from lidar to ego-vehicle
        # with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        'lidar2ego': None,
    }
    return lidar_points


def get_empty_radar_points():
    """ get_empty_lidar_points """
    radar_points = {
        # (int, optional) : Number of features for each point.
        'num_pts_feats': None,
        # (str, optional): Path of RADAR data file.
        'radar_path': None,
        # Transformation matrix from lidar to
        # ego-vehicle with shape [4, 4].
        # (Referenced camera coordinate system is ego in KITTI.)
        'radar2ego': None,
    }
    return radar_points


def get_empty_img_info():
    """ get_empty_img_info """
    img_info = {
        # (str, required): the path to the image file.
        'img_path': None,
        # (int) The height of the image.
        'height': None,
        # (int) The width of the image.
        'width': None,
        # (str, optional): Path of the depth map file
        'depth_map': None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to image with
        # shape [3, 3], [3, 4] or [4, 4].
        'cam2img': None,
        # (list[list[float]]): Transformation matrix from lidar
        # or depth to image with shape [4, 4].
        'lidar2img': None,
        # (list[list[float]], optional) : Transformation
        # matrix from camera to ego-vehicle
        # with shape [4, 4].
        'cam2ego': None
    }
    return img_info


def get_single_image_sweep(camera_types):
    """ get_single_image_sweep """
    single_image_sweep = {
        # (float, optional) : Timestamp of the current frame.
        'timestamp': None,
        # (list[list[float]], optional) : Transformation matrix
        # from ego-vehicle to the global
        'ego2global': None
    }
    # (dict): Information of images captured by multiple cameras
    images = {}
    for cam_type in camera_types:
        images[cam_type] = get_empty_img_info()
    single_image_sweep['images'] = images
    return single_image_sweep


def get_empty_standard_data_info(camera_types=['CAM0']):
    """ get_empty_standard_data_info """
    data_info = {
        # (str): Sample id of the frame.
        'sample_idx': None,
        # (str, optional): '000010'
        'token': None,
        **get_single_image_sweep(camera_types),
        # (dict, optional): dict contains information
        # of LiDAR point cloud frame.
        'lidar_points': get_empty_lidar_points(),
        # (dict, optional) Each dict contains
        # information of Radar point cloud frame.
        'radar_points': get_empty_radar_points(),
        # (list[dict], optional): Image sweeps data.
        'image_sweeps': [],
        'lidar_sweeps': [],
        'instances': [],
        # (list[dict], optional): Required by object
        # detection, instance  to be ignored during training.
        'instances_ignore': [],
        # (str, optional): Path of semantic labels for each point.
        'pts_semantic_mask_path': None,
        # (str, optional): Path of instance labels for each point.
        'pts_instance_mask_path': None
    }
    return data_info


def clear_instance_unused_keys(instance):
    """ clear_instance_unused_keys """
    keys = list(instance.keys())
    for k in keys:
        if instance[k] is None:
            del instance[k]
    return instance


def clear_data_info_unused_keys(data_info):
    """ clear_data_info_unused_keys """
    keys = list(data_info.keys())
    empty_flag = True
    for key in keys:
        # we allow no annotations in datainfo
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag is False:
                empty_flag = False
            else:
                # sub field is empty
                del data_info[key]
        else:
            empty_flag = False

    return data_info, empty_flag


def update_pkl_infos(dataset, out_dir, pkl_path):
    """ update pkl info based on the dataset name """
    if dataset.lower() == 'tao3d':
        update_tao3d_infos(pkl_path=pkl_path, out_dir=out_dir)
    elif dataset.lower() == 'kitti':
        update_kitti_infos(pkl_path=pkl_path, out_dir=out_dir)
    else:
        raise NotImplementedError(f'Do not support convert {dataset}.')
