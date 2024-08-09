import os
import torch
import torch.utils.data as data
from PIL import Image
import zipfile
from .camera_utils import transform_pose, generate_rays_with_extrinsics
import random
import numpy as np
import json
from typing import List
import warnings

class EScanDataset(data.Dataset):
    def __init__(self, 
                 configs,
                 transform=None,
                 random=True,
                 return_pt=True):
        self.configs = configs
        self.data_root = configs.data_path
        self.video_length = configs.num_frames
        self.sample_interval = configs.sample_interval
        self.img_size = configs.image_size
        self.mask_prefix = configs.mask_prefix
        self.ann_file = configs.escan_ann_file
        self.vg_file = configs.escan_vg_file
        self.transform = transform
        self.random = random
        self.return_pt = return_pt

        self.data_all = self.load_data()

    def __getitem__(self, index):
        item = self.data_all[index]
        
        frames = self.load_images(item['img_path'])
        camera_pose, rays = self.load_camera_pose(item['extrinsic'])
        prompt = item['text']
        mask = np.zeros(self.video_length, dtype=int)
        if self.random:
            mask[:np.random.randint(1, self.mask_prefix)] = 1
        else:
            mask[:self.mask_prefix] = 1
        mask = ~torch.tensor(mask, dtype=torch.bool)

        return {
            'frames': frames,
            'camera_pose': camera_pose,
            'ray': rays,
            'mask': mask,
            'prompt': prompt,
            'enable_time': True,
            'enable_camera': False,
        }

    def __len__(self):
        return len(self.data_all)

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            image = Image.open(path)
            if self.transform:
                image = self.transform(image)
            images.append(image)
        if self.return_pt:
            images = torch.stack(images)
        return images
    
    def load_camera_pose(self, extrinsics):
        poses, rays = [], []
        reference_pose = None

        for extrinsic in extrinsics:
            if reference_pose is None:
                reference_pose = extrinsic
            transformed_pose = transform_pose(reference_pose, extrinsic)
            
            pose = transformed_pose[:-1, 3].flatten()
            ray = generate_rays_with_extrinsics(transformed_pose, width=self.img_size, height=self.img_size)
            poses.append(torch.tensor(pose, dtype=torch.float32))
            rays.append(torch.tensor(ray, dtype=torch.float32))
        
        if self.return_pt:
            poses = torch.stack(poses)
            rays = torch.stack(rays)
        return poses, rays


    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): Info of a single sample data.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `axis_align_matrix'.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        # Because multi-view settings are different from original designs
        # we temporarily follow the ori design in ImVoxelNet
        info['img_path'] = []
        info['depth_img_path'] = []
        info['scan_id'] = info['sample_idx']
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'matterport3d':
            info['depth_shift'] = 4000.0
        else:
            info['depth_shift'] = 1000.0

        if 'cam2img' in info:
            cam2img = info['cam2img'].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in range(len(info['images'])):
            img_path = os.path.join(self.data_root,
                                    info['images'][i]['img_path'])
            depth_img_path = os.path.join(self.data_root,
                                          info['images'][i]['depth_path'])

            info['img_path'].append(img_path)
            info['depth_img_path'].append(depth_img_path)
            align_global2cam = np.linalg.inv(
                info['axis_align_matrix'] @ info['images'][i]['cam2global'])
            extrinsics.append(align_global2cam.astype(np.float32))
            if 'cam2img' not in info:
                cam2img.append(info['images'][i]['cam2img'].astype(np.float32))

        info['depth2img'] = dict(extrinsic=extrinsics,
                                 intrinsic=cam2img,
                                 origin=np.array([.0, .0,
                                                  .5]).astype(np.float32))

        if 'depth_cam2img' not in info:
            info['depth_cam2img'] = cam2img

        return info
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        If the annotation file does not follow `OpenMMLab 2.0 format dataset
        <https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html>`_ .
        The subclass must override this method for load annotations. The meta
        information of annotation file will be overwritten :attr:`METAINFO`
        and ``metainfo`` argument of constructor.

        Returns:
            list[dict]: A list of annotation.
        """  # noqa: E501
        # `self.ann_file` denotes the absolute annotation file path if
        # `self.root=None` or relative path if `self.root=/path/to/data/`.
        annotations = np.load(self.ann_file, allow_pickle=True)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'data_list' not in annotations or 'metainfo' not in annotations:
            raise ValueError('Annotation must have data_list and metainfo '
                             'keys')
        raw_data_list = annotations['data_list']

        # load and parse data_infos.
        data_list = []
        for raw_data_info in raw_data_list:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')

        return data_list

    def process_text(self, text: str) -> str:
        keywords = [
            "find", "select", "choose", "locate", "identify", "spot", "pick", 
            "detect", "recognize", "discover", "uncover", "determine", "observe", 
            "track", "search", "inspect", "examine", "investigate", "survey", 
            "scan", "explore", "probe", "study", "reveal", "notice", "perceive", 
            "discern", "ascertain", "pinpoint", "scrutinize", "analyze"
        ]

        detail_phrases = [
            "carefully", "quickly", "precisely", "with caution", "thoroughly", 
            "efficiently", "meticulously", "diligently", "promptly", "accurately",
            "methodically", "systematically", "attentively", "vigilantly",
            "consistently", "deliberately", "effectively", "rigorously",
            "strategically", "skillfully", "competently", "exhaustively",
            "intelligently", "resourcefully", "keenly", "conscientiously",
            "decisively", "prudently", "patiently"
        ]

        words = text.split()
        contains_keyword = any(keyword in words for keyword in keywords)

        if contains_keyword:
            for keyword in keywords:
                if keyword in words:
                    keyword_index = words.index(keyword)
                    new_keyword = random.choice([k for k in keywords if k != keyword])
                    detail_phrase = random.choice(detail_phrases)
                    words[keyword_index] = new_keyword
                    words.insert(keyword_index + 1, detail_phrase)
                    break
            processed_text = " ".join(words)
        else:
            keyword = random.choice(keywords)
            processed_text = f"{keyword} {text}"

        return processed_text

    def load_data(self):
        with open(self.vg_file, 'r') as f:
            language_annotations = json.load(f)
        language_annotations = language_annotations[::self.sample_interval]
        scans = dict()
        for data in self.load_data_list():
            scan_id = data['scan_id']
            scans[scan_id] = data

        language_infos = []
        for anno in language_annotations:
            if 'matterport' in anno['scan_id']:
                continue
            data = scans[anno['scan_id']]
            img_paths = data['img_path']

            sampled_img_paths = img_paths[::3]

            # Split img_paths into segments of length self.video_length
            segments = [(i, i + self.video_length) for i in range(0, len(sampled_img_paths), self.video_length)]

            for segment in segments:
                l, r = segment
                if len(img_paths[l:r]) != self.video_length:
                    continue
                    
                language_info = dict()
                language_info.update({
                    'scan_id': anno['scan_id'],
                    'text': self.process_text(anno['text']),
                    'axis_align_matrix': data['axis_align_matrix'],
                    'img_path': img_paths[l:r],
                    'depth_img_path': data['depth_img_path'][l:r],
                    'extrinsic': data['depth2img']['extrinsic'][l:r],
                    'depth_shift': data['depth_shift'],
                    'depth_cam2img': data['depth_cam2img']
                })

                if 'cam2img' in data:
                    language_info['cam2img'] = data['cam2img'][l:r]
                
                language_infos.append(language_info)

        return language_infos