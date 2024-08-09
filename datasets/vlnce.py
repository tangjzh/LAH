import os
import json
import random
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from .camera_utils import transform_pose, generate_rays_with_extrinsics, visualize_extrinsics
from glob import glob

class VLNCEDataset(data.Dataset):
    def __init__(self, configs, transform=None, return_pt=True, **kwargs):
        self.configs = configs
        self.data_root = configs.data_path
        self.video_length = configs.num_frames
        self.img_size = configs.image_size
        self.mask_prob = configs.mask_prob
        self.transform = transform
        self.return_pt = return_pt

        self.data_all = self.load_data(self.data_root)

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        item = self.data_all[index]

        frames = self.load_images(item['image_paths'])
        camera_pose, rays = self.load_poses_and_rays(item['pose_paths'])
        prompt = item['prompt']
        mask = torch.tensor(np.random.rand(self.video_length) < self.mask_prob, dtype=torch.bool)
        if torch.all(~mask):
            unmask_index = random.randint(0, self.video_length - 1)
            mask[unmask_index] = True

        return {
            'frames': frames,
            'camera_pose': camera_pose,
            'ray': rays,
            'mask': mask,
            'prompt': prompt,
            'enable_time': True,
            'enable_camera': True,
        }

    def load_data(self, root_dir):
        data = []
        for id_folder in os.listdir(root_dir):
            id_path = os.path.join(root_dir, id_folder)
            if not os.path.isdir(id_path):
                continue
            
            result_json_path = os.path.join(id_path, 'result.json')
            sequence_infos_path = os.path.join(id_path, 'sequence_infos.json')

            if not os.path.exists(result_json_path) or not os.path.exists(sequence_infos_path):
                continue

            with open(result_json_path, 'r') as result_file, open(sequence_infos_path, 'r') as seq_info_file:
                results = json.load(result_file)
                seq_info = json.load(seq_info_file)

                for seq, desc in results.items():
                    image_paths = []
                    pose_paths = []
                    valid_segments = []

                    try:
                        start, end = map(int, seq.split('-'))
                        
                        for i in range(start, end + 1):
                            waypoint_path = os.path.join(id_path, f'waypoint_{i}')
                            if not os.path.exists(waypoint_path):
                                continue
                            img_path = sorted(glob(os.path.join(waypoint_path, "*.jpg")), 
                                              key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
                            pose_path = sorted(glob(os.path.join(waypoint_path, "*.txt")), 
                                               key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
                            image_paths.extend(img_path)
                            pose_paths.extend(pose_path)
                    except Exception as e:
                        print(f'Warning: parse error: {result_json_path}')

                    if len(image_paths) >= self.video_length:
                        for j in range(0, len(image_paths) - self.video_length + 1, self.video_length):
                            segment_image_paths = image_paths[j:j + self.video_length]
                            segment_pose_paths = pose_paths[j:j + self.video_length]
                            if len(segment_image_paths) == self.video_length:
                                valid_segments.append({
                                    'image_paths': segment_image_paths,
                                    'pose_paths': segment_pose_paths,
                                    'prompt': desc,
                                    'episode_id': seq_info['episode_id']
                                })
                    data.extend(valid_segments)
        return data

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            with Image.open(path) as img:
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        if self.return_pt:
            return torch.stack(images)
        else:
            return images

    def load_poses_and_rays(self, pose_paths):
        poses, rays = [], []
        # temp = []
        align_matrix = np.array([
                [0, 1, 0, 0],  # y becomes x
                [0, 0, 1, 0],  # z becomes y
                [1, 0, 0, 0],  # x becomes z
                [0, 0, 0, 1]   # homogeneous coordinates remain the same
            ]) + 1e-17
        # align_matrix = np.eye(4)
        reference_pose = None
        for pose_path in pose_paths:
            pose = np.loadtxt(pose_path).reshape(4, 4)
            pose = align_matrix @ pose
            if reference_pose is None:
                reference_pose = pose
            transformed_pose = transform_pose(reference_pose, pose)
            transformed_pose[:-1, 3] *= -1
            # transformed_pose[1, 3], transformed_pose[2, 3] \
            #     = transformed_pose[2, 3], transformed_pose[1, 3]
            pose = transformed_pose[:-1, 3].flatten()
            ray = generate_rays_with_extrinsics(transformed_pose, width=self.img_size, height=self.img_size)
            # temp.append(transformed_pose)
            poses.append(torch.tensor(pose, dtype=torch.float32))
            rays.append(torch.tensor(ray, dtype=torch.float32))
        # visualize_extrinsics(temp, './vis.jpg')
        if self.return_pt:
            return torch.stack(poses), torch.stack(rays)
        else:
            return poses, rays