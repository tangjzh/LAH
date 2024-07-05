import os
import torch
import torch.utils.data as data
import zipfile
import numpy as np
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKL
from .camera_utils import transform_pose

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']

class MP3DDataset(data.Dataset):
    def __init__(self, 
                 configs,
                 transform=None):
        self.configs = configs
        self.data_root = configs.data_path
        self.transform = transform

        self.data_all = self.load_data(self.data_root)

    def __getitem__(self, index):
        item = self.data_all[index]
        frames = self.load_images(item['color_image_paths'])
        camera_pose = self.load_camera_pose(item['pose_paths'])
        prompt = self.load_text_descriptions(item['text_paths'])

        return {
            'frames': frames,
            'camera_pose': camera_pose,
            'prompt': prompt,
            'pano_name': item['uuid']
        }

    def __len__(self):
        return len(self.data_all)

    def load_data(self, root_dir):
        data = []
        for house_dir in os.listdir(root_dir):
            house_path = os.path.join(root_dir, house_dir)
            if not os.path.isdir(house_path):
                continue
            blip3_dir = os.path.join(house_path, 'blip3')
            pose_zip_path = os.path.join(house_path, 'matterport_camera_poses.zip')
            color_image_zip_path = os.path.join(house_path, 'matterport_color_images.zip')

            with zipfile.ZipFile(pose_zip_path, 'r') as pose_zip:
                uuids = set([os.path.basename(file).split('_')[0] for file in pose_zip.namelist() if 'pose' in file and file.endswith('.txt')])
                for uuid in uuids:
                    pose_path = [f"{pose_zip_path}//{house_dir}//matterport_camera_poses/{uuid}_pose_{camera_index}_{yaw_index}.txt" 
                                 for camera_index in range(3) for yaw_index in range(6)]
                    color_image_paths = [f"{color_image_zip_path}//{house_dir}//matterport_color_images/{uuid}_i{camera_index}_{yaw_index}.jpg" 
                                         for camera_index in range(3) for yaw_index in range(6)]
                    text_paths = [os.path.join(blip3_dir, f) for f in os.listdir(blip3_dir) if f.startswith(uuid)]

                    data.append({
                        'uuid': uuid,
                        'pose_paths': pose_path,
                        'color_image_paths': color_image_paths,
                        'text_paths': text_paths
                    })
        return data

    def load_images(self, image_paths):
        images = []
        for path in image_paths:
            zip_file_path, internal_path = path.split('//', 1)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                image = Image.open(zip_ref.open(internal_path))
                if self.transform:
                    image = self.transform(image)
                images.append(image)
        
        images = torch.stack(images)
        return images

    def load_camera_pose(self, pose_paths):
        poses = []
        reference_pose = None
        for pose_path in pose_paths:
            zip_file_path, internal_path = pose_path.split('//', 1)
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                pose_file = zip_ref.open(internal_path)
                pose = np.loadtxt(pose_file).reshape(4, 4)
                if reference_pose is None:
                    reference_pose = pose
                    transformed_pose = np.hstack((np.eye(3), np.zeros((3, 1))))
                    transformed_pose = np.vstack((transformed_pose, [0, 0, 0, 1]))
                else:
                    transformed_pose = transform_pose(reference_pose, pose)
                
                pose = transformed_pose[:-1, :].flatten()
                poses.append(torch.tensor(pose, dtype=torch.float32))
        
        poses = torch.stack(poses)
        return poses

    def load_text_descriptions(self, text_paths):
        descriptions = ""
        for path in text_paths:
            # degree = path.split('_')[-1].split('.')[0]
            with open(path, 'r') as f:
                descriptions += f.read() + ". "
        return descriptions.strip()