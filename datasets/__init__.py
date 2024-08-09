import os
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset
from datasets import video_transforms
from omegaconf import OmegaConf
from typing import List
from .embodiedscan import EScanDataset
from .mp3d import MP3DDataset
from .vlnce import VLNCEDataset
from torchvision import transforms
from copy import deepcopy

def get_transform(dataset_type, args):
    if isinstance(args, dict):
        args = OmegaConf.create(args)
    if args.type == 'VLNCEDataset':
        return transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.type == 'MP3DDataset':
        return transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif args.type == 'EScanDataset':
        return transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def create_dataset_instance(dataset_type: str, config: dict) -> data.Dataset:
    try:
        dataset_class = globals()[dataset_type]
        dataset = dataset_class(config, 
                                transform=get_transform(dataset_type, config),
                                **dict(config.param))
        return dataset
    except KeyError:
        raise ValueError(f"Dataset type '{dataset_type}' is not defined.")
    except Exception as e:
        raise RuntimeError(f"Error initializing dataset '{dataset_type}': {e}")

def create_datasets(configs: List[dict]) -> List[data.Dataset]:
    datasets = []
    for config in configs:
        config_copy = deepcopy(config)
        dataset_type = config_copy.type
        dataset = create_dataset_instance(dataset_type, config_copy)
        datasets.append(dataset)
    return ConcatDataset(datasets)

def get_dataset(args):
    return create_datasets(args.datasets)