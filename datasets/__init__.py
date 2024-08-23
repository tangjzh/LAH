import os
import torch
import torch.utils.data as data
from torch.utils.data import ConcatDataset, DistributedSampler, IterableDataset
from datasets import video_transforms
from omegaconf import OmegaConf, ListConfig
from typing import List
from .embodiedscan import EScanDataset
from .mp3d import MP3DDataset
from .vlnce import VLNCEDataset
from torchvision import transforms
from copy import deepcopy

class IterableDatasetWrapper(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.__len__()
    
    def __iter__(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]

class RepeatDatasetWrapper(data.Dataset):
    def __init__(self, dataset, repeat_times):
        self.dataset = dataset
        self.repeat_times = repeat_times

    def __len__(self):
        return len(self.dataset) * self.repeat_times

    def __getitem__(self, index):
        actual_index = index % len(self.dataset)
        return self.dataset[actual_index]

class DistributedConcatSampler(DistributedSampler):
    def __init__(self, dataset, samplers, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(
                dataset,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=seed
            )
        self.samplers = samplers

    def set_epoch(self, epoch):
        self.epoch = epoch
        for sampler in self.samplers:
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

    def __iter__(self):
        for sampler in self.samplers:
            yield from iter(sampler)

def get_sampler(datasets, shuffle_datasets, shuffle=False, seed=None, num_replicas=None, rank=None):
    if shuffle:
        return DistributedSampler(
                datasets,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=True,
                seed=seed
            )
    else:
        samplers = []
        for i, dataset in enumerate(datasets.datasets):
            if i in shuffle_datasets:
                samplers.append(DistributedSampler(
                    dataset, num_replicas=num_replicas, rank=rank, shuffle=True, seed=seed))
            else:
                samplers.append(DistributedSampler(
                    dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=seed))
        return DistributedConcatSampler(
                dataset, samplers,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=False,
                seed=seed
            )

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

def create_dataset_instance(dataset_type: str, config: dict, repeat: int=1) -> data.Dataset:
    try:
        dataset_class = globals()[dataset_type]
        dataset = dataset_class(config, 
                                transform=get_transform(dataset_type, config),
                                **dict(config.param))
        dataset = RepeatDatasetWrapper(dataset, repeat)
        return dataset
    except KeyError:
        raise ValueError(f"Dataset type '{dataset_type}' is not defined.")
    except Exception as e:
        raise RuntimeError(f"Error initializing dataset '{dataset_type}': {e}")

def create_datasets(configs: List[dict]) -> List[data.Dataset]:
    datasets = []
    for config in configs:
        dataset_type = config.type
        if dataset_type == 'MixedDataset':
            dataset = create_datasets(config.datasets)
        else:
            dataset = create_dataset_instance(dataset_type, config, config.repeat)
        datasets.append(dataset)
    return ConcatDataset(datasets)

def get_dataset(args):
    shuffle_dataset = []
    for i, config in enumerate(args.datasets):
        if config.type == 'MixedDataset':
            shuffle_dataset.append(i)
    return create_datasets(args.datasets), shuffle_dataset