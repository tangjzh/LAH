import unittest
import torch
from torchvision import transforms
import os
import json
import numpy as np
from datasets.vlnce import VLNCEDataset
from datasets.embodiedscan import EScanDataset
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from datasets import create_datasets

def save_frames_as_video(dataset, output_path, fps=1):
    """
    Save frames from the dataset as a video file.
    
    Args:
    - dataset (EScanDataset): The dataset object containing frames.
    - output_path (str): The path to save the output video file.
    - fps (int): Frames per second for the output video.
    """
    # Ensure the dataset has at least one item
    if len(dataset) == 0:
        raise ValueError("The dataset is empty.")
    
    # Get the first item to determine the frame size
    first_item = dataset[0]
    first_frame = first_item['frames'][0]
    
    # Convert the first frame to a format suitable for cv2
    if isinstance(first_frame, torch.Tensor):
        frame_size = (first_frame.shape[2], first_frame.shape[1])  # (width, height)
    elif isinstance(first_frame, Image.Image):
        frame_size = first_frame.size  # (width, height)
    elif isinstance(first_frame, np.ndarray):
        frame_size = (first_frame.shape[1], first_frame.shape[0])  # (width, height)
    else:
        raise TypeError(f"Unsupported frame type: {type(first_frame)}")

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec if needed
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print("Length", len(dataset))
    item = dataset[2078]

    frames = item['frames']
    for frame in frames:
        # Convert the frame to a format suitable for cv2
        if isinstance(frame, torch.Tensor):
            frame = frame.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            frame = (frame * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8
        elif isinstance(frame, Image.Image):
            frame = np.array(frame)
        elif isinstance(frame, np.ndarray):
            frame = (frame * 255).astype(np.uint8)  # Ensure the frame is in the right format
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        
        # Write the frame to the video
        video_writer.write(frame)
    
    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_path}")

configs = OmegaConf.load('configs/mp3d.yaml')
dataset = create_datasets(configs.datasets)

loader = DataLoader(dataset, batch_size=2, shuffle=True)
for data in loader:
    print(data)
# save_frames_as_video(dataset, 'output_video.mp4')
