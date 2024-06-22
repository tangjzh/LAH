from torchvision import transforms
from datasets import video_transforms

from .mp3d import MP3DDataset

def get_dataset(args):
    temporal_sample = video_transforms.TemporalRandomCrop(args.num_frames * args.frame_interval) # 16 1

    if args.dataset == 'mp3d':
        transform_mp3d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        return MP3DDataset(args, transform=transform_mp3d)