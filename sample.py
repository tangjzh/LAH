# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys

from diffusion import create_diffusion

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
from datasets.camera_utils import generate_poses, transform_pose
import imageio
from transformers import T5EncoderModel, T5Tokenizer
from torchvision import transforms
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from utils import visualize_frames

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    # Setup PyTorch:
    # torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    if args.ckpt is None:
        assert args.image_size in [256, 512]

    using_cfg = args.cfg_scale > 1.0

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps), learn_sigma=False)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(args.vae).to(device)
    # text_encoder = TextEmbedder().to(device)
    tokenizer = T5Tokenizer.from_pretrained(args.text_encoder)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder).to(device)

    vae.eval()
    text_encoder.eval()

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        text_encoder.to(dtype=torch.float16)

    # Labels to condition the model with (feel free to change):

    transform_mp3d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # Create sampling noise:
    z = torch.randn(1, args.num_frames, 4, latent_size, latent_size, device=device)
    camera_pose = generate_poses([-30, 0, 30], elevations=list(range(0, 360, 60)), anchor=0)
    
    if args.data:
        import glob

        img_path = os.path.join(args.data, 'images', '*.jpg')
        pose_path = os.path.join(args.data, 'poses', '*.txt')
        images_path, poses_path = glob.glob(img_path), glob.glob(pose_path)
        images, poses = [], []

        reference_pose = None
        for i, (image, pose) in enumerate(zip(images_path, poses_path)):
            image = transform_mp3d(Image.open(image)).to(device, dtype=vae.dtype) if i not in args.masked else None
            pose = np.loadtxt(pose).reshape(4, 4)
            if reference_pose is None:
                reference_pose = pose
                transformed_pose = np.hstack((np.eye(3), np.zeros((3, 1))))
                transformed_pose = np.vstack((transformed_pose, [0, 0, 0, 1]))
            else:
                transformed_pose = transform_pose(reference_pose, pose)

            if image is not None:
                image = vae.encode(image.unsqueeze(0)).latent_dist.sample().mul_(0.18215)
            images.append(image)

            pose = transformed_pose[:-1, :].flatten()
            poses.append(torch.tensor(pose, dtype=torch.float32))
        
        z = torch.stack(
            [img.squeeze() if img is not None
             else torch.randn(4, latent_size, latent_size, device=device) for img in images]
        ).unsqueeze(0).to(dtype=text_encoder.dtype, device=device)
        camera_pose = torch.stack(poses).unsqueeze(0).to(dtype=text_encoder.dtype, device=device)

    if args.use_fp16:
        z = z.to(dtype=torch.float16)
        camera_pose = camera_pose.to(dtype=torch.float16)

    max_length = 120
    text_inputs = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask.to(device)

    prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    sample_fn = model.forward
    model_kwargs = dict(camera_pose=camera_pose, encoder_hidden_states=prompt_embeds)

    # Sample images:
    if args.sample_method == 'ddim':
        samples = diffusion.ddim_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
    elif args.sample_method == 'ddpm':
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )

    print(samples.shape)
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)
    b, f, c, h, w = samples.shape
    samples = rearrange(samples, 'b f c h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
    # Save and display images:

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)


    video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    video_save_path = os.path.join(args.save_video_path, 'sample' + '.mp4')
    print(video_save_path)
    imageio.mimwrite(video_save_path, video_, fps=8, quality=9)
    print('save path {}'.format(args.save_video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--data", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.data = args.data
    omega_conf.save_video_path = args.save_video_path
    omega_conf.prompt = args.prompt
    main(omega_conf)
