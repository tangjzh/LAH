# TODO: Use diffusers to build training pipeline
from dataclasses import dataclass

import torch.utils
import torch.utils.data
from datasets import get_dataset
from models import get_models
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from diffusers.models import AutoencoderKL
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from omegaconf import OmegaConf
import torch
from PIL import Image
from diffusers import DDPMScheduler
from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os
from diffusers import DDPMPipeline
from diffusion.pipeline_lah import LattePipeline
from diffusers.utils import make_image_grid
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer
from glob import glob

from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, text_preprocessing)

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")    

def main(config):
    setup_distributed()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = config.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if rank == 0:
        os.makedirs(config.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{config.results_dir}/*"))
        model_string_name = config.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(config.num_frames) + 'S' + str(config.frame_interval)
        experiment_dir = f"{config.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{config.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, config)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(config, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    model = get_models(config)
    if config.pretrained:
        checkpoint = torch.load(config.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(config.pretrained))
    
    model = DDP(model.to(device), device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0)
    dataset = get_dataset(config)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=config.global_seed
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=int(config.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )

    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )

    vae = AutoencoderKL.from_pretrained(config.vae).to(device)
    tokenizer = T5Tokenizer.from_pretrained(config.text_encoder)
    text_encoder = T5EncoderModel.from_pretrained(config.text_encoder).to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    model.train()

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = LattePipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/mp3d.yaml")
    config = parser.parse_args()

    main(OmegaConf.load(config.config))