# updated train.py to work with checkpointing

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchinfo import summary
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import sys
import json

# from models import DiT_models
from diffusers.models import AutoencoderKL
from runners.diffusion import create_diffusion
from models.xswin_diffusion import XNetSwinTransformerDiffusion

# Profiling
from torch.profiler import profile, record_function, ProfilerActivity

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def load_checkpoint(checkpoint_dir, model, opt, device):
    checkpoint_files = glob(f"{checkpoint_dir}/*.pt")
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        last_epoch = checkpoint.get("epoch", -1)  # Default to -1 if not found
        train_steps = checkpoint.get("train_steps", 0)
    else:
        last_epoch, train_steps = -1, 0
    return last_epoch, train_steps

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Set experiment directory based on experiment-dir or results-dir
    if args.experiment_dir:
        assert os.path.exists(args.experiment_dir), "Experiment directory does not exist."
        experiment_dir = args.experiment_dir
        logger = create_logger(experiment_dir)
        logger.info(f"Resuming experiment from directory {experiment_dir}")
    else:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = 'xswin'
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8    
    
    
    # model = DiT_models[args.model](
        # input_size=latent_size,
        # num_classes=args.num_classes
    # )

    patch_size = [1, 1]
    embed_dim = 192
    depths = [3, 3]
    num_heads = [6, 12]
    window_size = [2, 2]
    num_classes = 101

    global_stages = 3
    input_size = [latent_size, latent_size]
    final_downsample = False
    residual_cross_attention = True
    input_channels = 4
    output_channels = 8
    class_dropout = 0.1
    smooth_conv = True

    model = XNetSwinTransformerDiffusion(patch_size, embed_dim, depths, 
                            num_heads, window_size, num_classes=num_classes,
                            global_stages=global_stages, input_size=input_size,
                            final_downsample=final_downsample, residual_cross_attention=residual_cross_attention,
                            input_channels=input_channels, output_channels=output_channels, 
                            class_dropout_prob=class_dropout, smooth_conv=smooth_conv,
                           )
    
    model_configs = {
        "patch_size" : patch_size,
        "embed_dim" : embed_dim,
        "depths" : depths,
        "num_heads" : num_heads,
        "window_size" : window_size,
        "num_class" : num_classes,
        "global_stages" : global_stages,
        "input_size" : input_size,
        "final_downsample" : final_downsample,
        "residual_cross_attention" : residual_cross_attention,
        "input_channels" : input_channels,
        "output_channels" : output_channels,
        "class_dropout_prob" : class_dropout,
        "smooth_conv" : smooth_conv,
    }

    # print(model_configs)
    model_summary = summary(model, input_size=[1, input_channels, *input_size], depth=4)
    # print(model)
    # print(args)
    if not args.experiment_dir:
        MODEL_SUMMARY_PATH = os.path.join(experiment_dir, "torchinfo.txt")
        MODEL_PRINT_PATH = os.path.join(experiment_dir, "modules.txt")
        MODEL_CONFIGS_PATH = os.path.join(experiment_dir, "config.json")
        SCRIPT_ARGS_PATH = os.path.join(experiment_dir, "args.txt")
        with open(MODEL_SUMMARY_PATH, "w") as f:
            f.write(model_summary.__str__())
        with open(MODEL_PRINT_PATH, "w") as f:
            f.write(model.__str__())
        with open(MODEL_CONFIGS_PATH, 'w') as f:
            json.dump(model_configs, f, indent=4)
        with open(SCRIPT_ARGS_PATH, 'w') as f:
            f.write(args.__str__())
    
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # Load checkpoint if experiment directory is provided
    last_epoch, train_steps = load_checkpoint(checkpoint_dir, model, opt, device)

    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(last_epoch + 1, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)

            # Profiling            
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            #     with record_function("module"):
            #         loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1

            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "epoch": epoch,
                    "train_steps": train_steps
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=20))

    model.eval()
    logger.info("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    # parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[32, 64, 128, 256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--experiment-dir", type=str, help="Path to the experiment directory to resume training")
    args = parser.parse_args()
    main(args)
