import time
import argparse
import json
import logging
import math
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

sys.path.append(os.path.join(_root, "SplatFormer"))
from models.feature_predictor import FeaturePredictor
from utils.transform_utils import MinMaxScaler

from stage3.common import (
    load_ply,
    preprocess_gaussian,
    normalize_position,
    GaussianEncoder,
    PositionEncoder,
    FeatureFusion,
    get_ply_path_from_item,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "TangoFlux"))

import yaml
from pathlib import Path
import diffusers
import datasets
import numpy as np
import pandas as pd
import wandb
import transformers
import torch
import torch.nn as nn
import gin
from plyfile import PlyData
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, Audio  # HuggingFace datasets
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler
from tangoflux.model import TangoFlux
from tangoflux.utils import read_wav_file, pad_wav

from diffusers import AutoencoderOobleck
import torchaudio

logger = get_logger(__name__)


def forward_with_stage3_encoder(model, latents, encoder_hidden_states, boolean_encoder_mask, duration):
    """Forward pass with Stage 3 fused features."""
    import random
    from diffusers.training_utils import compute_density_for_timestep_sampling
    
    device = latents.device
    audio_seq_length = model.audio_seq_len
    bsz = latents.shape[0]
    
    duration_hidden_states = model.duration_emebdder(duration)
    
    mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
    masked_data = torch.where(mask_expanded, encoder_hidden_states, torch.tensor(float("nan"), device=device))
    pooled = torch.nanmean(masked_data, dim=1)
    pooled_projection = model.fc(pooled)
    
    encoder_hidden_states = torch.cat([encoder_hidden_states, duration_hidden_states], dim=1)
    
    txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
    audio_ids = (
        torch.arange(audio_seq_length)
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(bsz, 1, 3)
        .to(device)
    )
    
    if model.uncondition:
        mask_indices = [k for k in range(bsz) if random.random() < 0.1]
        if len(mask_indices) > 0:
            encoder_hidden_states[mask_indices] = 0
    
    noise = torch.randn_like(latents)
    
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=bsz,
        logit_mean=0,
        logit_std=1,
        mode_scale=None,
    )
    
    indices = (u * model.noise_scheduler_copy.config.num_train_timesteps).long()
    timesteps = model.noise_scheduler_copy.timesteps[indices].to(device=latents.device)
    sigmas = model.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
    
    noisy_model_input = (1.0 - sigmas) * latents + sigmas * noise
    
    model_pred = model.transformer(
        hidden_states=noisy_model_input,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projection,
        img_ids=audio_ids,
        txt_ids=txt_ids,
        guidance=None,
        timestep=timesteps / 1000,
        return_dict=False,
    )[0]
    
    target = noise - latents
    loss = torch.mean(
        ((model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
        1,
    )
    loss = loss.mean()
    
    return loss


class Stage3Dataset(Dataset):
    """Dataset for Stage 3 training."""
    
    def __init__(self, raw_dataset, num_examples=-1):
        self.samples = []
        
        for item in raw_dataset:
            ply_path = item.get('ply_file', '')
            contact = item.get('contact', None)
            
            if not ply_path or not os.path.exists(ply_path):
                continue
            if contact is None or len(contact) != 3:
                continue
            
            self.samples.append({
                'audio_path': item['location'],
                'ply_path': ply_path,
                'contact': contact,
                'duration': item['duration'],
            })
        
        if num_examples > 0:
            self.samples = self.samples[:num_examples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_num_instances(self):
        return len(self.samples)
    
    def collate_fn(self, batch):
        audio_paths = [item['audio_path'] for item in batch]
        ply_paths = [item['ply_path'] for item in batch]
        contacts = [item['contact'] for item in batch]
        durations = [item['duration'] for item in batch]
        return audio_paths, ply_paths, contacts, durations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: Fine Tuning with Position Awareness"
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage3.yaml",
        help="Config file path.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-8,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=None,
        help="Override warmup steps from config.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for the Adam optimizer",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="best",
        help="Whether to save at 'epoch' or 'best' validation loss.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save model every N epochs when checkpointing_steps='best'.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint folder.",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="Load model weights from checkpoint.",
    )

    args = parser.parse_args()
    return args


def load_stage2_checkpoint(model, gaussian_encoder, checkpoint_path, accelerator):
    """Load Stage 2 checkpoint to initialize TangoFlux and Gaussian Encoder."""
    from safetensors.torch import load_file
    
    if not os.path.exists(checkpoint_path):
        accelerator.print(f"WARNING: Stage 2 checkpoint not found at {checkpoint_path}")
        return
    
    if os.path.isdir(checkpoint_path):
        model_file = os.path.join(checkpoint_path, "model_1.safetensors")
        if os.path.exists(model_file):
            weights = load_file(model_file)
            result = model.load_state_dict(weights, strict=False)
            accelerator.print(f"Loaded TangoFlux from {model_file}")
            accelerator.print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
        else:
            accelerator.print(f"WARNING: TangoFlux checkpoint not found at {model_file}")
        
        ge_file = os.path.join(checkpoint_path, "model_2.safetensors")
        if os.path.exists(ge_file):
            ge_weights = load_file(ge_file)
            result = gaussian_encoder.load_state_dict(ge_weights, strict=False)
            accelerator.print(f"Loaded GaussianEncoder from {ge_file}")
            accelerator.print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
        else:
            accelerator.print(f"WARNING: GaussianEncoder checkpoint not found at {ge_file}")
    else:
        if checkpoint_path.endswith('.safetensors'):
            weights = load_file(checkpoint_path)
        else:
            weights = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(weights, strict=False)
        accelerator.print(f"Loaded model from {checkpoint_path}")


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    config = load_config(args.config)

    learning_rate = args.learning_rate or float(config["training"]["learning_rate"])
    num_train_epochs = int(config["training"]["num_train_epochs"])
    num_warmup_steps = args.num_warmup_steps or int(config["training"]["num_warmup_steps"])
    per_device_batch_size = int(config["training"]["per_device_batch_size"])
    gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])
    freeze_gaussian_encoder = config["training"].get("freeze_gaussian_encoder", False)

    output_dir = config["paths"]["output_dir"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory and wandb
    if accelerator.is_main_process:
        if output_dir is None or output_dir == "":
            output_dir = "saved/" + str(int(time.time()))
            if not os.path.exists("saved"):
                os.makedirs("saved")
            os.makedirs(output_dir, exist_ok=True)
        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(
            project="SonicGauss Stage 3",
            settings=wandb.Settings(_disable_stats=True),
        )

    accelerator.wait_for_everyone()

    data_files = {}
    if config["paths"]["train_file"]:
        data_files["train"] = config["paths"]["train_file"]
    if config["paths"]["val_file"]:
        data_files["validation"] = config["paths"]["val_file"]
    if config["paths"].get("test_file"):
        data_files["test"] = config["paths"]["test_file"]
    else:
        data_files["test"] = config["paths"]["val_file"]

    raw_datasets = load_dataset("json", data_files=data_files)

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gin_config_path = os.path.join(root_dir, 'SplatFormer/configs/model/ptv3.gin')
    stride_binding = [f"PointTransformerV3Model.stride=({config['model']['ptv3_stride']})"]
    gin.parse_config_files_and_bindings([gin_config_path], stride_binding)

    feature_predictor = FeaturePredictor()
    gaussian_encoder = GaussianEncoder(
        feature_predictor,
        output_dim=config['model']['text_embed_dim'],
        output_seq_len=config['model']['audio_seq_len'],
        mode='generative'
    )

    pos_config = config['model']['position_encoder']
    position_encoder = PositionEncoder(
        num_frequencies=pos_config['num_frequencies'],
        input_dim=pos_config['input_dim'],
        hidden_dims=pos_config['hidden_dims'],
        output_dim=pos_config['output_dim'],
    )

    fusion_config = config['model']['feature_fusion']
    feature_fusion = FeatureFusion(
        embed_dim=config['model']['text_embed_dim'],
        num_heads=fusion_config['num_heads'],
        dropout=fusion_config['dropout'],
    )

    model = TangoFlux(config=config["model"])
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )

    for param in vae.parameters():
        vae.requires_grad = False
    vae.eval()

    for param in model.text_encoder.parameters():
        param.requires_grad = False
    model.text_encoder.eval()

    stage2_checkpoint = config["paths"].get("stage2_checkpoint", "")
    if stage2_checkpoint:
        load_stage2_checkpoint(model, gaussian_encoder, stage2_checkpoint, accelerator)

    if freeze_gaussian_encoder:
        for param in gaussian_encoder.parameters():
            param.requires_grad = False
        gaussian_encoder.eval()
        accelerator.print("Gaussian Encoder is frozen.")

    with accelerator.main_process_first():
        train_dataset = Stage3Dataset(raw_datasets["train"], args.num_examples)
        eval_dataset = Stage3Dataset(raw_datasets["validation"], args.num_examples)
        test_dataset = Stage3Dataset(raw_datasets["test"], args.num_examples)

        accelerator.print(
            f"Num instances in train: {train_dataset.get_num_instances()}, "
            f"validation: {eval_dataset.get_num_instances()}, "
            f"test: {test_dataset.get_num_instances()}"
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=per_device_batch_size,
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=per_device_batch_size,
        collate_fn=eval_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=per_device_batch_size,
        collate_fn=test_dataset.collate_fn,
    )

    trainable_params = (
        list(model.transformer.parameters()) +
        list(model.fc.parameters()) +
        list(position_encoder.parameters()) +
        list(feature_fusion.parameters())
    )
    
    if not freeze_gaussian_encoder:
        trainable_params += list(gaussian_encoder.parameters())

    num_trainable_parameters = sum(p.numel() for p in trainable_params if p.requires_grad)
    accelerator.print(f"Num trainable parameters: {num_trainable_parameters}")

    if args.load_from_checkpoint:
        from safetensors.torch import load_file
        w1 = load_file(args.load_from_checkpoint)
        model.load_state_dict(w1, strict=False)
        logger.info(f"Weights loaded from {args.load_from_checkpoint}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * gradient_accumulation_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * gradient_accumulation_steps,
    )

    (
        vae, model, gaussian_encoder, position_encoder, feature_fusion,
        optimizer, lr_scheduler
    ) = accelerator.prepare(
        vae, model, gaussian_encoder, position_encoder, feature_fusion,
        optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    total_batch_size = per_device_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running Stage 3 Training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0
    
    resume_from_checkpoint = config["paths"].get("resume_from_checkpoint", "")
    if resume_from_checkpoint:
        accelerator.load_state(resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {resume_from_checkpoint}")

    best_loss = np.inf
    length = config["training"]["max_audio_duration"]

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        if not freeze_gaussian_encoder:
            gaussian_encoder.train()
        position_encoder.train()
        feature_fusion.train()
        
        total_loss, total_val_loss = 0, 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                device = accelerator.device
                audios, ply_paths, contacts, duration = batch

                with torch.no_grad():
                    audio_list = []
                    for audio_path in audios:
                        wav = read_wav_file(audio_path, length)
                        if wav.shape[0] == 1:
                            wav = wav.repeat(2, 1)
                        audio_list.append(wav)

                    audio_input = torch.stack(audio_list, dim=0).to(device)
                    unwrapped_vae = accelerator.unwrap_model(vae)

                    duration_tensor = torch.tensor(duration, device=device)
                    duration_tensor = torch.clamp(duration_tensor, max=length)

                    audio_latent = unwrapped_vae.encode(audio_input).latent_dist.sample()
                    audio_latent = audio_latent.transpose(1, 2)

                encoder_hidden_states_list = []
                unwrapped_gaussian_encoder = accelerator.unwrap_model(gaussian_encoder)
                unwrapped_position_encoder = accelerator.unwrap_model(position_encoder)
                unwrapped_feature_fusion = accelerator.unwrap_model(feature_fusion)

                for i, (ply_path, contact) in enumerate(zip(ply_paths, contacts)):
                    try:
                        gs_params = load_ply(ply_path)
                        normalized_gs, scaler = preprocess_gaussian(gs_params, device, return_scaler=True)
                        gaussian_features = unwrapped_gaussian_encoder(normalized_gs, device)
                        normalized_contact = normalize_position(contact, scaler, device)
                        position_features = unwrapped_position_encoder(normalized_contact)
                        position_features = position_features.unsqueeze(0)
                        fused_features = unwrapped_feature_fusion(gaussian_features, position_features)
                        encoder_hidden_states_list.append(fused_features)

                    except Exception as e:
                        logger.warning(f"Error processing {ply_path}: {e}")
                        fused_seq_len = config['model']['audio_seq_len'] * 2
                        encoder_hidden_states_list.append(
                            torch.zeros(1, fused_seq_len, config['model']['text_embed_dim'], device=device)
                        )

                encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0)

                boolean_encoder_mask = torch.ones(
                    encoder_hidden_states.shape[0], encoder_hidden_states.shape[1],
                    dtype=torch.bool, device=device
                )

                loss = forward_with_stage3_encoder(
                    model, audio_latent, encoder_hidden_states, boolean_encoder_mask, duration_tensor
                )
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                optimizer.step()
                lr_scheduler.step()

            if accelerator.sync_gradients and completed_steps % 10 == 0 and accelerator.is_main_process:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                logger.info(f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}")

                lr = lr_scheduler.get_last_lr()[0]
                result = {
                    "train_loss": loss.item(),
                    "grad_norm": total_norm,
                    "learning_rate": lr,
                }
                wandb.log(result, step=completed_steps)

            if accelerator.sync_gradients and isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    ckpt_dir = f"step_{completed_steps}"
                    if output_dir is not None:
                        ckpt_dir = os.path.join(output_dir, ckpt_dir)
                    accelerator.save_state(ckpt_dir)

        if completed_steps >= args.max_train_steps:
            break

        model.eval()
        gaussian_encoder.eval()
        position_encoder.eval()
        feature_fusion.eval()
        
        eval_progress_bar = tqdm(
            range(len(eval_dataloader)), disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                device = accelerator.device
                audios, ply_paths, contacts, duration = batch

                audio_list = []
                for audio_path in audios:
                    wav = read_wav_file(audio_path, length)
                    if wav.shape[0] == 1:
                        wav = wav.repeat(2, 1)
                    audio_list.append(wav)

                audio_input = torch.stack(audio_list, dim=0).to(device)
                duration_tensor = torch.tensor(duration, device=device)
                unwrapped_vae = accelerator.unwrap_model(vae)
                audio_latent = unwrapped_vae.encode(audio_input).latent_dist.sample()
                audio_latent = audio_latent.transpose(1, 2)

                encoder_hidden_states_list = []
                unwrapped_gaussian_encoder = accelerator.unwrap_model(gaussian_encoder)
                unwrapped_position_encoder = accelerator.unwrap_model(position_encoder)
                unwrapped_feature_fusion = accelerator.unwrap_model(feature_fusion)

                for ply_path, contact in zip(ply_paths, contacts):
                    try:
                        gs_params = load_ply(ply_path)
                        normalized_gs, scaler = preprocess_gaussian(gs_params, device, return_scaler=True)
                        gaussian_features = unwrapped_gaussian_encoder(normalized_gs, device)
                        normalized_contact = normalize_position(contact, scaler, device)
                        position_features = unwrapped_position_encoder(normalized_contact).unsqueeze(0)

                        fused_features = unwrapped_feature_fusion(gaussian_features, position_features)
                        encoder_hidden_states_list.append(fused_features)
                    except Exception as e:
                        fused_seq_len = config['model']['audio_seq_len'] * 2
                        encoder_hidden_states_list.append(
                            torch.zeros(1, fused_seq_len, config['model']['text_embed_dim'], device=device)
                        )

                encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0)
                boolean_encoder_mask = torch.ones(
                    encoder_hidden_states.shape[0], encoder_hidden_states.shape[1],
                    dtype=torch.bool, device=device
                )

                val_loss = forward_with_stage3_encoder(
                    model, audio_latent, encoder_hidden_states, boolean_encoder_mask, duration_tensor
                )
                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        if accelerator.is_main_process:
            result = {}
            result["epoch"] = float(epoch + 1)
            result["epoch/train_loss"] = round(total_loss.item() / len(train_dataloader), 4)
            result["epoch/val_loss"] = round(total_val_loss.item() / len(eval_dataloader), 4)

            wandb.log(result, step=completed_steps)

            result_string = f"Epoch: {epoch}, Loss Train: {result['epoch/train_loss']}, Val: {result['epoch/val_loss']}\n"
            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["epoch/val_loss"] < best_loss:
                best_loss = result["epoch/val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        accelerator.wait_for_everyone()
        
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(output_dir, "best"))
            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state("{}/{}".format(output_dir, "epoch_" + str(epoch + 1)))

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            accelerator.save_state("{}/{}".format(output_dir, "epoch_" + str(epoch + 1)))


if __name__ == "__main__":
    main()
