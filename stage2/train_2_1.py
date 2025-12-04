import time
import argparse
import json
import logging
import math
import os
import sys
import re

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SplatFormer"))

import yaml
from pathlib import Path
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from plyfile import PlyData
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5TokenizerFast, get_scheduler

# PTv3 imports
from models.feature_predictor import FeaturePredictor
from utils.transform_utils import MinMaxScaler

# Stage2 shared utilities
from stage2.common import load_ply, preprocess_gaussian, GaussianEncoder

logger = get_logger(__name__)


class GaussianTextDataset(Dataset):
    """Dataset for Gaussian-Text contrastive learning"""
    
    def __init__(self, json_path, audio_base, ply_base):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.audio_base = audio_base
        self.ply_base = ply_base
        
        self.samples = []
        for item in self.data:
            audio_path = item['location']
            caption = item['captions']
            # Extract object ID from audio path: .../audio_results/802/802.wav -> 802
            match = re.search(r'/(\d+)/\d+\.wav$', audio_path)
            if match:
                obj_id = match.group(1)
                ply_path = os.path.join(ply_base, obj_id, "model.ply")
                if os.path.exists(ply_path):
                    self.samples.append({
                        'caption': caption,
                        'ply_path': ply_path,
                        'obj_id': obj_id
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'caption': sample['caption'],
            'ply_path': sample['ply_path'],
            'obj_id': sample['obj_id']
        }


def collate_fn(batch):
    """Simple collate function that returns list of dicts"""
    return batch


# GaussianEncoder is now imported from stage2.utils


class TextEncoder(nn.Module):

    def __init__(self, model_name, projection_dim):
        super().__init__()
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        
        # Freeze T5
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
    
    def forward(self, texts, device):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # T5 encoder forward (no grad since frozen)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            hidden_states = outputs.last_hidden_state
        
        # Mean pooling over sequence
        attention_mask = inputs.attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * attention_mask
        pooled_feat = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)
        
        return pooled_feat


class InfoNCELoss(nn.Module):

    def __init__(self, temperature_init=0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))
    
    @property
    def temperature(self):
        return torch.exp(self.log_temperature)
    
    def forward(self, gaussian_features, text_features):
        gaussian_features = F.normalize(gaussian_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        logits = torch.matmul(gaussian_features, text_features.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_g2t = F.cross_entropy(logits, labels)
        loss_t2g = F.cross_entropy(logits.T, labels)
        
        loss = (loss_g2t + loss_t2g) / 2
        
        return loss

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2.1: Contrastive Semantic Matching")
    parser.add_argument("--config", type=str, default="configs/stage2.yaml", help="Path to config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if args.seed is not None:
        set_seed(args.seed)
    
    output_dir = config['paths']['output_dir']
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        
        wandb.init(
            project="SonicGauss-Stage2",
            config=config,
            settings=wandb.Settings(_disable_stats=True),
        )
    
    accelerator.wait_for_everyone()
    
    train_dataset = GaussianTextDataset(
        config['paths']['train_file'],
        config['paths']['audio_base'],
        config['paths']['ply_base']
    )
    val_dataset = GaussianTextDataset(
        config['paths']['val_file'],
        config['paths']['audio_base'],
        config['paths']['ply_base']
    )
    
    accelerator.print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config['training']['per_device_batch_size'],
        collate_fn=collate_fn,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config['training']['per_device_batch_size'],
        collate_fn=collate_fn,
        num_workers=4
    )
    
    # Initialize PTv3 via gin
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gin_config_path = os.path.join(root_dir, 'SplatFormer/configs/model/ptv3.gin')
    stride_binding = [f"PointTransformerV3Model.stride=({config['model']['ptv3_stride']})"]
    gin.parse_config_files_and_bindings([gin_config_path], stride_binding)
    
    feature_predictor = FeaturePredictor()
    gaussian_encoder = GaussianEncoder(
        feature_predictor,
        output_dim=config['model']['projection_dim'],
        mode='contrastive'
    )
    
    text_encoder = TextEncoder(
        config['model']['text_encoder_name'],
        projection_dim=config['model']['projection_dim']
    )
    
    # InfoNCE loss
    infonce_loss = InfoNCELoss(temperature_init=config['model']['temperature_init'])
    
    trainable_params = sum(p.numel() for p in gaussian_encoder.parameters() if p.requires_grad)
    # trainable_params += sum(p.numel() for p in text_encoder.projection.parameters() if p.requires_grad)
    trainable_params += sum(p.numel() for p in infonce_loss.parameters() if p.requires_grad)
    accelerator.print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer_params = list(gaussian_encoder.parameters()) + \
                       list(infonce_loss.parameters())
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=float(config['training']['learning_rate']),
        weight_decay=1e-4
    )
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config['training']['gradient_accumulation_steps']
    )
    max_train_steps = config['training']['num_train_epochs'] * num_update_steps_per_epoch
    
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=config['training']['num_warmup_steps'],
        num_training_steps=max_train_steps
    )
    
    gaussian_encoder, text_encoder, infonce_loss, optimizer, lr_scheduler = accelerator.prepare(
        gaussian_encoder, text_encoder, infonce_loss, optimizer, lr_scheduler
    )
    train_dataloader, val_dataloader = accelerator.prepare(train_dataloader, val_dataloader)
    
    if config['paths']['resume_from_checkpoint'] and config['paths']['resume_from_checkpoint'] != "":
        accelerator.load_state(config['paths']['resume_from_checkpoint'])
        accelerator.print(f"Resumed from: {config['paths']['resume_from_checkpoint']}")
    
    # Training
    contrastive_batch_size = config['training']['contrastive_batch_size']
    best_val_loss = float('inf')
    completed_steps = 0
    
    logger.info("***** Running training *****")
    logger.info(f"  Num epochs = {config['training']['num_train_epochs']}")
    logger.info(f"  Contrastive batch size = {contrastive_batch_size}")
    
    for epoch in range(config['training']['num_train_epochs']):
        gaussian_encoder.train()
        total_train_loss = 0
        
        progress_bar = tqdm(
            range(len(train_dataloader)),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch+1}"
        )
        
        gaussian_features_buffer = []
        text_features_buffer = []
        
        for step, batch in enumerate(train_dataloader):
            device = accelerator.device
            
            for sample in batch:
                caption = sample['caption']
                ply_path = sample['ply_path']
                
                try:
                    gs_params = load_ply(ply_path)
                    normalized_gs = preprocess_gaussian(gs_params, device)
                    with torch.set_grad_enabled(True):
                        g_feat = gaussian_encoder(normalized_gs, device)
                    gaussian_features_buffer.append(g_feat)
                    
                    t_feat = text_encoder([caption], device)
                    text_features_buffer.append(t_feat)
                    
                except Exception as e:
                    logger.warning(f"Error processing {ply_path}: {e}")
                    continue
            
            if len(gaussian_features_buffer) >= contrastive_batch_size:
                gaussian_features = torch.cat(gaussian_features_buffer[:contrastive_batch_size], dim=0)
                text_features = torch.cat(text_features_buffer[:contrastive_batch_size], dim=0)
                
                with accelerator.accumulate(gaussian_encoder):
                    loss = infonce_loss(gaussian_features, text_features)
                    total_train_loss += loss.detach().float()
                    
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    completed_steps += 1
                
                # Clear buffers
                gaussian_features_buffer = gaussian_features_buffer[contrastive_batch_size:]
                text_features_buffer = text_features_buffer[contrastive_batch_size:]
                
                # Log
                if completed_steps % config['training']['log_every'] == 0 and accelerator.is_main_process:
                    temp = infonce_loss.module.temperature.item() if hasattr(infonce_loss, 'module') else infonce_loss.temperature.item()
                    lr = lr_scheduler.get_last_lr()[0]
                    logger.info(f"Step {completed_steps}, Loss: {loss.item():.4f}, Temp: {temp:.4f}, LR: {lr:.6f}")
                    wandb.log({
                        "train_loss": loss.item(),
                        "temperature": temp,
                        "learning_rate": lr
                    }, step=completed_steps)
            
            progress_bar.update(1)
        
        # Validation
        if (epoch + 1) % config['training']['eval_every'] == 0:
            gaussian_encoder.eval()
            total_val_loss = 0
            val_steps = 0
            
            gaussian_features_buffer = []
            text_features_buffer = []
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validation", disable=not accelerator.is_local_main_process):
                    device = accelerator.device
                    
                    for sample in batch:
                        caption = sample['caption']
                        ply_path = sample['ply_path']
                        
                        try:
                            gs_params = load_ply(ply_path)
                            normalized_gs = preprocess_gaussian(gs_params, device)
                            
                            g_feat = gaussian_encoder(normalized_gs, device)
                            gaussian_features_buffer.append(g_feat)
                            
                            t_feat = text_encoder([caption], device)
                            text_features_buffer.append(t_feat)
                            
                        except Exception as e:
                            continue
                    
                    if len(gaussian_features_buffer) >= contrastive_batch_size:
                        gaussian_features = torch.cat(gaussian_features_buffer[:contrastive_batch_size], dim=0)
                        text_features = torch.cat(text_features_buffer[:contrastive_batch_size], dim=0)
                        
                        loss = infonce_loss(gaussian_features, text_features)
                        total_val_loss += loss.item()
                        val_steps += 1
                        
                        gaussian_features_buffer = gaussian_features_buffer[contrastive_batch_size:]
                        text_features_buffer = text_features_buffer[contrastive_batch_size:]
            
            avg_val_loss = total_val_loss / max(val_steps, 1)
            
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}")
                wandb.log({"val_loss": avg_val_loss, "epoch": epoch + 1}, step=completed_steps)
                
                # Save best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    accelerator.save_state(os.path.join(output_dir, "checkpoints", "best"))
                    logger.info(f"Saved best model at epoch {epoch+1}")
        
        # Save periodic checkpoint
        if accelerator.is_main_process and (epoch + 1) % config['training']['save_every'] == 0:
            accelerator.save_state(os.path.join(output_dir, "checkpoints", f"epoch_{epoch+1}"))
    
    accelerator.print("Training complete!")


if __name__ == "__main__":
    main()

