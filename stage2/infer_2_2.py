import json
import torchaudio
import sys
import os
import yaml
import torch
import numpy as np
import gin
from tqdm import tqdm
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck

# Setup paths
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)  # Project root for stage2.common
sys.path.insert(0, os.path.join(_root, "TangoFlux"))
sys.path.insert(0, os.path.join(_root, "SplatFormer"))  # This ends up at index 0

from tangoflux.model import TangoFlux
from models.feature_predictor import FeaturePredictor

# Stage2 shared utilities
from stage2.common import (
    load_ply, 
    preprocess_gaussian, 
    audio_path_to_ply_path, 
    GaussianEncoder,
    retrieve_timesteps
)


@torch.no_grad()
def inference_with_gaussian(
    model,
    gaussian_encoder,
    normalized_gs,
    device,
    duration=3.0,
    num_inference_steps=50,
    guidance_scale=4.5,
    seed=0,
):
    """
    Generate audio from Gaussian (PLY) using trained model.
    Replaces text encoder with Gaussian encoder.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    num_samples_per_prompt = 1
    bsz = num_samples_per_prompt
    scheduler = model.noise_scheduler

    # Duration tensor
    if not isinstance(duration, torch.Tensor):
        duration_tensor = torch.tensor([duration], device=device)
    else:
        duration_tensor = duration.to(device)
    
    classifier_free_guidance = guidance_scale > 1.0

    encoder_hidden_states = gaussian_encoder(normalized_gs, device)
    
    boolean_encoder_mask = torch.ones(
        encoder_hidden_states.shape[0], encoder_hidden_states.shape[1],
        dtype=torch.bool, device=device
    )

    # Duration embedding
    duration_hidden_states = model.duration_emebdder(duration_tensor)

    if classifier_free_guidance:
        bsz = 2 * num_samples_per_prompt
        # CFG order: [unconditional, conditional] - same as TangoFlux
        # Create unconditional (zeros) and conditional encoder states
        uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
        encoder_hidden_states = torch.cat([uncond_encoder_states, encoder_hidden_states], dim=0)
        boolean_encoder_mask = torch.cat([boolean_encoder_mask, boolean_encoder_mask], dim=0)
        duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)

    mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
    masked_data = torch.where(mask_expanded, encoder_hidden_states, torch.tensor(float("nan"), device=device))
    pooled = torch.nanmean(masked_data, dim=1)
    pooled_projection = model.fc(pooled)

    encoder_hidden_states = torch.cat([encoder_hidden_states, duration_hidden_states], dim=1)

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas)

    audio_seq_len = model.audio_seq_len
    latents = torch.randn(num_samples_per_prompt, audio_seq_len, 64, device=device)

    txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3).to(device)
    audio_ids = (
        torch.arange(audio_seq_len)
        .unsqueeze(0)
        .unsqueeze(-1)
        .repeat(bsz, 1, 3)
        .to(device)
    )

    # Sampling loop
    for i, t in enumerate(timesteps):
        latents_input = torch.cat([latents] * 2) if classifier_free_guidance else latents

        noise_pred = model.transformer(
            hidden_states=latents_input,
            timestep=torch.tensor([t / 1000], device=device),
            guidance=None,
            pooled_projections=pooled_projection,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=audio_ids,
            return_dict=False,
        )[0]

        if classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    return latents


class Stage22Inference:
    def __init__(
        self,
        model_path,
        config_path="./configs/stage2.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.config = config
        
        # Initialize PTv3 via gin
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gin_config_path = os.path.join(root_dir, 'SplatFormer/configs/model/ptv3.gin')
        stride_binding = [f"PointTransformerV3Model.stride=({config['model']['ptv3_stride']})"]
        gin.parse_config_files_and_bindings([gin_config_path], stride_binding)
        
        feature_predictor = FeaturePredictor()
        self.gaussian_encoder = GaussianEncoder(
            feature_predictor,
            output_dim=config['model']['text_embed_dim'],
            output_seq_len=config['model']['audio_seq_len'],
            mode='generative'
        )
        
        self.model = TangoFlux(config=config["model"])
        
        self.vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", subfolder="vae"
        )
        
        self._load_checkpoint(model_path)
        
        self.model.to(device)
        self.gaussian_encoder.to(device)
        self.vae.to(device)
        
        # Set eval mode
        self.model.eval()
        self.gaussian_encoder.eval()
        self.vae.eval()
    
    def _load_checkpoint(self, model_path):
        """Load model checkpoint from Accelerator save format"""
        if os.path.isdir(model_path):
            model_file = os.path.join(model_path, "model_1.safetensors")
            if os.path.exists(model_file):
                weights = load_file(model_file)
                result = self.model.load_state_dict(weights, strict=False)
                print(f"Loaded TangoFlux from {model_file}")
                print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            else:
                # Fallback to model.safetensors
                model_file = os.path.join(model_path, "model.safetensors")
                if os.path.exists(model_file):
                    weights = load_file(model_file)
                    self.model.load_state_dict(weights, strict=False)
                    print(f"Loaded TangoFlux from {model_file}")
            
            # Load GaussianEncoder (model_2)
            ge_file = os.path.join(model_path, "model_2.safetensors")
            if os.path.exists(ge_file):
                ge_weights = load_file(ge_file)
                result = self.gaussian_encoder.load_state_dict(ge_weights, strict=False)
                print(f"Loaded GaussianEncoder from {ge_file}")
                print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            else:
                print(f"WARNING: GaussianEncoder checkpoint not found at {ge_file}")
                print("  Using randomly initialized GaussianEncoder!")
        else:
            # Single file - assume it's combined weights
            if model_path.endswith('.safetensors'):
                weights = load_file(model_path)
            else:
                weights = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(weights, strict=False)
            print(f"Loaded model from {model_path}")
    
    def generate(self, ply_path, duration=3.0, steps=50, guidance_scale=4.5, seed=0):
        gs_params = load_ply(ply_path)
        normalized_gs = preprocess_gaussian(gs_params, self.device)
        
        # Generate latents
        with torch.no_grad():
            latents = inference_with_gaussian(
                self.model,
                self.gaussian_encoder,
                normalized_gs,
                self.device,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
            
            # Decode with VAE
            wave = self.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]
        
        # Trim to duration
        waveform_end = int(duration * self.vae.config.sampling_rate)
        wave = wave[:, :waveform_end]
        
        return wave


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 2.2 Inference: Generate audio from PLY files")
    parser.add_argument("--model_path", type=str, default="./ckpts/stage2/best",
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="./configs/stage2.yaml",
                        help="Path to config file")
    parser.add_argument("--input", type=str, default="./datas/objectfolder_2.0_val.json",
                        help="Input JSON file with PLY paths or single PLY file")
    parser.add_argument("--output_dir", type=str, default="./generated/stage2",
                        help="Output directory for generated audio")
    parser.add_argument("--ply_base", type=str, default=None,
                        help="Base path for PLY files (overrides config)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=-1,
                        help="Classifier-free guidance scale")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override audio duration (default: use from data)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()
    
    assert args.guidance_scale == -1, "Guidance scale must be disabled"
    
    config_path = args.config
    model_path = args.model_path
    input_file = args.input
    output_dir = args.output_dir
    
    if not os.path.exists(config_path) and os.path.exists(f"../{config_path}"):
        config_path = f"../{config_path}"
    if not os.path.exists(model_path) and os.path.exists(f"../{model_path}"):
        model_path = f"../{model_path}"
    if not os.path.exists(input_file) and os.path.exists(f"../{input_file}"):
        input_file = f"../{input_file}"
        output_dir = f"../{output_dir}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    inference = Stage22Inference(
        model_path=model_path,
        config_path=config_path,
    )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    ply_base = args.ply_base or config["paths"]["ply_base"]
    
    if input_file.endswith('.ply'):
        print(f"Generating audio from {input_file}...")
        duration = args.duration if args.duration else 3.0
        audio = inference.generate(
            input_file,
            duration=duration,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        output_path = os.path.join(output_dir, "output.wav")
        torchaudio.save(output_path, audio, 44100)
        print(f"Saved to {output_path}")
    else:
        print(f"Loading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Found {len(data)} items to generate.")
        print(f"Output directory: {output_dir}")
        
        for i, item in tqdm(enumerate(data), total=len(data)):
            audio_path = item.get('location', '')
            ply_path = audio_path_to_ply_path(audio_path, ply_base)
            
            if ply_path is None or not os.path.exists(ply_path):
                print(f"Skipping {i}: PLY not found for {audio_path}")
                continue
            
            duration = args.duration if args.duration else item.get('duration', 3.0)
            
            try:
                audio = inference.generate(
                    ply_path,
                    duration=duration,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + i,
                )
                
                output_path = os.path.join(output_dir, f"{i}.wav")
                torchaudio.save(output_path, audio, 44100)
                
            except Exception as e:
                print(f"Error generating {i} ({ply_path}): {e}")
    
    print("Done!")


if __name__ == "__main__":
    main()
