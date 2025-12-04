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

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "TangoFlux"))
sys.path.insert(0, os.path.join(_root, "SplatFormer"))

from tangoflux.model import TangoFlux
from models.feature_predictor import FeaturePredictor

from stage3.common import (
    load_ply,
    preprocess_gaussian,
    normalize_position,
    GaussianEncoder,
    PositionEncoder,
    FeatureFusion,
    retrieve_timesteps,
)


@torch.no_grad()
def inference_with_stage3(
    model,
    gaussian_encoder,
    position_encoder,
    feature_fusion,
    normalized_gs,
    position,
    device,
    duration=3.0,
    num_inference_steps=50,
    guidance_scale=-1,
    seed=0,
):
    """
    Generate audio from Gaussian (PLY) + Position using Stage 3 model.
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

    # Encode Gaussian
    gaussian_features = gaussian_encoder(normalized_gs, device)  # (1, seq_len, 1024)

    # Encode position
    if not isinstance(position, torch.Tensor):
        position = torch.tensor(position, dtype=torch.float32, device=device)
    position_features = position_encoder(position)  # (1024,)
    position_features = position_features.unsqueeze(0)  # (1, 1024)

    # Fuse features
    encoder_hidden_states = feature_fusion(gaussian_features, position_features)  # (1, seq_len*2, 1024)

    # Create attention mask
    boolean_encoder_mask = torch.ones(
        encoder_hidden_states.shape[0], encoder_hidden_states.shape[1],
        dtype=torch.bool, device=device
    )

    # Duration embedding
    duration_hidden_states = model.duration_emebdder(duration_tensor)

    if classifier_free_guidance:
        bsz = 2 * num_samples_per_prompt
        # CFG: [unconditional, conditional]
        uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
        encoder_hidden_states = torch.cat([uncond_encoder_states, encoder_hidden_states], dim=0)
        boolean_encoder_mask = torch.cat([boolean_encoder_mask, boolean_encoder_mask], dim=0)
        duration_hidden_states = duration_hidden_states.repeat(bsz, 1, 1)

    # Pooled projection
    mask_expanded = boolean_encoder_mask.unsqueeze(-1).expand_as(encoder_hidden_states)
    masked_data = torch.where(mask_expanded, encoder_hidden_states, torch.tensor(float("nan"), device=device))
    pooled = torch.nanmean(masked_data, dim=1)
    pooled_projection = model.fc(pooled)

    # Concatenate duration to encoder hidden states
    encoder_hidden_states = torch.cat([encoder_hidden_states, duration_hidden_states], dim=1)

    # Setup timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=sigmas)

    # Initialize latents
    audio_seq_len = model.audio_seq_len
    latents = torch.randn(num_samples_per_prompt, audio_seq_len, 64, device=device)

    # Position IDs
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


class Stage3Inference:
    """Inference class for Stage 3 (Gaussian + Position to Audio)"""

    def __init__(
        self,
        model_path,
        config_path="./configs/stage3.yaml",
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

        # Build Gaussian Encoder
        feature_predictor = FeaturePredictor()
        self.gaussian_encoder = GaussianEncoder(
            feature_predictor,
            output_dim=config['model']['text_embed_dim'],
            output_seq_len=config['model']['audio_seq_len'],
            mode='generative'
        )

        # Build Position Encoder
        pos_config = config['model']['position_encoder']
        self.position_encoder = PositionEncoder(
            num_frequencies=pos_config['num_frequencies'],
            input_dim=pos_config['input_dim'],
            hidden_dims=pos_config['hidden_dims'],
            output_dim=pos_config['output_dim'],
        )

        # Build Feature Fusion
        fusion_config = config['model']['feature_fusion']
        self.feature_fusion = FeatureFusion(
            embed_dim=config['model']['text_embed_dim'],
            num_heads=fusion_config['num_heads'],
            dropout=fusion_config['dropout'],
        )

        # Build TangoFlux model
        self.model = TangoFlux(config=config["model"])

        # Load VAE
        self.vae = AutoencoderOobleck.from_pretrained(
            "stabilityai/stable-audio-open-1.0", subfolder="vae"
        )

        # Load checkpoints
        self._load_checkpoint(model_path)

        # Move to device
        self.model.to(device)
        self.gaussian_encoder.to(device)
        self.position_encoder.to(device)
        self.feature_fusion.to(device)
        self.vae.to(device)

        # Set eval mode
        self.model.eval()
        self.gaussian_encoder.eval()
        self.position_encoder.eval()
        self.feature_fusion.eval()
        self.vae.eval()

    def _load_checkpoint(self, model_path):
        """Load model checkpoint from Accelerator save format"""
        if os.path.isdir(model_path):
            # Accelerator format:
            # model_0: vae (frozen)
            # model_1: TangoFlux
            # model_2: GaussianEncoder
            # model_3: PositionEncoder
            # model_4: FeatureFusion

            # Load TangoFlux (model_1)
            model_file = os.path.join(model_path, "model_1.safetensors")
            if os.path.exists(model_file):
                weights = load_file(model_file)
                result = self.model.load_state_dict(weights, strict=False)
                print(f"Loaded TangoFlux from {model_file}")
                print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            else:
                print(f"WARNING: TangoFlux checkpoint not found at {model_file}")

            # Load GaussianEncoder (model_2)
            ge_file = os.path.join(model_path, "model_2.safetensors")
            if os.path.exists(ge_file):
                ge_weights = load_file(ge_file)
                result = self.gaussian_encoder.load_state_dict(ge_weights, strict=False)
                print(f"Loaded GaussianEncoder from {ge_file}")
                print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            else:
                print(f"WARNING: GaussianEncoder checkpoint not found at {ge_file}")

            # Load PositionEncoder (model_3)
            pe_file = os.path.join(model_path, "model_3.safetensors")
            if os.path.exists(pe_file):
                pe_weights = load_file(pe_file)
                result = self.position_encoder.load_state_dict(pe_weights, strict=False)
                print(f"Loaded PositionEncoder from {pe_file}")
                print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            else:
                print(f"WARNING: PositionEncoder checkpoint not found at {pe_file}")

            # Load FeatureFusion (model_4)
            ff_file = os.path.join(model_path, "model_4.safetensors")
            if os.path.exists(ff_file):
                ff_weights = load_file(ff_file)
                result = self.feature_fusion.load_state_dict(ff_weights, strict=False)
                print(f"Loaded FeatureFusion from {ff_file}")
                print(f"  Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
            else:
                print(f"WARNING: FeatureFusion checkpoint not found at {ff_file}")
        else:
            if model_path.endswith('.safetensors'):
                weights = load_file(model_path)
            else:
                weights = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(weights, strict=False)
            print(f"Loaded model from {model_path}")

    def generate(self, ply_path, position, duration=3.0, steps=50, guidance_scale=-1, seed=0):
        """
        Generate audio from PLY file and impact position.
        """
        gs_params = load_ply(ply_path)
        normalized_gs, scaler = preprocess_gaussian(gs_params, self.device, return_scaler=True)

        normalized_position = normalize_position(position, scaler, self.device)

        # Generate latents
        with torch.no_grad():
            latents = inference_with_stage3(
                self.model,
                self.gaussian_encoder,
                self.position_encoder,
                self.feature_fusion,
                normalized_gs,
                normalized_position,
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
    parser = argparse.ArgumentParser(description="Stage 3 Inference: Generate audio from PLY + Position")
    parser.add_argument("--model_path", type=str, default="./ckpts/Stage3",
                        help="Path to trained model directory (contains model.safetensors)")
    parser.add_argument("--config", type=str, default="./configs/stage3.yaml",
                        help="Path to config file")
    parser.add_argument("--input", type=str, default="./datas/objectfolder_real_val.json",
                        help="Input JSON file with PLY paths and positions, or single PLY file")
    parser.add_argument("--output_dir", type=str, default="./generated/stage3",
                        help="Output directory for generated audio")
    parser.add_argument("--position", type=str, default=None,
                        help="Impact position as 'x,y,z' (for single PLY input)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=-1,
                        help="CFG scale (disabled if <= 1)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override audio duration")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

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
    inference = Stage3Inference(
        model_path=model_path,
        config_path=config_path,
    )

    if input_file.endswith('.ply'):
        if args.position is None:
            raise ValueError("Position must be provided for single PLY input (--position x,y,z)")
        
        position = [float(x) for x in args.position.split(',')]
        if len(position) != 3:
            raise ValueError("Position must have 3 values (x,y,z)")

        print(f"Generating audio from {input_file} at position {position}...")
        duration = args.duration if args.duration else 3.0
        audio = inference.generate(
            input_file,
            position=position,
            duration=duration,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        )
        output_path = os.path.join(output_dir, "output.wav")
        torchaudio.save(output_path, audio, 44100)
        print(f"Saved to {output_path}")
    else:
        # JSON file with multiple items
        print(f"Loading data from {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Found {len(data)} items to generate.")
        print(f"Output directory: {output_dir}")

        for i, item in tqdm(enumerate(data), total=len(data)):
            ply_path = item.get('ply_file', '')
            contact = item.get('contact', None)
            audio_location = item.get('location', '')

            if not ply_path or not os.path.exists(ply_path):
                print(f"Skipping {i}: PLY not found at {ply_path}")
                continue
            if contact is None or len(contact) != 3:
                print(f"Skipping {i}: Invalid contact position")
                continue

            duration = args.duration if args.duration else item.get('duration', 3.0)

            # Extract original filename from location path
            original_filename = os.path.basename(audio_location) if audio_location else f"{i}.wav"

            try:
                audio = inference.generate(
                    ply_path,
                    position=contact,
                    duration=duration,
                    steps=args.steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + i,
                )

                output_path = os.path.join(output_dir, original_filename)
                torchaudio.save(output_path, audio, 44100)

            except Exception as e:
                print(f"Error generating {original_filename} ({ply_path}): {e}")

    print("Done!")


if __name__ == "__main__":
    main()
