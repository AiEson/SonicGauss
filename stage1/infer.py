import json
import torchaudio
import sys, os
import yaml
import torch
from tqdm import tqdm
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck

# Add TangoFlux to path
sys.path.append('./TangoFlux')
try:
    from tangoflux import TangoFluxInference
    from tangoflux.model import TangoFlux
except ImportError:
    # Fallback if running from stage1 directory
    sys.path.append('../TangoFlux')
    from tangoflux import TangoFluxInference
    from tangoflux.model import TangoFlux


def custom_init(
    self,
    name="declare-lab/TangoFlux",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    # Try to find config file
    config_path = './configs/stage1.yaml'
    if not os.path.exists(config_path):
        config_path = '../configs/stage1.yaml'
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config file at {config_path}")

    config = load_config(config_path)

    self.vae = AutoencoderOobleck.from_pretrained(
        "/hdd2/wcs/.cache/huggingface/hub/models--stabilityai--stable-audio-open-1.0/snapshots/67351841772475731424cf358c9fb785716ac78f", subfolder="vae"
    )
    
    # Handle loading weights from split files
    all_weights = load_file(f"{name}/model_1.safetensors")
    all_weights.update(load_file(f"{name}/model.safetensors"))
    
    self.model = TangoFlux(config=config["model"])
    print(self.model.load_state_dict(all_weights, strict=False))
    self.vae.to(device)
    self.model.to(device)

# Patch the init method
TangoFluxInference.__init__ = custom_init

def main():
    # Setup paths
    output_dir = './generated/stage1'
    model_path = './ckpts/stage1/best'
    input_file = 'datas/objectfolder_2.0_val.json'
    
    # Adjust paths if running from stage1 directory
    if not os.path.exists(input_file) and os.path.exists(f"../{input_file}"):
        input_file = f"../{input_file}"
        output_dir = '../generated/stage1'
        model_path = '../ckpts/stage1/best'
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    model = TangoFluxInference(name=model_path)
    
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Found {len(data)} items to generate.")
    print(f"Output directory: {output_dir}")
    
    for i, item in tqdm(enumerate(data), total=len(data)):
        caption = item['captions']
        duration = item.get('duration', 3.0)
        
        # Optional: Skip if already exists? No, user might want to regenerate.
        
        try:
            with torch.no_grad():
                # Using parameters from infer.py
                audio = model.generate(
                    caption, 
                    steps=50, 
                    duration=duration, 
                    guidance_scale=4.5
                )
            
            output_path = os.path.join(output_dir, f"{i}.wav")
            torchaudio.save(output_path, audio, 44100)
            
        except Exception as e:
            print(f"Error generating index {i} ({caption[:30]}...): {e}")

if __name__ == "__main__":
    main()

