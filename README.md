<div align="center">

# SonicGauss

### Position-Aware Physical Sound Synthesis for 3D Gaussian Representations

<p align="center">
  <a href="https://chunshi.wang/SonicGauss/"><img src="https://img.shields.io/badge/🌐_Project_Page-SonicGauss-blue" alt="Project Page"></a>
  <a href="https://dl.acm.org/doi/10.1145/3746027.3755035"><img src="https://img.shields.io/badge/📄_Paper-ACM_MM_2025-green" alt="Paper"></a>
</p>

<p align="center">
  <strong>Chunshi Wang</strong>, <strong>Hongxing Li</strong>, <strong>Yawei Luo</strong>
  <br>
  <em>Zhejiang University</em>
</p>

</div>

---

## Overview

**SonicGauss** is a novel framework for synthesizing **impact sounds** from 3D Gaussian Splatting (3DGS) representations by leveraging their inherent geometric and material properties.

<p align="center">
  <img src="assets/pipeline.png" alt="SonicGauss Pipeline" width="100%">
</p>

---

## Getting Started

### Prerequisites

```bash
# Clone the repository with submodules
git clone https://github.com/AiEson/SonicGauss.git --recursive

cd SonicGauss

# Create conda environment
conda create -n sonicgauss python=3.10
conda activate sonicgauss
```

```txt
Please install the requirements for the following repositories: 
- TangoFlux: https://github.com/declare-lab/TangoFlux.git
- SplatFormer: https://github.com/ChenYutongTHU/SplatFormer.git
```

---

## Dataset

We train and evaluate on the [ObjectFolder](https://objectfolder.stanford.edu/) dataset:

- **ObjectFolder 2.0**: 1,000 objects with synthetic impact sounds
- **ObjectFolder Real**: Real-world recordings with diverse materials

### Dataset Download

**For detailed download instructions, please refer to [datas/README.md](datas/README.md)**

The dataset includes:
- ObjectFolder 2.0 and ObjectFolder Real datasets
- 3DGS PLY files
- Rendered images
- Impact sound recordings
- Training/validation split files

**Total Size**: ~24.7 GB

---

## Pretrained Models

**For model download instructions, please refer to [ckpts/README.md](ckpts/README.md)**

**Total Size**: ~3.5 GB

---

## Training

### Stage 1: Text-to-Audio Pretraining

```bash
bash stage1/train.sh
```

### Stage 2: Gaussian-to-Audio Learning

```bash
bash stage2/train_2_1.sh
bash stage2/train_2_2.sh
```

### Stage 3: Position-Aware Fine-tuning

```bash
bash stage3/train_3.sh
```

---

## Inference

Generate impact sounds from a 3DGS representation:

```python
from stage3.infer_3 import Stage3Inference

# Initialize model
model = Stage3Inference(
    model_path="./ckpts/stage3/best",
    config_path="./configs/stage3.yaml",
    device="cuda"
)

# Generate audio
audio = model.generate(
    ply_path="path/to/object.ply",
    position=[0.1, 0.2, 0.3],  # Impact position (x, y, z)
    duration=3.0,              # Audio duration in seconds
    steps=50,                  # Diffusion steps
    seed=42
)

# Save audio
import torchaudio
torchaudio.save("output.wav", audio, 44100)
```

### Inference with JSON input

```bash
python stage3/infer_3.py
```

Then the generated audio files can be found in the `generated/stage3` directory.


## Evaluation

We evaluate using standard audio generation metrics:

```bash
cd stage3
python eval.py
```

---

## Project Structure

```
SonicGauss/
├── configs/                 # Configuration files
│   ├── stage1.yaml
│   ├── stage2.yaml
│   └── stage3.yaml
├── stage1/                  # Text-to-Audio pretraining
├── stage2/                  # Gaussian-to-Audio learning
│   ├── train_2_1.py        # Contrastive learning
│   ├── train_2_2.py        # Generative training
│   └── common.py           # Shared utilities
├── stage3/                  # Position-aware fine-tuning
│   ├── train_3.py          # Training script
│   ├── infer_3.py          # Inference script
│   ├── eval.py             # Evaluation script
│   └── common.py           # Position encoder & fusion
├── SplatFormer/            # PTv3 backbone (submodule)
├── TangoFlux/              # Diffusion model (submodule)
├── audioldm_eval/          # Evaluation toolkit (submodule)
├── datas/                  # Dataset files
├── ckpts/                # Training checkpoints
└── assets/                 # Images and resources
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2025sonicgauss,
  title={SonicGauss: Position-Aware Physical Sound Synthesis for 3D Gaussian Representations},
  author={Wang, Chunshi and Li, Hongxing and Luo, Yawei},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia},
  pages={10886--10895},
  year={2025}
}
```

---

## Acknowledgements

We thank the authors of the following projects for their excellent work:

- **[ObjectFolder](https://objectfolder.stanford.edu/)** - Multi-modal object dataset with impact sounds
- **[TangoFlux](https://github.com/declare-lab/TangoFlux)** - Text-to-audio diffusion model
- **[SplatFormer](https://github.com/ChenYutongTHU/SplatFormer)** - 3DGS feature extraction with PointTransformer
- **[audioldm_eval](https://github.com/haoheliu/audioldm_eval)** - Audio generation evaluation toolkit

---

<div align="center">

**[🌐 Project Page](https://chunshi.wang/SonicGauss/)** · **[📄 Paper](https://dl.acm.org/doi/10.1145/3664647.3681603)** · **[🐛 Issues](https://github.com/AiEson/SonicGauss/issues)**

</div>
