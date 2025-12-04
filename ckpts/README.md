# Model Download

### Method 1: Using huggingface-cli (Recommended)

1. **Install huggingface-hub**

```bash
pip install huggingface-hub
```

2. **Download Model to Directory**

```bash
# Download the entire model repository to ./ckpts/Stage3/ directory
huggingface-cli download AiEson2/SonicGauss --local-dir ./ckpts/Stage3/
```

### Method 2: Using Python API

```python
from huggingface_hub import snapshot_download

# Download model
snapshot_download(
    repo_id="AiEson2/SonicGauss",
    local_dir="./ckpts/Stage3/"
)
```

## Directory Structure

After downloading, the `./ckpts/Stage3/` directory structure should be as follows:

```
ckpts/Stage3/
├── model.safetensors
├── model_1.safetensors
├── model_2.safetensors
├── model_3.safetensors
├── model_4.safetensors
├── optimizer.bin
├── random_states_0.pkl
└── scheduler.bin
```
