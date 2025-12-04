# Dataset Download

The dataset contains the following files (split archive):

| Filename | Size | Description |
|--------|------|------|
| datas.zip | 3.18 GB | Main split archive file |
| datas.z01 | 4.29 GB | Split archive part 1 |
| datas.z02 | 4.29 GB | Split archive part 2 |
| datas.z03 | 4.29 GB | Split archive part 3 |
| datas.z04 | 4.29 GB | Split archive part 4 |
| datas.z05 | 4.29 GB | Split archive part 5 |

**Total Size**: Approximately 24.7 GB

## Download Methods

### Method 1: Using huggingface-cli (Recommended)

1. **Install huggingface-hub**

```bash
pip install huggingface-hub
```

2. **Download Dataset**

```bash
huggingface-cli download AiEson2/SonicGauss --repo-type dataset --local-dir ./datas/
```

### Method 2: Using Python API

```python
from huggingface_hub import snapshot_download

# Download dataset
snapshot_download(
    repo_id="AiEson2/SonicGauss",
    repo_type="dataset",
    local_dir="./datas/"
)
```

## Extract Dataset

After downloading, you need to extract the split archive:

```bash
cd datas

# Extract on Linux/macOS
zip -F datas.zip --out combined.zip
unzip combined.zip

# Or use 7z (requires p7zip installation)
7z x datas.zip
```


### Data Structure

```
datas/
├── objectfolder_2.0_train.json
├── objectfolder_2.0_val.json
├── objectfolder_real_train.json
├── objectfolder_real_val.json
├── OF_Real/
│   └── ObjectFolderResults
└── OF_2.0/
    ├── audio_results/    # Impact sound recordings
    └── vision_results/   # 3DGS PLY files
```