accelerate launch \
    --config_file='configs/accelerator_config.yaml' \
    stage1/train.py \
    --checkpointing_steps="best" \
    --save_every=5 \
    --config='configs/stage1.yaml'