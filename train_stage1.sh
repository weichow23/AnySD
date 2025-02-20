# --resume_from_checkpoint ""\
PYTHON_PATH='./' accelerate launch --num_processes 8 --multi_gpu --gpu_ids "0,1,2,3,4,5,6,7" --mixed_precision="fp16" --main_process_port 21503 \
    anysd/train/train_stage1.py \
    --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --yaml_file="anysd/scripts/train_stage1.yaml" \
    --use_ema \
    --enable_xformers_memory_efficient_attention \
    --resolution=256 --random_flip \
    --train_batch_size=128 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=1000000000000000 \
    --checkpointing_steps=200 --checkpoints_total_limit=100000 \
    --valid_per_ckpt=1 \
    --learning_rate=1e-04 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=23 \
    --report_to=wandb \
    --output_dir="runs/stage1"
