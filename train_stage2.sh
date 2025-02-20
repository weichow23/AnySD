TYPE='viewpoint'

PYTHON_PATH='./'  accelerate launch --num_processes 8 --multi_gpu --gpu_ids "0,1,2,3,4,5,6,7" --mixed_precision "fp16" --main_process_port 21505 \
  anysd/train/train_stage2.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --image_encoder_path="WeiChow/AnySD" \
  --yaml_file="anysd/scripts/train_stage2_${TYPE}.yaml" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=64 \
  --learning_rate=1e-04 \
  --valid_per_ckpt=1 \
  --adam_weight_decay=0.01 \
  --resume_the_unet='WeiChow/AnySD' \
  --output_dir="runs/stage2-${TYPE}" \
  --checkpointing_steps=300 \
  --dataset_repeat_num=500 \
  --seed=23 \
  --report_to=wandb \
  --expert_name="${TYPE}"
