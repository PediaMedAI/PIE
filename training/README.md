## Description

Finetuning stable diffusion base model on each of the three medical imaging dataset


## Usage
Script for finetuning on Chexpert:
```bash
#!/bin/bash
source ~/.bashrc
conda activate diffusion
export HF_HOME=./cache/
torchrun --nproc_per_node=8 train.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
            --instance_data_dir="<instance_data_dir>" \
            --output_dir="<output_dir>" \
            --instance_prompt="A chest X-ray image" \
            --resolution=512 \
            --train_batch_size=8 \
            --gradient_accumulation_steps=2 \
            --learning_rate=5e-5 \
            --lr_warmup_steps=1000 \
            --max_train_steps=20000 \
            --lr_scheduler "cosine" \
            --checkpoints_total_limit 2 \
            --gradient_checkpointing \
            --mixed_precision bf16 \
            --center_crop \
            --instance_dataset chexpert \
            --checkpointing_steps 5000
```
Script for finetuning on Retinopathy:
```bash
#!/bin/bash
source ~/.bashrc
conda activate diffusion
echo $CUDA_VISIBLE_DEVICES
export HF_HOME=./cache/
torchrun --nproc_per_node=8 --master_port 0 train_dreambooth.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
            --instance_data_dir="<instance_data_dir>" \
            --output_dir="<output_dir>" \
            --instance_prompt="A retinopathy image" \
            --resolution=512 \
            --train_batch_size=8 \
            --gradient_accumulation_steps=2 \
            --learning_rate=5e-5 \
            --lr_warmup_steps=1000 \
            --max_train_steps=20000 \
            --lr_scheduler "cosine" \
            --checkpoints_total_limit 2 \
            --gradient_checkpointing \
            --mixed_precision bf16 \
            --center_crop \
            --instance_dataset retinopathy \
            --checkpointing_steps 5000
```

Script for finetuning on ISIC:
```bash
#!/bin/bash
source ~/.bashrc
conda activate diffusion
export HF_HOME=./cache/
torchrun --nproc_per_node=8 train_dreambooth.py \
            --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
            --instance_data_dir="<instance_data_dir>" \
            --output_dir="<output_dir>" \
            --instance_prompt="An image of skin" \
            --resolution=512 \
            --train_batch_size=8 \
            --gradient_accumulation_steps=2 \
            --learning_rate=5e-5 \
            --lr_warmup_steps=1000 \
            --max_train_steps=20000 \
            --lr_scheduler "cosine" \
            --checkpoints_total_limit 2 \
            --gradient_checkpointing \
            --mixed_precision bf16 \
            --center_crop \
            --instance_dataset isic \
            --checkpointing_steps 5000
```

## License

This project is licensed under the [MIT License](LICENSE).
