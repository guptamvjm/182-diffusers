output_dir=$1

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR=$output_dir #"path-to-save-model"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --concepts_list=./concepts_list.json \
  --num_class_images=150 \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --scale_lr --hflip  \
  --no_safe_serialization \
  --modifier_token "<nimsi>+<guptamvjm>" \
  --validation_prompt="<nimsi> and <guptamvjm> standing on a beach" \
  --report_to="wandb"