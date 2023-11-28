
# Usage: bash learn.sh CLASS TOKEN OUTPUT_DIR
# class is the general class something is. For example, Nima is a person (?)
# token is the token you assign to what you're finetuning. For example, <nimsi>
# output_dir is self explanatory
# instance_data_folder is the name of the folder in the data directory that contains the finetuning set
# prompt is the end of the sentence: TOKEN CLASS _____. For example, if i wanted "<mycat> cat sitting in a bucket", I would put "sitting in a bucket" 

# Reminder: run: python3 retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
# if finetuning on a new class

#EXAMPLE: bash learn.sh person \<guptamvjm\> guptamvjm-experiment guptamvjm "at the beach"



class=$1
token=$2
output_dir=$3
instance_data_folder=$4
prompt=$5


export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR=$output_dir #"path-to-save-model"
export INSTANCE_DIR="./data/$instance_data_folder"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_$class/ \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt=$class --num_class_images=150 \
  --instance_prompt="photo of a $token $class"  \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=800 \
  --scale_lr --hflip  \
  --no_safe_serialization \
  --modifier_token "$token" \
  --validation_prompt="$token $class $prompt" \
  --report_to="wandb"

