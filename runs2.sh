deepspeed --master_port=24999 train_stage_v2.py \
  --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
  --constrative --constrative_dataset_dir=/localscratch/gna23/cd-datasets/ \
  --dataset_dir=/localscratch/gna23/cd-datasets/ \
  --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
  --vision-tower './mbin/clip-vit-large-patch14' \
  --sample_rates='1' \
  --epochs='1000' \
  --dataset='contrastive_cd_dataset' \
  --exp_name="new_stagev2_xbd" \
  --batch_size 4 \
  --steps_per_epoch 50 \
  --const_seg_data "xbd" \
  --num_classes_per_sample 5

#  stagev2_xbd_fixed_t13