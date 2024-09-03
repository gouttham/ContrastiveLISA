deepspeed --master_port=24999 train_stage_1.py \
  --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
  --constrative --constrative_dataset_dir=/localscratch/gna23/cd-datasets/ \
  --dataset_dir=/localscratch/gna23/cd-datasets/ \
  --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
  --vision-tower './mbin/clip-vit-large-patch14' \
  --sample_rates='1' \
  --epochs='200' \
  --dataset='contrastive_cd_dataset' \
  --exp_name="debbug" \
  --batch_size 4 \
  --steps_per_epoch 1541 \
  --const_seg_data "xbd"



#  --exp_name="stagev1_xbd_fixed_t13" \