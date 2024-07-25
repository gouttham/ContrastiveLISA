#!/bin/bash
#SBATCH --account=def-amahdavi
#SBATCH --job-name=LISA_DALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=150G
#SBATCH --exclusive
#SBATCH --time=3-24:00:00

module load StdEnv/2020
module load cuda/11.0
module use cuda/11.0

cd ~/$projects/projects/def-amahdavi/gna23/LISA2/
source ./lisa_env/bin/activate

cd ~/$projects/projects/def-amahdavi/gna23/LISA5/
cp ./cd-datasets.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzvf cd-datasets.tar.gz ./

cd ~/$projects/projects/def-amahdavi/gna23/LISA5/
# 12.2



ls $SLURM_TMPDIR/cd-datasets/


deepspeed --master_port=24999 train_stage_1.py \
  --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
  --constrative \
  --constrative_dataset_dir=$SLURM_TMPDIR/cd-datasets/ \
  --dataset_dir=$SLURM_TMPDIR/cd-datasets/ \
  --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
  --vision-tower './mbin/clip-vit-large-patch14' \
  --sample_rates='1' \
  --epochs='200' \
  --dataset='contrastive_cd_dataset' \
  --exp_name="stage1" \
  --batch_size 4 \
  --steps_per_epoch 1541 \

echo "Job finished with exit code $? at: `date`"
