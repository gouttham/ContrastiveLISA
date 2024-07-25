#!/bin/bash
#SBATCH --account=def-amahdavi
#SBATCH --job-name=gouttham-LISA-constrative
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=200G
#SBATCH --exclusive
#SBATCH --time=01:00:00

cd ~/$projects/projects/def-amahdavi/gna23/LISA2/
cp ./cd-datasets.tar.gz $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar -xzvf cd-datasets.tar.gz ./


cd ~/$projects/projects/def-amahdavi/gna23/LISA2/
source ./lisa_env/bin/activate

cd ~/$projects/projects/def-amahdavi/gna23/LISA3/


module load cuda/11.0
module use cuda/11.0

ls $SLURM_TMPDIR/cd-datasets/

deepspeed --master_port=24999 train_ds.py \
  --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
  --constrative \
  --constrative_dataset_dir=$SLURM_TMPDIR/cd-datasets/ \
  --dataset_dir=$SLURM_TMPDIR/cd-datasets/ \
  --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
  --vision-tower './mbin/clip-vit-large-patch14' \
  --sample_rates='1' \
  --epochs='20' \
  --dataset='contrastive_cd_dataset' \
  --exp_name="lisa-7b-cd-const-7days_new" \
  --batch_size 4 \
  --steps_per_epoch 500 \
  --eval_only

echo "Job finished with exit code $? at: `date`"
