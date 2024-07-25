#!/bin/bash
#SBATCH --account=def-amahdavi
#SBATCH --job-name=export
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --exclusive
#SBATCH --time=00:10:00



module load StdEnv/2020
module load cuda/11.0
module use cuda/11.0

cd ~/$projects/projects/def-amahdavi/gna23/LISA2/
source ./lisa_env/bin/activate




cd ~/$projects/projects/def-amahdavi/gna23/LISA5/



cd ./runs/b1_freeze_xbd_Dahi-FFFT2/ckpt_model && python zero_to_fp32.py . ./pytorch_model.bin



CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model_b1.py --version="./mbin/test/LLaVA-7B-Lightening-v1-1/" --weight="./runs/b1_freeze_xbd_Dahi-FFFT2/ckpt_model/pytorch_model.bin" --vision-tower './mbin/clip-vit-large-patch14' --save_path="./runs/b1_freeze_xbd_Dahi-FFFT2/export/" --constrative

