import torch
import torch.nn as nn
from utils.dataset import HybridDataset_eval, ValDataset, collate_fn,collate_fn3
import argparse
import transformers
from functools import partial
import pickle
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from model.LISA_Dahi import LISAForCausalLM
from collections import OrderedDict


class Args(argparse.Namespace):
    local_rank='cpu'
    version='./mbin/test/LLaVA-7B-Lightening-v1-1/'
    vis_save_path='./vis_output'
    precision='bf16'
    image_size=1024
    model_max_length=512 
    lora_r=8
    vision_tower='./mbin/clip-vit-large-patch14'
    load_in_8bit=False
    load_in_4bit=False
    dataset='contrastive_cd_dataset'
    sample_rates='1'
    sem_seg_data='ade20k'
    refer_seg_data='refclef||refcoco||refcoco+||refcocog'
    vqa_data='llava_instruct_150k'
    reason_seg_data='ReasonSeg|train'
    val_dataset='ReasonSeg|val'
    log_base_dir='./runs'
    exp_name='debug'
    epochs=10
    steps_per_epoch=500
    batch_size=4
    grad_accumulation_steps=10
    val_batch_size=1
    workers=4
    lr=0.0003
    ce_loss_weight=1.0
    dice_loss_weight=0.5
    bce_loss_weight=2.0
    lora_alpha=16
    lora_dropout=0.05
    lora_target_modules='q_proj,v_proj'
    explanatory=0.1
    beta1=0.9
    beta2=0.95
    num_classes_per_sample=4
    exclude_val=False
    no_eval=False
    eval_only=False
    vision_pretrained='./mbin/sam_vit_h_4b8939.pth'
    out_dim=256
    resume=''
    print_freq=1
    start_epoch=0
    gradient_checkpointing=True
    train_mask_decoder=True
    use_mm_start_end=True
    auto_resume=True
    conv_type='llava_v1'
    log_dir='./runs/lisa'
    seg_token_idx=32000
    distributed=False
    constrative=True
    constrative_dataset_dir= '/home/gna23/projects/def-amahdavi/gna23/cd-datasets/'
    dataset_dir='/localscratch/gna23.27143499.0/cd-datasets/'
    world_size=4

args=Args()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.version,
    cache_dir=None,
    model_max_length=args.model_max_length,
    padding_side="right",
    use_fast=False,
)
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens("[SEG]")
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

if args.use_mm_start_end:
    tokenizer.add_tokens(
        [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    )

model_args = {
    "train_mask_decoder": args.train_mask_decoder,
    "out_dim": args.out_dim,
    "ce_loss_weight": args.ce_loss_weight,
    "dice_loss_weight": args.dice_loss_weight,
    "bce_loss_weight": args.bce_loss_weight,
    "seg_token_idx": args.seg_token_idx,
    "vision_pretrained": args.vision_pretrained,
    "vision_tower": args.vision_tower,
    "use_mm_start_end": args.use_mm_start_end,
    "constrative": args.constrative,
}

torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half
model = LISAForCausalLM.from_pretrained(
    args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype, device=args.local_rank)

if args.constrative:
    model.cross_attn.load_state_dict(torch.load('./mbin/cross_attn_dahi.pt'), strict=True)
    model.cross_attn.to(dtype=torch_dtype, device=args.local_rank)



model.get_model().initialize_lisa_modules(model.get_model().config)

model.load_state_dict(torch.load('./runs/b1_freeze_xbd_Dahi/ckpt_model/pytorch_model.bin'),strict=True)

# wt = torch.load('./runs/b1_freeze_xbd_Dahi-FFFT2/ckpt_model/pytorch_model.bin')


# model.load_state_dict(wt,strict=True)

# new_wt = OrderedDict()
# for ech in wt.keys():
#     new_wt['base_model.model.'+ech] = wt[ech]
    
# model.load_state_dict(new_wt,strict=True)
print('success!!!')




