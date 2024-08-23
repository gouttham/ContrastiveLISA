
import os
import shutil
import sys
import time
from functools import partial

import numpy as np
import torch
import tqdm

from peft import LoraConfig, get_peft_model

from model.LISA_Dahi import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn,collate_fn3
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.cd_dataset import Contrastive_CD_Dataset
import cv2

import random

import my_utils




args = my_utils.parse_args(sys.argv[1:])

# args.local_rank = "cpu"
# args.exp_name = "cpu_dev"
# args.version = "mmaaz60/LLaVA-7B-Lightening-v1-1"
# args.vision_pretrained="./mbin/sam_vit_h_4b8939.pth"


wandb = my_utils.wandb_init(args)


tokenizer = my_utils.get_tokenizer(args)
'''
tokenizer.pad_token = tokenizer.unk_token
num_added_tokens = tokenizer.add_tokens("[SEG]")
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
if args.use_mm_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

model_args = my_utils.get_model_args(args)


torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half
model = LISAForCausalLM.from_pretrained(args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args)
model.train()

model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


model.enable_input_require_grads()
model.gradient_checkpointing_enable()

model.get_model().initialize_vision_modules(model.get_model().config)
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(dtype=torch_dtype, device=args.local_rank)

if args.constrative:
    if os.path.exists('./mbin/cross_attn_dahi.pt'):
        model.cross_attn.load_state_dict(torch.load('./mbin/cross_attn_dahi.pt'), strict=True)
        model.cross_attn.to(dtype=torch_dtype, device=args.local_rank)

model.get_model().initialize_lisa_modules(model.get_model().config)

for p in vision_tower.parameters():
    p.requires_grad = False
for p in model.get_model().mm_projector.parameters():
    p.requires_grad = False

conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

lora_r = args.lora_r
if lora_r > 0:
    def find_linear_layers(model, lora_target_modules):
        cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (
                    isinstance(module, cls)
                    and all(
                [
                    x not in name
                    for x in [
                    "visual_model",
                    "vision_tower",
                    "mm_projector",
                    "text_hidden_fcs",
                ]
                ]
            )
                    and any([x in name for x in lora_target_modules])
            ):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))


    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

model.resize_token_embeddings(len(tokenizer))

for n, p in model.named_parameters():
    if any([x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "lora_"]]):
        print("n: ", n, "p.shape: ", p.shape)
        p.requires_grad = False
        
model.cross_attn.train()
for param in model.cross_attn.parameters():
    param.requires_grad = True
'''

# args.workers = 0
# args.const_seg_data="xbd||s2looking"

train_dataset = HybridDataset(
            args.constrative_dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=10000,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=False,
            dataset=args.dataset,
            sample_rate=[1],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
            const_seg_data = args.const_seg_data
        )



train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False,
            collate_fn=partial(
                collate_fn3,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )


# val_dict = {}
# for ech in range(len(train_dataset)):
#     print(ech)
#     name = train_dataset.__getitem__(ech)[0][0]
#     val_dict[name] = int(val_dict.get(name,0)) + 1

for train_idx,input_dict in enumerate(train_loader):
    print(train_idx)
    # print(input_dict.keys())