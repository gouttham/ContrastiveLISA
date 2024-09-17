
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
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

import torch.nn as nn




args = my_utils.parse_args(sys.argv[1:])

args.exp_name = "NP_S1_cls_1_noCELoss_2"
args.const_seg_data="xbd"
args.version="./mbin/test/LLaVA-7B-Lightening-v1-1/"
args.constrative_dataset_dir="/localscratch/gna23/cd-datasets/"
args.dataset_dir="/localscratch/gna23/cd-datasets/"
args.use_scheduler = False
args.lr = 0.0001
args.epochs = 300
args.ce_loss_weight = 0.0

args.num_classes_per_sample = 5

# args.local_rank = "cpu"
# args.version = "mmaaz60/LLaVA-7B-Lightening-v1-1"
# args.vision_pretrained="./mbin/sam_vit_h_4b8939.pth"
# args.workers = 0



wandb = my_utils.wandb_init(args)


tokenizer = my_utils.get_tokenizer(args)


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

print("****** Loading Pretrained weights ******")
model.load_state_dict(torch.load("./runs/lisa-7b-xbd-14days/ckpt_model/pytorch_model.bin"),strict=False)



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
            if (isinstance(module, cls) and all([x not in name for x in ["visual_model","vision_tower","mm_projector","text_hidden_fcs",]])
                    and any([x in name for x in lora_target_modules])):
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


new_model = torch.load("./new_pipeline_model/NP_S1_cls_1_noCELoss/best.pth")
from collections import OrderedDict
corrected_model = OrderedDict()
for ech_lay in new_model:
    corrected_model[ech_lay.replace("base_model.model.","")] = new_model[ech_lay]
model.load_state_dict(corrected_model,strict=True)

for n, p in model.named_parameters():
    if any([x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "lora_"]]):
        print("n: ", n, "p.shape: ", p.shape)
        p.requires_grad = False

model.cross_attn.train()
for param in model.cross_attn.parameters():
    param.requires_grad = True





train_dataset = HybridDataset(
            args.constrative_dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=3000,
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

val_dataset = HybridDataset(
            args.constrative_dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=1200,
            precision=args.precision,
            image_size=args.image_size,
            num_classes_per_sample=args.num_classes_per_sample,
            exclude_val=True,
            dataset=args.dataset,
            sample_rate=[1],
            sem_seg_data=args.sem_seg_data,
            refer_seg_data=args.refer_seg_data,
            vqa_data=args.vqa_data,
            reason_seg_data=args.reason_seg_data,
            explanatory=args.explanatory,
            const_seg_data = args.const_seg_data
        )

val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,
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


# optimizer

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(args.beta1, args.beta2),
    weight_decay=0.0
)

if args.use_scheduler:
    # Learning rate scheduler setup
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=total_steps
    )


# Mixed precision training
# scaler = torch.cuda.amp.GradScaler(enabled=(args.precision in ["fp16", "bf16"]))


# val_dict = {}
# for ech in range(len(train_dataset)):
#     print(ech)
#     name = train_dataset.__getitem__(ech)[0][0]
#     val_dict[name] = int(val_dict.get(name,0)) + 1

# model.bfloat16()
# model.to(device=args.local_rank)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda:0")

# model = nn.DataParallel(model,device_ids=[0,1,2,3])

model = model.to(dtype=torch_dtype)
model = model.to(device)

clss = [
    "no building","undamaged building", "building with minor damage",
    "building with major damage", "completely destroyed building"
]

for epoch in range(args.epochs):

    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    model.train()

    # if epoch > 50:
    #     for n, p in model.named_parameters():
    #         if any([x in n for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "lora_"]]):
    #             print("n: ", n, "p.shape: ", p.shape)
    #             p.requires_grad = True
    #
    #     model.cross_attn.train()
    #     for param in model.cross_attn.parameters():
    #         param.requires_grad = True
    #
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 0.0001

    for train_idx,input_dict in enumerate(train_loader):
        print(train_idx,end='\r')
        optimizer.zero_grad()

        input_dict = my_utils.typecasting_inputs(input_dict,args,device)

        output_dict = model(**input_dict)
        loss = output_dict["loss"]

        loss.backward()

        # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()


        if args.use_scheduler:
            scheduler.step()

        losses.update(loss.item(), input_dict["images"].size(0))
        ce_losses.update(output_dict["ce_loss"].item(), input_dict["images"].size(0))
        mask_bce_losses.update(output_dict["mask_bce_loss"].item(), input_dict["images"].size(0))
        mask_dice_losses.update(output_dict["mask_dice_loss"].item(), input_dict["images"].size(0))
        mask_losses.update(output_dict["mask_loss"].item(), input_dict["images"].size(0))

        if train_idx % 100 ==0:
            print("epoch : ",epoch," iter : ",train_idx," loss : ",losses.avg)
            wandb.log({
                "train/loss":losses.avg,
                "train/ce_loss": ce_losses.avg,
                "train/mask_bce_loss": mask_bce_losses.avg,
                "train/mask_dice_loss": mask_dice_losses.avg,
                "train/mask_loss": mask_losses.avg,
                "train/lr": optimizer.param_groups[0]['lr']
            })
        # break

    print("Eval pipeline")
    model.eval()
    best_iou = 0

    iou_dict = {}
    for val_idx, input_dict in enumerate(val_loader):
        print(val_idx, end='\r')
        input_dict = my_utils.typecasting_inputs(input_dict, args, device)
        input_dict['inference'] = True

        save_name = input_dict['image_paths'][0][0].split('/')[-1]

        # with torch.no_grad():
        output_dict = model(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()

        log_exp_img = []
        image_logger = {}
        for mask_i, output_i, prmpt in zip(masks_list, output_list, input_dict['sampled_classes_list'][0]):
            pd = output_i.cpu().numpy().astype(np.uint8)
            gt = mask_i.cpu().numpy().astype(np.uint8)
            intersection = np.logical_and(pd, gt)
            union = np.logical_or(pd, gt)
            iou_score = np.sum(intersection) / np.sum(union)
            # print("iou_score : ",iou_score)
            iou_score = round(iou_score, 2)
            if np.isnan(iou_score):
                # print("Caught")
                iou_score = 0
            iou_lists = iou_dict.get(prmpt, [])
            iou_lists.append(iou_score)
            iou_dict[prmpt] = iou_lists


            pd = cv2.cvtColor(cv2.resize(pd, (224, 224)), cv2.COLOR_GRAY2RGB)
            gt = cv2.cvtColor(cv2.resize(gt, (224, 224)), cv2.COLOR_GRAY2RGB)

            sv_image = np.zeros([224, 448, 3], np.uint8)
            sv_image[:224, :224] = pd
            sv_image[:224, 224:] = gt

            sv_image = sv_image*255.0
            cv2.putText(sv_image, prmpt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            temp_name_i = str(clss.index(prmpt)) + "("+str(iou_score)+")_" + save_name
            image_logger[str(clss.index(prmpt))] = wandb.Image(sv_image, caption=f"{temp_name_i}")
            # log_exp_img.append(wandb.Image(sv_image, caption=f"{temp_name_i}"))

        for ech_cls in ['0','1','2','3','4']:
            if ech_cls in image_logger:
                log_exp_img.append(image_logger[ech_cls])
            else:
                log_exp_img.append(wandb.Image(sv_image * 0, caption=f"{ech_cls}_fillers"))


        wandb.log({"visualization": log_exp_img})


    total_avg = []
    wandb_dict = {}
    for ech in iou_dict:
        cur_avg = np.average(iou_dict[ech])
        wandb_dict['val/'+ech]=cur_avg
        total_avg.append(cur_avg)
    cur_iou = np.average(total_avg)
    wandb_dict['val/iou'] = np.average(total_avg)
    wandb.log(wandb_dict)


    ckpt_pth = os.path.join("./new_pipeline_model",args.exp_name)
    if not os.path.exists(ckpt_pth):
        os.makedirs(ckpt_pth)
        print(f"Directory '{ckpt_pth}' created.")

    if cur_iou>best_iou:
        torch.save(model.state_dict(), os.path.join(ckpt_pth,'best.pth'))