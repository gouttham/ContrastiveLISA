import argparse
import os
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

# from model.LISA2 import LISAForCausalLM
from model.LISA_Dahi import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn,collate_fn3
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.cd_dataset import Contrastive_CD_Dataset
import cv2
import pdb
import metrics


# deepspeed --master_port=24999 inference.py \
#   --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
#   --constrative \
#   --constrative_dataset_dir=$SLURM_TMPDIR/cd-datasets/ \
#   --dataset_dir=$SLURM_TMPDIR/cd-datasets/ \
#   --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
#   --vision-tower './mbin/clip-vit-large-patch14' \
#   --sample_rates='1' \
#   --epochs='200' \
#   --dataset='contrastive_cd_dataset' \
#   --exp_name="inference" \
#   --batch_size 4 \
#   --steps_per_epoch 1541 \
#   --const_seg_data s2looking \

# deepspeed --master_port=24999 inference.py \
#   --version=./mbin/test/LLaVA-7B-Lightening-v1-1/ \
#   --constrative \
#   --constrative_dataset_dir=/localscratch/gna23/cd-datasets/ \
#   --dataset_dir=/localscratch/gna23/cd-datasets/ \
#   --vision_pretrained=./mbin/sam_vit_h_4b8939.pth \
#   --vision-tower './mbin/clip-vit-large-patch14' \
#   --sample_rates='1' \
#   --epochs='200' \
#   --dataset='contrastive_cd_dataset' \
#   --exp_name="inference" \
#   --batch_size 4 \
#   --steps_per_epoch 1541 \
#   --const_seg_data xbd \

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="sem_seg||refer_seg||vqa||reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="9,3,3,1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train", type=str)
    parser.add_argument("--val_dataset", default="ReasonSeg|val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--constrative_dataset_dir", default="./cd-datasets", type=str)
    parser.add_argument("--constrative", action="store_true", default=True)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=5, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    parser.add_argument(
        "--const_seg_data", default="xbd||s2looking||levircd||levircdplus||3dcd", type=str
    )

    return parser.parse_args(args)


def main(args):
    epoch = 1
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
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

        
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False
    
            
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

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

    # make text_hidden_fcs, mask_decoder, lm_head, embed_tokens trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "lora_"]
            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = False
            
    model.cross_attn.train()
    for param in model.cross_attn.parameters():
        param.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1


    val_dataset = HybridDataset(
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
            const_seg_data = args.const_seg_data)

    print(
            f"validating with {len(val_dataset)} examples."
        )


    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

        
    # pdb.set_trace()
    print("****** Loading Pretrained weights ******")
    model.load_state_dict(torch.load("./runs/lisa-7b-xbd-14days/ckpt_model/pytorch_model.bin"), strict=False)
    # model.load_state_dict(torch.load('./runs/stagev1_xbd_fixed_t13/pytorch_model.bin'), strict=True)
    model.load_state_dict(torch.load('./runs/stagev2_xbd_fixed_t13/ckpt_model/pytorch_model.bin'),strict=True)
    
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=val_dataset,
        collate_fn=partial(
            collate_fn3,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )
    


    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, shuffle=False, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=val_sampler,
        collate_fn=partial(
            collate_fn3,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
    )

    
    save_dir_iou = validate(val_loader, model_engine, epoch, writer, args)
    metrics.get_iou(save_dir_iou)



def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    save_dir_iou = "./visualize_iou_"+args.exp_name+"/"
    save_dir = "./visualize_" + args.exp_name + "/"
    try:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    except:
        print("Error deleting ",save_dir)

    try:
        if os.path.exists(save_dir_iou):
            shutil.rmtree(save_dir_iou)
        os.makedirs(save_dir_iou)
    except:
        print("Error deleting ",save_dir_iou)

    model_engine.eval()
    

    ctr = 0
    for input_dict in tqdm.tqdm(val_loader):
        ctr+=1
        torch.cuda.empty_cache()

        save_name = input_dict['image_paths'][0][0].split('/')[-1]
        
        input_dict = dict_to_cuda(input_dict)
        # pdb.set_trace()
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict['inference'] = True
        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        intersection, union, acc_iou = 0.0, 0.0, 0.0

        for mask_i, output_i, prmpt in zip(masks_list, output_list, input_dict['sampled_classes_list'][0]):
            im_array_pred = output_i.cpu().numpy().astype(np.uint8)
            im_array_gt = mask_i.cpu().numpy().astype(np.uint8)

            cv2.imwrite(os.path.join(save_dir_iou, str(prmpt) + "_pd_" + save_name), im_array_pred)
            cv2.imwrite(os.path.join(save_dir_iou, str(prmpt) + "_gt_" + save_name), im_array_gt)

        local_ctr = 0
        for mask_i, output_i,prmpt in zip(masks_list, output_list,input_dict['sampled_classes_list'][0]):

            im_array = output_i.cpu().numpy()
            print("min : ", im_array.min())
            print("max : ", im_array.max())

            im_array = ((im_array - im_array.min()) / (im_array.max() - im_array.min())) * 255
            im_array_pred = im_array.astype(np.uint8)

            im_array = mask_i.cpu().numpy()
            im_array = ((im_array - im_array.min()) / (im_array.max() - im_array.min())) * 255
            im_array_gt = im_array.astype(np.uint8)

            pd = cv2.cvtColor(cv2.resize(im_array_pred, (224, 224)),cv2.COLOR_GRAY2RGB)
            gt = cv2.cvtColor(cv2.resize(im_array_gt, (224, 224)),cv2.COLOR_GRAY2RGB)

            sv_image = np.zeros([224, 448, 3], np.uint8)
            sv_image[:224, :224] = pd
            sv_image[:224, 224:] = gt

            cv2.putText(sv_image, prmpt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)

            cv2.imwrite(os.path.join(save_dir, str(local_ctr)+"_" +save_name), sv_image)

            local_ctr += 1

            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

    return save_dir_iou


if __name__ == "__main__":
    main(sys.argv[1:])

