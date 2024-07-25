import argparse
import os
import sys
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

# from model.LISA2 import LISAForCausalLM
from model.LISA_Dahi import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument(
        "--version", default="/home/gna23/projects/def-amahdavi/gna23/LISA5/runs/b1_freeze_xbd_Dahi-FFFT2/export")
    parser.add_argument("--vis_save_path", default="/home/gna23/projects/def-amahdavi/gna23/LISA5/vis_output/b1_freeze_xbd_Dahi/", type=str)
    parser.add_argument("--data_dir", default="/home/gna23/projects/def-amahdavi/gna23/xbd_test/images/")
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024,
                        type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default='./mbin/clip-vit-large-patch14', type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end",
                        action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def load_imgs_s2(dataset_root):
    unchecked_image_1_fns = sorted(
        os.listdir(os.path.join(dataset_root, "Image1"))
    )
    unchecked_image_1_fns = [
        os.path.join(dataset_root, "Image1", fn)
        for fn in unchecked_image_1_fns
    ]

    image_1_fns = []

    for fn in unchecked_image_1_fns:
        if fn.endswith(".png"):
            image_1_fns.append(fn)

    image_2_fns = [fn.replace("Image1", "Image2") for fn in image_1_fns]
    image_2_fns = [fn.replace("Image1", "Image2") for fn in image_1_fns]

    # checking if loaded correctly
    for i in range(len(image_1_fns)):
        print(image_1_fns[i])
        assert (os.path.isfile(image_1_fns[i]))
        assert (os.path.isfile(image_2_fns[i]))
    print(len(image_1_fns))
    return image_1_fns, image_2_fns


def load_imgs_xbd(dataset_root, pre_img_keyword="pre_disaster", post_img_keyword="post_disaster"):
    images_root = dataset_root
    if not images_root.endswith("/"):
        images_root += "/"
    in_filenames = glob.glob(images_root + "*.png")

    in_filenames = sorted(in_filenames)
    pre_images = [fn for fn in in_filenames if pre_img_keyword in fn]
    post_images = [fn for fn in in_filenames if post_img_keyword in fn]

    # checking if loaded correctly
    assert (len(pre_images) == len(post_images))
    for i in range(len(pre_images)):
        assert (pre_images[i].replace(pre_img_keyword,
                post_img_keyword) == post_images[i])

    print(pre_images, post_images)
    return pre_images, post_images


def color_mask(mask):
    uniques = np.unique(mask)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    print("colors", uniques)
    unique_to_color = {0: [0, 0, 0], 1: [255, 0, 0], 2: [
        0, 255, 0], 3: [0, 0, 255], 4: [255, 255, 255]}

    if uniques[-1] == 1:
        unique_to_color = {0: [0, 0, 0], 1: [255, 255, 255]}

    for u in unique_to_color.keys():
        colored_mask[mask == u] = unique_to_color[u]
    return colored_mask


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)
    print(args)
    # ********** Create and initialize model **********
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer(
        "[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype,"constrative":True}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (
            not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(
        model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    # *************************************************

    xbd_prompts = [
        "can you segment undamaged region from this image?",
        "can you segment region with minor damage from this image?",
        "can you segment region with major damage from this image?",
        "can you segment completely destroyed region from this image?"
    ]
    print(xbd_prompts)
    print('args.data_dir : ',args.data_dir)
    pre_images, post_images = load_imgs_xbd(args.data_dir)
    print(len(pre_images))
    print(len(post_images))
    for i in range(len(pre_images)):
        print('i : ',i)
        pre_img_path = pre_images[i]
        post_img_path = post_images[i]
        save_path = "{}/{}_mask_{}.jpg".format(
            args.vis_save_path, post_img_path.split("/")[-1].split(".")[0], 0
        )

        # if os.path.isfile(save_path):
        #     continue

        mask = None

        prompts = xbd_prompts  # NOTE: change it to custom_prompts if you want to change the prompts
        for prompt_idx, prompt in enumerate(prompts):
            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            # prompt = input("Please input your prompt: ")

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            # image_path = input("Please input the image path: ")
            if not os.path.exists(pre_img_path):
                print("File not found in {}".format(pre_img_path))
                break
            if not os.path.exists(post_img_path):
                print("File not found in {}".format(post_img_path))
                break

            pre_image_np = cv2.imread(pre_img_path)
            post_image_np = cv2.imread(post_img_path)
            pre_image_np = cv2.cvtColor(pre_image_np, cv2.COLOR_BGR2RGB)
            post_image_np = cv2.cvtColor(post_image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [post_image_np.shape[:2]]

            pre_image_clip = (
                clip_image_processor.preprocess(pre_image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )

            post_image_clip = (
                clip_image_processor.preprocess(post_image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )

            if args.precision == "bf16":
                pre_image_clip = pre_image_clip.bfloat16()
                post_image_clip = post_image_clip.bfloat16()
            elif args.precision == "fp16":
                pre_image_clip = pre_image_clip.half()
                post_image_clip = post_image_clip.half()
            else:
                pre_image_clip = pre_image_clip.float()
                post_image_clip = post_image_clip.float()

            pre_image = transform.apply_image(pre_image_np)
            post_image = transform.apply_image(post_image_np)
            resize_list = [pre_image.shape[:2]]

            pre_image = (
                preprocess(torch.from_numpy(
                    pre_image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )

            post_image = (
                preprocess(torch.from_numpy(
                    post_image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )

            if args.precision == "bf16":
                pre_image = pre_image.bfloat16()
                post_image = post_image.bfloat16()
            elif args.precision == "fp16":
                pre_image = pre_image.half()
                post_image = post_image.bfloat16()
            else:
                pre_image = pre_image.float()
                post_image = post_image.bfloat16()

            print("prompt : ,", prompt)
            input_ids = tokenizer_image_token(
                prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            print("input_ids : ,", input_ids)
            output_ids, pred_masks = model.evaluate(
                post_image_clip,
                (pre_image, post_image),
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

            text_output = tokenizer.decode(
                output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            print("text_output: ", text_output)

            for i, pred_mask in enumerate(pred_masks):
                if pred_mask.shape[0] == 0:
                    continue

                pred_mask = pred_mask.detach().cpu().numpy()[0]
                print("UNIQUE=====", np.unique(pred_mask))
                pred_mask = (pred_mask > 0)
                print("prompt idx", prompt_idx)
                if mask is None:
                    mask = pred_mask * (prompt_idx)
                else:
                    mask = np.maximum(mask, pred_mask * (prompt_idx))

        save_path = "{}/{}_mask_{}.jpg".format(
            args.vis_save_path, post_img_path.split("/")[-1].split(".")[0], i
        )
        save_path_colored = "{}/{}_colored_mask_{}.jpg".format(
            args.vis_save_path, post_img_path.split("/")[-1].split(".")[0], i
        )

        cv2.imwrite(save_path, mask)
        print("{} has been saved.".format(save_path))

        colored_mask = color_mask(mask)
        cv2.imwrite(save_path_colored, colored_mask)
        print("{} has been saved.".format(save_path_colored))


if __name__ == "__main__":
    main(sys.argv[1:])