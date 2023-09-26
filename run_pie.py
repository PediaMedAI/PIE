import argparse
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms

from pipeline_stable_diffusion_pie import StableDiffusionPIEPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a PIE inference script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--finetuned_path",
        type=str,
        default=None,
        required=False,
        help="Path to domain specific finetuned unet from any healthcare text-to-image dataset",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="Path to the input instance images.",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        required=False,
        help="Path to mask.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument("--step", type=int, default=10, help="N in the paper, Number to images / steps for PIE generation")
    parser.add_argument("--strength", type=float, default=0.5, help="Roll back ratio garmma")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="guidance scale")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./simulation",
        help="The output directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )

    args = parser.parse_args()
    return args


def main(args):
    seed = args.seed
    set_all_seeds(seed)

    image_path = args.image_path
    mask_path = args.mask_path
    prompt = args.prompt

    model_id_or_path = args.pretrained_model_name_or_path
    finetuned_path = args.finetuned_path
    resolution = args.resolution
    ddim_times = args.step
    strength = args.strength
    guidance_scale = args.guidance_scale

    output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    device = "cuda"
    pipe = StableDiffusionPIEPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32, cache_dir="./checkpoints", safety_checker=None)
    if finetuned_path != None:
        unet = UNet2DConditionModel.from_pretrained(
            finetuned_path, subfolder="text_encoder"
        )
        pipe.unet = unet
    pipe = pipe.to(device)

    image_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution)
        ]
    )

    images = []
    step_i = 0
    init_image = Image.open(image_path).convert("RGB")   # The unedited image
    init_image = image_transforms(init_image)
    init_image.save(os.path.join(output_dir, str(step_i) + ".png"))

    if mask_path != None:
        mask = Image.open(mask_path).convert("RGB")
        mask = image_transforms(mask)
        mask.save(os.path.join(output_dir, "mask" + ".png"))
    else:
        mask = None

    step_i += 1
    img = init_image
    images.append(img)

    while step_i <= ddim_times:
        img = pipe(prompt=prompt, image=img, mask=mask, init_image=init_image, strength=strength, guidance_scale=guidance_scale).images[0]
        images.append(img)
        img.save(os.path.join(output_dir, str(step_i) + ".png"))
        step_i += 1

    duration = 1000
    images[0].save('output.gif', save_all=True, append_images=images[1:], duration=duration)

if __name__ == "__main__":
    args = parse_args()
    main(args)
