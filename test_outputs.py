import os

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from CorridorKeyModule.inference_engine import CorridorKeyEngine

# there is some compile weirdness when generating the images
torch._dynamo.config.cache_size_limit = 1024


def load_engine(img_size, precision, mixed_precision):
    return CorridorKeyEngine(
        checkpoint_path="CorridorKeyModule/checkpoints/CorridorKey_v1.0.pth",
        img_size=img_size,
        device="cuda",
        model_precision=precision,
        mixed_precision=mixed_precision,
    )


def generate_test_images(img_path, mask_path):
    img = read_image(img_path).permute(1, 2, 0).numpy()
    mask = read_image(mask_path).permute(1, 2, 0).numpy()
    img_sizes = [512, 1024, 2048]
    precisions = [torch.float16, torch.float32, torch.float64]
    for precision in precisions:
        for img_size in img_sizes:
            # Reset stats
            torch.cuda.reset_peak_memory_stats()

            if precision == torch.float64 and img_size > 1024:
                continue

            engine = load_engine(img_size, precision)
            out = engine.process_frame(img, mask)

            save_image(
                torch.from_numpy(out["fg"]).permute(2, 0, 1),
                f"./Output/foreground_{img_size}_{str(precision)[-7:]}.png",
            )
            save_image(
                torch.from_numpy(out["alpha"]).permute(2, 0, 1), f"./Output/alpha_{img_size}_{str(precision)[-7:]}.png"
            )

            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Precision: {precision}, Image Size: {img_size}, Peak VRAM: {peak_vram:.2f} GB")


def compare_implementations(src, comparison, output_dir="./Output"):
    for _, _, files in os.walk(src):
        for file in files:
            src_img = read_image(str(os.path.join(src, file))).float()
            comp_img = read_image(str(os.path.join(comparison, file))).float()

            is_mask = src_img.shape[0] == 1 or (src_img[0] == src_img[1]).all() and (src_img[1] == src_img[2]).all()

            difference = (src_img - comp_img).float() / 255

            if is_mask:
                difference = difference[0].unsqueeze(0)
                difference = torch.cat(
                    (difference.clamp(-1, 0).abs(), difference.clamp(0, 1), torch.zeros_like(difference)), dim=0
                )
                print(difference.shape)
                print(difference.min(), difference.max())
            else:
                difference = difference.abs()

            os.makedirs(output_dir, exist_ok=True)

            save_image(difference, f"{output_dir}/diff_{file}")


def compare_floating_point_precision(folder, ref="float64"):
    for _, _, files in os.walk(folder):
        for file in files:
            name, fmt = file.split(".")
            typ, img_size, precision = name.split("_")
            if precision != ref:
                continue
            float_ref = read_image(str(os.path.join(folder, file))).float()
            float_32 = read_image(str(os.path.join(folder, f"{typ}_{img_size}_float32.{fmt}"))).float()

            is_mask = typ == "alpha"

            difference = (float_ref - float_32).float() / 255

            if is_mask:
                difference = difference[0].unsqueeze(0)
                difference = torch.cat(
                    (difference.clamp(-1, 0).abs(), difference.clamp(0, 1), torch.zeros_like(difference)), dim=0
                )
            else:
                difference = difference.abs()
            print(
                is_mask,
                difference.min().item(),
                difference.max().item(),
                difference.mean().item(),
                difference.median().item(),
            )

            save_image(difference, f"./Output/prec_{ref}_{typ}_{img_size}.{fmt}")


def compare_img_sizes(folder, ref=1024):
    for _, _, files in os.walk(folder):
        for file in files:
            name, fmt = file.split(".")
            typ, img_size, precision = name.split("_")
            if img_size != str(ref):
                continue
            if precision == "float64":
                continue
            img_ref = read_image(str(os.path.join(folder, file))).float()
            img_2048 = read_image(str(os.path.join(folder, f"{typ}_2048_{precision}.{fmt}"))).float()

            is_mask = typ == "alpha"

            difference = (img_ref - img_2048).float() / 255

            if is_mask:
                difference = difference[0].unsqueeze(0)
                difference = torch.cat(
                    (difference.clamp(-1, 0).abs(), difference.clamp(0, 1), torch.zeros_like(difference)), dim=0
                )
            else:
                difference = difference.abs()
            print(
                is_mask,
                difference.min().item(),
                difference.max().item(),
                difference.mean().item(),
                difference.median().item(),
            )

            save_image(difference, f"./Output/img_{ref}_{typ}_{precision}.{fmt}")


if __name__ == "__main__":
    compare_implementations("./Output/base/Comp", "./Output/compare/Comp", output_dir="./Output/diff")
