import os
import gc
import mimetypes
import shutil
import tempfile
from zipfile import ZipFile
from subprocess import call, check_call
from argparse import Namespace
import time
import torch
from typing import Set

from cog import BasePredictor, Input, Path
from lora_diffusion.cli_lora_pti import train as lora_train
from lora_diffusion import (
    UNET_DEFAULT_TARGET_REPLACE,
    TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
)

import sys


def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)


class Predictor(BasePredictor):
    def setup(self):
        # HACK: wait a little bit for instance to be ready
        time.sleep(10)
        check_call("nvidia-smi", shell=True)
        assert torch.cuda.is_available()

    def predict(
        self,
        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
        ),
        class_data: Path = Input(
            description="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
            default=None,
        ),
        num_class_images: int = Input(
            description="Minimal class images for prior preservation loss. If not enough images are provided in class_data, additional images will be"
            " sampled with class_prompt.",
            default=50,
        ),
        save_sample_prompt: str = Input(
            description="The prompt used to generate sample outputs to save.",
            default=None,
        ),
        save_sample_negative_prompt: str = Input(
            description="The negative prompt used to generate sample outputs to save.",
            default=None,
        ),
        n_save_sample: int = Input(
            description="The number of samples to save.",
            default=4,
        ),
        save_guidance_scale: float = Input(
            description="CFG for save sample.",
            default=7.5,
        ),
        save_infer_steps: int = Input(
            description="The number of inference steps for save sample.",
            default=50,
        ),
        with_prior_preservation: bool = Input(
            description="Flag to add prior preservation loss.",
            default=True,
        ),
        prior_loss_weight: float = Input(
            description="Weight of prior preservation loss.",
            default=1.0,
        ),
        seed: int = Input(description="A seed for reproducible training", default=1337),
        resolution: int = Input(
            description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=512,
        ),
        center_crop: bool = Input(
            description="Whether to center crop images before resizing to resolution",
            default=False,
        ),
        train_text_encoder: bool = Input(
            description="Whether to train the text encoder",
            default=True,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=1,
        ),
        sample_batch_size: int = Input(
            description="Batch size (per device) for sampling images.",
            default=4,
        ),
        num_train_epochs: int = Input(default=1),
        max_train_steps: int = Input(
            description="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
            default=2000,
        ),
        gradient_accumulation_steps: int = Input(
            description="Number of updates steps to accumulate before performing a backward/update pass.",
            default=4,
        ),
        gradient_checkpointing: bool = Input(
            description="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
            default=False,
        ),
        scale_lr: bool = Input(
            description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
            default=False,
        ),
        lr_scheduler: str = Input(
            description="The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="constant",
        ),
        lr_warmup_steps: int = Input(
            description="Number of steps for the warmup in the lr scheduler.",
            default=0,
        ),
        use_8bit_adam: bool = Input(
            description="Whether or not to use 8-bit Adam from bitsandbytes.",
            default=False,
        ),
        max_grad_norm: float = Input(
            default=1.0,
            description="Max gradient norm.",
        ),
        clip_ti_decay: bool = Input(
            default=True,
            description="Whether or not to clip the TI decay to be between 0 and 1.",
        ),
        color_jitter: bool = Input(
            default=True,
            description="Whether or not to use color jitter.",
        ),
        continue_inversion: bool = Input(
            default=False,
            description="Whether or not to continue an inversion.",
        ),
        continue_inversion_lr: float = Input(
            default=1e-4,
            description="The learning rate for continuing an inversion.",
        ),
        device: str = Input(
            default="cuda:0",
            description="The device to use. Can be 'cuda' or 'cpu'.",
        ),
        initializer_tokens: str = Input(
            default=None,
            description="The tokens to use for the initializer. If not provided, will randomly initialize from gaussian N(0,0.017^2)",
        ),
        learning_rate_text: float = Input(
            default=1e-4,
            description="The learning rate for the text encoder.",
        ),
        learning_rate_ti: float = Input(
            default=5e-4,
            description="The learning rate for the TI.",
        ),
        learning_rate_unet: float = Input(
            default=1e-5,
            description="The learning rate for the unet.",
        ),
        lora_rank: int = Input(
            default=4,
            description="The rank for the LORA loss.",
        ),
        lr_scheduler_lora: str = Input(
            description="The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="constant",
        ),
        lr_warmup_steps_lora: int = Input(
            description="Number of steps for the warmup in the lr scheduler.",
            default=0,
        ),
        mask_temperature: float = Input(
            default=1.0,
            description="The temperature for the mask.",
        ),
        max_train_steps_ti: int = Input(
            default=500,
            description="The maximum number of training steps for the TI.",
        ),
        max_train_steps_tuning: int = Input(
            default=1000,
            description="The maximum number of training steps for the tuning.",
        ),
        perform_inversion: bool = Input(
            default=True,
            description="Whether or not to perform an inversion.",
        ),
        placeholder_token_at_data: str = Input(
            default=None,
            description="Whether or not to use a placeholder token at the data.",
        ),
        placeholder_tokens: str = Input(
            default="<s1>|<s2>",
            description="The placeholder tokens to use for the initializer. If not provided, will use the first tokens of the data.",
        ),
        save_steps: int = Input(
            default=100,
            description="The number of steps between saving checkpoints.",
        ),
        use_extended_lora: bool = Input(
            default=False,
            description="Whether or not to use the extended LORA loss.",
        ),
        use_face_segmentation_condition: bool = Input(
            default=True,
            description="Whether or not to use the face segmentation condition.",
        ),
        use_mask_captioned_data: bool = Input(
            default=False,
            description="Whether or not to use the mask captioned data.",
        ),
        use_template: str = Input(
            default="object",
            description="The template to use for the inversion.",
        ),
        weight_decay_lora: float = Input(
            default=0.001,
            description="The weight decay for the LORA loss.",
        ),
        weight_decay_ti: float = Input(
            default=0.00,
            description="The weight decay for the TI.",
        ),
    ) -> Path:
        # check that the data is provided
        cog_instance_data = "cog_instance_data"
        cog_class_data = "cog_class_data"
        cog_output_dir = "checkpoints"
        for path in [cog_instance_data, cog_output_dir, cog_class_data]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # extract zip contents, flattening any paths present within it
        with ZipFile(str(instance_data), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, cog_instance_data)

        if class_data is not None:
            with ZipFile(str(class_data), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                        "__MACOSX"
                    ):
                        continue
                    mt = mimetypes.guess_type(zip_info.filename)
                    if mt and mt[0] and mt[0].startswith("image/"):
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, cog_class_data)
        # if class_data is not None:
        #     with ZipFile(str(class_data), "r") as zip_ref:
        #         for zip_info in zip_ref.infolist():
        #             if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
        #                 "__MACOSX"
        #             ):
        #                 continue
        #             mt = mimetypes.guess_type(zip_info.filename)
        #             if mt and mt[0] and mt[0].startswith("image/"):
        #                 zip_info.filename = os.path.basename(zip_info.filename)
        #                 zip_ref.extract(zip_info, cog_class_data)

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": "./stable-diffusion-v1-5-cache",
            "pretrained_vae_name_or_path": None,
            "revision": None,
            # "tokenizer_name": None,
            "instance_data_dir": cog_instance_data,
            "class_data_dir": cog_class_data,
            # "save_sample_prompt": save_sample_prompt,
            # "save_sample_negative_prompt": save_sample_negative_prompt,
            # "n_save_sample": n_save_sample,
            # "save_guidance_scale": save_guidance_scale,
            # "save_infer_steps": save_infer_steps,
            # "with_prior_preservation": with_prior_preservation,
            # "prior_loss_weight": prior_loss_weight,
            # "num_class_images": num_class_images,
            "seed": seed,
            "resolution": resolution,
            # "center_crop": center_crop,
            "train_text_encoder": train_text_encoder,
            "train_batch_size": train_batch_size,
            # "sample_batch_size": sample_batch_size,
            # "num_train_epochs": num_train_epochs,
            # "max_train_steps": max_train_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "scale_lr": scale_lr,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "use_8bit_adam": use_8bit_adam,
            # "max_grad_norm": max_grad_norm,
            # "push_to_hub": False,
            # "hub_token": None,
            # "hub_model_id": None,
            # "save_interval": 10000,  # not used
            # "save_min_steps": 0,
            "mixed_precision": "fp16",
            # "not_cache_latents": False,
            # "local_rank": -1,
            "output_dir": cog_output_dir,
            # "concepts_list": None,
            # "logging_dir": "logs",
            # "log_interval": 10,
            # "hflip": False,
            "clip_ti_decay": clip_ti_decay,
            "color_jitter": color_jitter,
            "continue_inversion": continue_inversion,
            "continue_inversion_lr": continue_inversion_lr,
            "device": device,
            "weight_decay_ti": weight_decay_ti,
            "initializer_tokens": initializer_tokens,
            "learning_rate_text": learning_rate_text,
            "learning_rate_ti": learning_rate_ti,
            "learning_rate_unet": learning_rate_unet,
            "lora_clip_target_modules": TEXT_ENCODER_DEFAULT_TARGET_REPLACE,
            "lora_rank": lora_rank,
            "lora_unet_target_modules": UNET_DEFAULT_TARGET_REPLACE,
            "lr_scheduler_lora": lr_scheduler_lora,
            "lr_warmup_steps_lora": lr_warmup_steps_lora,
            # "mask_temperature": mask_temperature,
            "max_train_steps_ti": max_train_steps_ti,
            "max_train_steps_tuning": max_train_steps_tuning,
            "perform_inversion": perform_inversion,
            "placeholder_token_at_data": placeholder_token_at_data,
            "placeholder_tokens": placeholder_tokens,
            "save_steps": save_steps,
            # "use_extended_lora": use_extended_lora,
            "use_face_segmentation_condition": use_face_segmentation_condition,
            # "use_mask_captioned_data": use_mask_captioned_data,
            "use_template": use_template,
            "weight_decay_lora": weight_decay_lora,
            "weight_decay_ti": weight_decay_ti,
        }

        lora_train(**args)

        gc.collect()
        torch.cuda.empty_cache()
        call("nvidia-smi")

        out_path = "output.zip"

        directory = Path(cog_output_dir)
        with ZipFile(out_path, "w") as zip:
            for file_path in directory.rglob("*"):
                print(file_path)
                zip.write(file_path, arcname=file_path.relative_to(directory))

        return Path(out_path)
