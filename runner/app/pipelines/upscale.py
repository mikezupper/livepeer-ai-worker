import logging
import os
from typing import List, Optional, Tuple

import PIL
import torch
from app.pipelines.base import Pipeline
from app.pipelines.utils import (
    SafetyChecker,
    get_model_dir,
    get_torch_device,
    is_lightning_model,
    is_turbo_model,
)
from diffusers import StableDiffusionUpscalePipeline
from huggingface_hub import file_download
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class UpscalePipeline(Pipeline):
    def __init__(self, model_id: str):
        self.model_id = model_id
        kwargs = {"cache_dir": get_model_dir()}

        torch_device = get_torch_device()
        folder_name = file_download.repo_folder_name(
            repo_id=model_id, repo_type="model"
        )
        folder_path = os.path.join(get_model_dir(), folder_name)
        has_fp16_variant = any(
            ".fp16.safetensors" in fname
            for _, _, files in os.walk(folder_path)
            for fname in files
        )
        if torch_device != "cpu" and has_fp16_variant:
            logger.info("UpscalePipeline loading fp16 variant for %s", model_id)

            kwargs["torch_dtype"] = torch.float16
            kwargs["variant"] = "fp16"

        self.ldm = StableDiffusionUpscalePipeline.from_pretrained(
            model_id, **kwargs
        ).to(torch_device)

        sfast_enabled = os.getenv("SFAST", "").strip().lower() == "true"
        deepcache_enabled = os.getenv("DEEPCACHE", "").strip().lower() == "true"
        if sfast_enabled and deepcache_enabled:
            logger.warning(
                "Both 'SFAST' and 'DEEPCACHE' are enabled. This is not recommended "
                "as it may lead to suboptimal performance. Please disable one of them."
            )

        if sfast_enabled:
            logger.info(
                "UpscalePipeline will be dynamically compiled with stable-fast "
                "for %s",
                model_id,
            )
            from app.pipelines.optim.sfast import compile_model

            self.ldm = compile_model(self.ldm)

            # Warm-up the pipeline.
            # TODO: Not yet supported for UpscalePipeline.
            if os.getenv("SFAST_WARMUP", "true").lower() == "true":
                logger.warning(
                    "The 'SFAST_WARMUP' flag is not yet supported for the "
                    "UpscalePipeline and will be ignored. As a result the first "
                    "call may be slow if 'SFAST' is enabled."
                )

        if deepcache_enabled and not (
            is_lightning_model(model_id) or is_turbo_model(model_id)
        ):
            logger.info(
                "UpscalePipeline will be optimized with DeepCache for %s",
                model_id,
            )
            from app.pipelines.optim.deepcache import enable_deepcache

            self.ldm = enable_deepcache(self.ldm)
        elif deepcache_enabled:
            logger.warning(
                "DeepCache is not supported for Lightning or Turbo models. "
                "TextToImagePipeline will NOT be optimized with DeepCache for %s",
                model_id,
            )

        safety_checker_device = os.getenv("SAFETY_CHECKER_DEVICE", "cuda").lower()
        self._safety_checker = SafetyChecker(device=safety_checker_device)

    def __call__(
        self, prompt: str, image: PIL.Image, **kwargs
    ) -> Tuple[List[PIL.Image], List[Optional[bool]]]:
        seed = kwargs.pop("seed", None)
        num_inference_steps = kwargs.get("num_inference_steps", None)
        safety_check = kwargs.pop("safety_check", True)

        if seed is not None:
            if isinstance(seed, int):
                kwargs["generator"] = torch.Generator(get_torch_device()).manual_seed(
                    seed
                )
            elif isinstance(seed, list):
                kwargs["generator"] = [
                    torch.Generator(get_torch_device()).manual_seed(s) for s in seed
                ]

        if num_inference_steps is None or num_inference_steps < 1:
            del kwargs["num_inference_steps"]

        output = self.ldm(prompt, image=image, **kwargs)

        if safety_check:
            _, has_nsfw_concept = self._safety_checker.check_nsfw_images(output.images)
        else:
            has_nsfw_concept = [None] * len(output.images)

        self.print_gpu_memory_usage()
        torch.cuda.empty_cache()
        self.print_gpu_memory_usage()
        return output.images, has_nsfw_concept

    def __str__(self) -> str:
        return f"UpscalePipeline model_id={self.model_id}"
