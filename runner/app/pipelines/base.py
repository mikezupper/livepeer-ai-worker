from abc import ABC, abstractmethod
from typing import Any
import logging
import torch

logger = logging.getLogger(__name__)

class Pipeline(ABC):
    @abstractmethod
    def __init__(self, model_id: str, model_dir: str):
        raise NotImplementedError("Pipeline should implement an __init__ method")

    @abstractmethod
    def __call__(self, inputs: Any) -> Any:
        raise NotImplementedError("Pipeline should implement a __call__ method")

    def print_gpu_memory_usage(self):
        allocated = torch.cuda.memory_allocated() / (1024**3)
        cached = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"Allocated memory: {allocated:.2f} GB")
        logger.info(f"Cached memory: {cached:.2f} GB")