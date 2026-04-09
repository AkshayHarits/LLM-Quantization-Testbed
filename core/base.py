import torch
from abc import ABC, abstractmethod

class BaseQuantizer(ABC):
    """
    The master blueprint for all quantization algorithms.
    Any new method (AbsMax, OQMM, NestQuant) must inherit from this class.
    """
    def __init__(self, bits: int = 8, block_size: int = 1):
        self.bits = bits
        self.block_size = block_size # Crucial later for vector/lattice quantization
        self.qmax = (2**(self.bits - 1)) - 1
        self.qmin = -(2**(self.bits - 1))

    @abstractmethod
    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Takes an FP32/FP16 tensor and returns:
        1. The quantized tensor (e.g., INT8 or indices for a lattice)
        2. A dictionary of metadata needed to reverse it (scales, zero-points, etc.)
        """
        pass

    @abstractmethod
    def dequantize(self, q_tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Takes the quantized tensor and metadata, returning a simulated (fake) FP32 tensor 
        that is ready for matrix multiplication.
        """
        pass