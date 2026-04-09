import torch
from core.base import BaseQuantizer

class ZeroPointQuantizer(BaseQuantizer):
    """
    Asymmetric quantization.
    Useful for skewed distributions (like ReLU activations) where values 
    do not center naturally around zero.
    """
    def __init__(self, bits: int = 8):
        super().__init__(bits=bits, block_size=1)
        # For asymmetric, the total number of levels is 255 for 8-bit
        self.levels = (2**self.bits) - 1 

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        t_min = torch.min(tensor)
        t_max = torch.max(tensor)
        
        # Calculate scale based on the full range of the tensor
        ip_range = t_max - t_min
        ip_range = torch.clamp(ip_range, min=1e-8)
        scale = self.levels / ip_range
        
        # Calculate zero-point and shift the distribution
        zeropoint = (-scale * t_min + self.qmin).round()
        
        # Quantize and clip
        q_tensor = torch.clamp((tensor * scale + zeropoint).round(), self.qmin, self.qmax)
        
        return q_tensor.to(torch.int8), {"scale": scale, "zeropoint": zeropoint}

    def dequantize(self, q_tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        # Reconstruct the simulated FP32 tensor
        scale = metadata["scale"]
        zeropoint = metadata["zeropoint"]
        return (q_tensor.float() - zeropoint) / scale