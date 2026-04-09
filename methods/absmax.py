import torch
from core.base import BaseQuantizer

class AbsMaxQuantizer(BaseQuantizer):
    """
    Symmetric quantization based on the absolute maximum value.
    Maps values symmetrically around zero.
    """
    def __init__(self, bits: int = 8):
        super().__init__(bits=bits, block_size=1)

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Avoid division by zero
        max_val = torch.max(torch.abs(tensor))
        max_val = torch.clamp(max_val, min=1e-8)
        
        # Calculate scale
        scale = self.qmax / max_val
        
        # Quantize and clamp to ensure values stay within the int8 range [-128, 127]
        q_tensor = torch.clamp((tensor * scale).round(), self.qmin, self.qmax)
        
        # Return quantized tensor (cast to int8 to save memory) and metadata
        return q_tensor.to(torch.int8), {"scale": scale}

    def dequantize(self, q_tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        # Reconstruct the simulated FP32 tensor
        return q_tensor.float() / metadata["scale"]