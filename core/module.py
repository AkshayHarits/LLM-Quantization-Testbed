import torch
import torch.nn as nn
import torch.nn.functional as F
from core.base import BaseQuantizer

class QuantizedLinear(nn.Module):
    """
    Optional drop-in wrapper for experiments that want to keep a separate
    quantized payload. The default orchestrator now uses faster in-place fake
    quantization in core.surgery, so this class is mostly a compatibility path.
    """
    def __init__(self, original_layer: nn.Linear, quantizer: BaseQuantizer):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.quantizer = quantizer
        
        # 1. Quantize the weights immediately upon initialization
        q_weight, metadata = self.quantizer.quantize(original_layer.weight.data)
        
        # 2. Store the quantized weights as a buffer
        self.register_buffer('q_weight', q_weight)
        
        # 3. Store metadata dynamically as buffers
        self.metadata_keys = []
        self.metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, list, dict, tuple)):
                self.metadata[key] = value
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            self.register_buffer(f'meta_{key}', value)
            self.metadata_keys.append(key)

        # 4. Handle the bias
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)

    @property
    def weight(self) -> torch.Tensor:
        """
        A dynamic property to satisfy Hugging Face architecture checks.
        Whenever the model explicitly asks for layer.weight, this reconstructs 
        the simulated FP32 tensor on the fly.
        """
        metadata = dict(self.metadata)
        metadata.update({key: getattr(self, f'meta_{key}') for key in self.metadata_keys})
        return self.quantizer.dequantize(self.q_weight, metadata)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perform the standard matrix multiplication using the dynamic property
        return F.linear(x, self.weight, self.bias)
