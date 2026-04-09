import torch.nn as nn
from core.base import BaseQuantizer
from core.module import QuantizedLinear
from eval.plot import plot_weight_error , plot_weight_distribution

def swap_linear_layers(
    module: nn.Module, 
    quantizer: BaseQuantizer, 
    skip_layers: list = None, 
    hessian_dict: dict = None, 
    x_var_dict: dict = None, 
    name_prefix: str = ""
) -> nn.Module:
    """
    WHAT: Recursively searches a PyTorch model and quantizes nn.Linear layers.
    WHY: Handles both dynamic quantization (via wrapper modules) and static 
    simulated quantization (direct weight injection) for maximum CPU performance.
    """
    if skip_layers is None:
        skip_layers = ["lm_head"] 

    for name, child in module.named_children():
        full_name = f"{name_prefix}.{name}" if name_prefix else name

        if isinstance(child, nn.Linear) and not any(skip in name for skip in skip_layers):
            
            # --- NESTQUANT STATS INJECTION ---
            if hasattr(quantizer, 'hessian'):
                quantizer.hessian = hessian_dict.get(full_name, None) if hessian_dict else None
                quantizer.x_var = x_var_dict.get(full_name, None) if x_var_dict else None

            print(f"Quantizing {full_name}...")
            
            # Perform the quantization
            q_weight, metadata = quantizer.quantize(child.weight.data)
            
            # ---------------------------------------------------------
            # VISUALIZATION HOOK
            # ---------------------------------------------------------
            layers_to_plot = [
                "encoder.block.0.layer.0.SelfAttention.q", 
                "encoder.block.5.layer.1.DenseReluDense.wi_1"
            ]
            
            if full_name in layers_to_plot:
                # We must dequantize the integers back to FP32 to compare shapes
                simulated_q_weight = quantizer.dequantize(q_weight, metadata)

                print(f"    -> Generating Error Plot for {full_name}...")
                # Plot using simulated_q_weight instead of q_weight
                plot_weight_error(child.weight.data, simulated_q_weight, full_name)

                print(f"    -> Generating Distribution Plot for {full_name}...")
                # Plot using simulated_q_weight instead of q_weight
                plot_weight_distribution(child.weight.data, simulated_q_weight, full_name)
            # ---------------------------------------------------------

            # --- THE SIMULATION BYPASS (Fast CPU Path) ---
            if metadata.get("simulated", False):
                # If it's a simulated float tensor, just permanently overwrite the weights.
                # No custom nn.Module wrapper needed! Inference will be 100x faster.
                child.weight.data = q_weight.to(child.weight.dtype)
            else:
                # --- THE DYNAMIC WRAPPER (Old Backward-Compatible Path) ---
                # Use this for strict AbsMax/ZeroPoint integer tracking
                q_layer = QuantizedLinear(child, quantizer)
                # Manually inject the quantized weights since we already computed them
                q_layer.q_weight = q_weight 
                setattr(module, name, q_layer)
                
        else:
            swap_linear_layers(child, quantizer, skip_layers, hessian_dict, x_var_dict, name_prefix=full_name)
            
    return module