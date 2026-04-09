import torch
import matplotlib.pyplot as plt
import seaborn as sns

def plot_weight_error(original_weights: torch.Tensor, quantized_weights: torch.Tensor, layer_name: str):
    """
    WHAT: Generates a histogram of the relative error introduced by quantization.
    WHY: Visual proof for your thesis of how the E8 lattice preserves weight fidelity.
    """
    # Flatten the tensors
    real = original_weights.flatten().float()
    approx = quantized_weights.flatten().float()
    
    # Calculate relative error: |approx - real| / |real|
    # Add 1e-5 to denominator to prevent division by zero
    relative_error = (approx - real).abs() / (real.abs() + 1e-5)
    
    # Filter out near-zero errors for a cleaner log plot
    relative_error = relative_error[relative_error > 1e-5]
    log_relative_error = torch.log10(relative_error + 1e-10)
    
    # Plotting
    plt.figure(figsize=(8, 5))
    sns.histplot(log_relative_error.detach().cpu().numpy(), bins=100, color='blue', kde=True)
    
    plt.xlabel("Log10 of Relative Error")
    plt.ylabel("Frequency (Number of Weights)")
    plt.title(f"Quantization Error Distribution: {layer_name}")
    
    # Save the figure so you can put it in your report
    plt.savefig(f"error_plot_{layer_name.replace('.', '_')}.png")
    plt.close()
    print(f"     Saved error plot for {layer_name}")

def plot_weight_distribution(original_weights: torch.Tensor, quantized_weights: torch.Tensor, layer_name: str):
    """
    WHAT: Plots the actual distribution (shape) of the weights before and after quantization.
    WHY: Standard quantization often clips (destroys) the long tails (outliers) of the 
    weight distribution. This plot visually proves that the E8 lattice preserves them.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot the original FP32 weights (Smooth, continuous)
    sns.kdeplot(original_weights.flatten().cpu().numpy(), 
                color='blue', label='Original FP32', fill=True, alpha=0.3)
    
    # Plot the Quantized weights (Will look slightly 'stepped' due to the lattice)
    sns.kdeplot(quantized_weights.flatten().cpu().numpy(), 
                color='red', label='Quantized 4-bit', linestyle="--")
    
    plt.title(f"Weight Distribution Preservation: {layer_name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.legend()
    
    # Zoom in on the main body of the weights to see the detail
    plt.xlim(-0.5, 0.5) 
    
    filename = f"dist_plot_{layer_name.replace('.', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"     Saved distribution plot to {filename}")