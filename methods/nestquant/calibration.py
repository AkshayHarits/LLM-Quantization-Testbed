import torch
import torch.nn as nn
from datasets import load_dataset

def get_calibration_data(tokenizer, num_samples: int = 50, seq_len: int = 512, device='cpu') -> torch.Tensor:
    """
    WHAT: Fetches Wikitext-2 to 'wake up' the neurons for calibration.
    """
    print("Downloading/Loading Wikitext-2 calibration data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text_blocks = [s['text'] for s in dataset if len(s['text'].strip()) > 0]
    combined_text = " ".join(text_blocks[:num_samples * 2])
    
    encodings = tokenizer(
        combined_text, 
        return_tensors="pt", 
        max_length=seq_len, 
        truncation=True
    )
    return encodings.input_ids.to(device)

def capture_statistics(model: nn.Module, input_ids: torch.Tensor) -> tuple[dict, dict]:
    """
    WHAT: Captures both the Hessian (H) and the Activation Variance (x_var).
    WHY: H is used by QA-LDLQ to shift the weights. x_var is used to calculate 
    the 'eps' ridge-dampening factor to prevent the math from exploding.
    """
    hessians = {}
    x_vars = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            x = input[0].detach().float()
            
            # Flatten to [total_tokens, in_features]
            x = x.reshape(-1, x.shape[-1])
            n_tokens = x.shape[0]
            n_features = x.shape[1]
            
            # 1. Capture uncentered covariance (Hessian)
            h_batch = torch.matmul(x.t(), x)
            
            # 2. Capture empirical variance of activations (Semyon's err.py logic)
            # var = sum((x - mean)^2)
            mean_x = x.mean(dim=0)
            var_batch = ((x - mean_x) ** 2).sum().item()
            
            # Accumulate
            if name not in hessians:
                hessians[name] = h_batch / n_tokens
                x_vars[name] = var_batch / (n_tokens * n_features)
            else:
                # Moving average
                hessians[name] = (hessians[name] + (h_batch / n_tokens)) / 2.0
                x_vars[name] = (x_vars[name] + (var_batch / (n_tokens * n_features))) / 2.0
                
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(get_hook(name)))

    print(f"Running calibration pass with {input_ids.shape[1]} tokens...")
    model.eval()
    with torch.no_grad():
        model(input_ids=input_ids, labels=input_ids)

    for h in hooks:
        h.remove()

    print(f"Successfully captured statistics for {len(hessians)} layers.")
    return hessians, x_vars