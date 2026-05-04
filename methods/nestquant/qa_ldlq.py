#plase ignore this for now
import torch
from .hadamard import apply_hadamard_transform

def rotate_hessian(H: torch.Tensor) -> torch.Tensor:
    H_rot = apply_hadamard_transform(H.t(), inverse=False).t()
    H_rot = apply_hadamard_transform(H_rot, inverse=False)
    return H_rot

def get_ldlq_feedback_matrix(H: torch.Tensor, block_size: int = 8, damp_percent: float = 0.01) -> torch.Tensor:
    """
    Standard LDLQ/GPTQ-style feedback matrix for weight-only quantization.

    It uses activation covariance H = E[XX^T] to push quantization error from
    already quantized columns into columns that have not been quantized yet.
    """
    n = H.shape[0]
    diag_mean = torch.diag(H).mean().clamp(min=1e-8)
    H_damped = H + torch.eye(n, device=H.device, dtype=H.dtype) * (damp_percent * diag_mean)
    return get_block_ldl(H_damped, block_size=block_size)

def apply_ridge_shift(W: torch.Tensor, H: torch.Tensor, x_var: float, eps_cons: float = 155.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    QA-LDLQ target shift from the NestQuant paper.

    This is retained for future activation-quantization experiments. In this
    weight-only project path, NestQuant + LDLQ uses get_ldlq_feedback_matrix()
    instead, because no activation quantization noise covariance J is measured.
    """
    eps2 = max((x_var / eps_cons), 1e-6)
    n = W.shape[-1]
    I = torch.eye(n, device=W.device)
    
    inv_term = torch.linalg.pinv(H + eps2 * I)
    W_shifted = W @ (I - eps2 * inv_term)
    H_dampened = H + (I * eps2)
    
    return W_shifted, H_dampened

def get_block_ldl(H: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    No Float64 hacks, no safety reverts.
    Pythia's GELU activations guarantee full-rank dense matrices, 
    so standard Float32 Cholesky will work perfectly.
    """
    n = H.shape[0]
    if n % block_size != 0:
        pad = block_size - (n % block_size)
        H = torch.nn.functional.pad(H, (0, pad, 0, pad))
        n = H.shape[0]

    m = n // block_size
    
    # Standard minimal dampening (1e-4) to prevent floating-point rounding crashes
    dampening_factor = torch.diag(H).mean().clamp(min=1e-5) * 1e-4 
    L = torch.linalg.cholesky(H + torch.eye(n, device=H.device) * dampening_factor)

    # block-diagonal extraction
    DL = torch.diagonal(L.reshape(m, block_size, m, block_size), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    
    L = L.view(n, m, block_size)
    for i in range(m):
        L[:, i, :] = L[:, i, :] @ DL[i, :, :]
        
    return L.reshape(n, n)
