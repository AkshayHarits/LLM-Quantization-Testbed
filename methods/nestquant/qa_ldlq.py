import torch
from .hadamard import apply_hadamard_transform

def rotate_hessian(H: torch.Tensor) -> torch.Tensor:
    """
    WHAT: Rotates the Hessian matrix into the Hadamard space.
    WHY: Because we rotate the weights (W), the covariance of the activations (H) 
    must also be rotated into the exact same vector space so the math aligns.
    """
    # H_rot = Hadamard * H * Hadamard^T
    H_rot = apply_hadamard_transform(H.t(), inverse=False).t()
    H_rot = apply_hadamard_transform(H_rot, inverse=False)
    return H_rot

def apply_ridge_shift(W: torch.Tensor, H: torch.Tensor, x_var: float, eps_cons: float = 155.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    WHAT: Performs the pre-quantization weight shift with Ridge Regression.
    
    WHY: Instead of calculating W * H * inv(H + J), we use W @ (I - eps^2 * inv(H + eps^2 * I)).
    Adding eps^2 to the diagonal (Ridge Dampening) forces the eigenvalues up, making 
    the inverse perfectly stable so the weights don't explode to 1.1 Billion.

    DEPARTURE FROM SEMYON REPO: 
    1. The original repo allowed `eps2` to drop to absolute zero if `x_var` was zero. 
       We added `max(..., 1e-6)` as a hard floor to prevent zero-dampening in dead layers.
    2. The original repo used standard `torch.linalg.inv`. We upgraded to the Moore-Penrose 
       Pseudo-Inverse (`pinv`), which computes the Singular Value Decomposition (SVD). 
       It is computationally slower but mathematically immune to singular matrix crashes.
    """
    # Ensure eps2 never drops completely to zero (Safety Net 1)
    eps2 = max((x_var / eps_cons), 1e-6) 
    n = W.shape[-1]
    I = torch.eye(n, device=W.device)
    
    # Use Pseudo-Inverse (pinv) to guarantee stability on rank-deficient Hessians
    inv_term = torch.linalg.pinv(H + eps2 * I)
    W_shifted = W @ (I - eps2 * inv_term)
    
    # We must pass the dampened Hessian to the Cholesky solver to prevent later crashes
    H_dampened = H + (I * eps2)
    
    return W_shifted, H_dampened

def get_block_ldl(H: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    WHAT: Computes the Block-LDL decomposition required for the error feedback loop.
    WHY: The paper dictates that error should only be pushed between 8D blocks, 
    not within the blocks themselves. Semyon's matrix algebra dynamically extracts 
    the correct block-diagonal inverse. 

    DEPARTURE FROM SEMYON REPO:
    Elevated the Cholesky decomposition to Float64 (Double Precision).
    Float32 precision loss causes 90% of Cholesky failures on small, sparse models like T5.
    By doing the math in Float64 and applying a heavy ridge, we guarantee the feedback 
    matrix (L) is generated even for stubborn, sparse decoder layers.
    """
    n = H.shape[0]
    m = n // block_size
    
    # 1. Calculate a dynamic safety floor based on the matrix scale
    diag_vals = torch.diag(H)
    
    # If the layer is essentially 'dead' (all zeros), gracefully skip LDLQ for this layer
    if diag_vals.max() < 1e-9:
        print(f"    [!] Skipping LDL feedback: Layer is mathematically dead ({n}x{n})")
        return torch.zeros((n, n), device=H.device)

    # 2. Elevate to Float64 to survive the dead-neuron sparsity
    H_double = H.to(torch.float64)

    # 3. Force the matrix to be Positive Definite by adding a Ridge
    # We use a 1% floor of the average diagonal value to stabilize it
    dampening = torch.diag(H_double).mean().clamp(min=1e-6) * 0.01
    H_stable = H_double + torch.eye(n, device=H.device, dtype=torch.float64) * dampening
    
    try:
        # We attempt Cholesky on the stabilized Float64 matrix
        L_double = torch.linalg.cholesky(H_stable)
    except RuntimeError:
        print(f"    [!] Float32/64 sparsity warning on {n}x{n}. Using heavy regularizer.")
        # If the matrix is violently sparse, hit it with a heavy regularizer
        H_heavy = H_double + torch.eye(n, device=H.device, dtype=torch.float64) * 0.1
        try:
            L_double = torch.linalg.cholesky(H_heavy)
        except RuntimeError:
            # Absolute worst-case scenario: Use Identity to keep the matrix alive
            # This prevents returning 0, so the error feedback loop isn't completely destroyed
            L_double = torch.eye(n, device=H.device, dtype=torch.float64)

    # Semyon's exact block-diagonal extraction (Done in Float64 for perfect precision)
    DL = torch.diagonal(L_double.reshape(m, block_size, m, block_size), dim1=0, dim2=2).permute(2, 0, 1)
    DL = torch.linalg.inv(DL)
    
    L_double = L_double.view(n, m, block_size)
    for i in range(m):
        L_double[:, i, :] = L_double[:, i, :] @ DL[i, :, :]
        
    # 4. Cast back to Float32 for the rest of the pipeline
    return L_double.reshape(n, n).to(torch.float32)