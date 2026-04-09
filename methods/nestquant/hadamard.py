import torch
import numpy as np
from scipy.linalg import hadamard

def get_hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    """
    WHAT: Constructs an orthogonal rotation matrix tailored to dimension n.
    WHY: Rotating weights by this matrix spreads 'spiky' outliers evenly 
    across all 8 dimensions, preventing single massive weights from overloading 
    the 4-bit lattice bounds.
    """
    # Fast path: Check if dimension 'n' is a clean power of 2 (e.g., 512, 4096)
    if (n & (n - 1) == 0) and n > 0:
        H = torch.from_numpy(hadamard(n)).to(device).float()
    else:
        # Slow path: For weird dimensions (like vocab sizes, e.g. 32128), 
        # we decompose n = (2^k) * m and use Kronecker products to build the matrix.
        k = 0
        m = n
        while m % 2 == 0:
            m //= 2
            k += 1
            
        H_power_of_2 = torch.from_numpy(hadamard(2**k)).float()
        
        try:
            H_remainder = torch.from_numpy(hadamard(m)).float()
            # .contiguous() prevents PyTorch memory fragmentation crashes on CPU
            H = torch.kron(H_remainder.contiguous(), H_power_of_2.contiguous()).to(device)
        except ValueError:
            # Fallback for prime number remainders: Use a random orthogonal matrix
            H_random, _ = torch.linalg.qr(torch.randn(m, m))
            H = torch.kron(H_random.contiguous(), H_power_of_2.contiguous()).to(device)

    # IMPORTANT: Divide by sqrt(n) to ensure the matrix is perfectly Orthogonal.
    # Without this, the rotation would massively inflate the magnitude of the weights.
    return H / np.sqrt(n)

def apply_hadamard_transform(tensor: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    WHAT: Applies the actual spatial rotation to the weights or activations.
    
    TWEAK HERE: The 'inverse' flag is crucial. During quantization, we rotate 
    forward (inverse=False). During generation/inference, if we quantize activations, 
    we must rotate them back (inverse=True) before passing them to the next layer.
    """
    n = tensor.shape[-1]
    H = get_hadamard_matrix(n, tensor.device)
    
    if inverse:
        # Multiply by the transpose to reverse the rotation
        return torch.matmul(tensor, H)
    else:
        # Standard forward rotation
        return torch.matmul(tensor, H.t())