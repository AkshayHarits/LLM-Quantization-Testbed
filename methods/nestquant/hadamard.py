import torch
import numpy as np
from scipy.linalg import hadamard

_HADAMARD_CACHE: dict[tuple[int, str, int | None], torch.Tensor] = {}


def get_hadamard_matrix(n: int, device: torch.device) -> torch.Tensor:
    """
    Constructs an orthogonal rotation matrix tailored to dimension n.
    Rotating weights by this matrix spreads 'spiky' outliers evenly 
    across all 8 dimensions, preventing single massive weights from overloading 
    the 4-bit lattice bounds.
    """
    cache_key = (n, device.type, device.index)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

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
            # Fallback for non-Hadamard remainders such as GPT-2's 768 = 3 * 256.
            # Make it deterministic and scale it like an unnormalized Hadamard block
            # so the final division by sqrt(n) still produces an orthogonal matrix.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(1729 + n)
            H_random, _ = torch.linalg.qr(torch.randn(m, m, generator=generator))
            H_remainder = H_random * np.sqrt(m)
            H = torch.kron(H_remainder.contiguous(), H_power_of_2.contiguous()).to(device)

    # Divide by sqrt(n) to ensure the matrix is perfectly Orthogonal.
    # Without this, the rotation would massively inflate the magnitude of the weights.
    H = H / np.sqrt(n)
    _HADAMARD_CACHE[cache_key] = H
    return H

def apply_hadamard_transform(tensor: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies the actual spatial rotation to the weights or activations.
    
    The 'inverse' flag is crucial. During quantization, we rotate 
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
