import torch

# -------------------------------------------------------------------------
# GLOBAL LATTICE MATRICES
# -------------------------------------------------------------------------
# The Generator Matrix (G) defines the spatial structure of the E8 Lattice.
# Multiplying an integer vector by G maps it to a valid E8 lattice point.
# If you want to experiment with a differently scaled E8 lattice 
# (e.g., the 2*E8 basis used in their CUDA kernels), you change this matrix. 
# G_inv will automatically update itself.
G = torch.tensor([
    [2, -1,  0,  0,  0,  0,  0, 0.5],
    [0,  1, -1,  0,  0,  0,  0, 0.5],
    [0,  0,  1, -1,  0,  0,  0, 0.5],
    [0,  0,  0,  1, -1,  0,  0, 0.5],
    [0,  0,  0,  0,  1, -1,  0, 0.5],
    [0,  0,  0,  0,  0,  1, -1, 0.5],
    [0,  0,  0,  0,  0,  0,  1, 0.5],
    [0,  0,  0,  0,  0,  0,  0, 0.5]
], dtype=torch.float32)

G_inv = torch.inverse(G)

# -------------------------------------------------------------------------
# ALGORITHMS
# -------------------------------------------------------------------------
def encode_e8(x: torch.Tensor) -> torch.Tensor:
    """
    Maps a batch of 8D continuous vectors to their nearest E8 lattice points.
    (taken from author's Repo)
    This is 'branchless' optimization. By avoiding Python if/else 
    statements, this function can process millions of weights on a CPU/GPU instantly 
    without stalling the execution pipeline.
    """
    N = x.shape[0]
    
    # 1. Base D8 lattice rounding
    # d is the floor (e.g., 1.7 -> 1.0). g checks if we are past the halfway point.
    d = torch.floor(x)
    g = x > (d + 0.5)
    
    # opt is Candidate A (Standard integer rounding)
    # opt2 is Candidate B (Half-integer grid rounding)
    opt = d + g.float()
    opt2 = d + 0.5
    
    # 2. Fast Parity Check (The Non-Trivial Optimization)
    # E8 requires the sum of coordinates to be an even integer. 
    # '& 1' is a bitwise AND operator. It instantly checks if the sum is odd (1) or even (0).
    bad = torch.sum(opt, dim=1).to(torch.int32) & 1
    bad2 = torch.sum(opt2, dim=1).to(torch.int32) & 1
    
    # 3. Calculate distance to the half-integer boundary to find the cheapest coordinate to flip
    dist_to_half = (opt2 - x) * (1 - 2 * g.float())

    # 4. Parity Correction Masks
    # We create a mask (flip1, flip2) that targets the exact coordinate with the minimum penalty.
    flip1 = torch.zeros_like(x)
    flip1[torch.arange(N), torch.argmin(dist_to_half, dim=1)] = 1.0
    flip1 = flip1 * bad[:, None].float() # ONLY apply the flip if parity was 'bad' (odd)

    flip2 = torch.zeros_like(x)
    flip2[torch.arange(N), torch.argmax(dist_to_half, dim=1)] = 1.0
    flip2 = flip2 * bad2[:, None].float()

    # 5. Apply the flips mathematically (no if-statements)
    opt = flip1 * (2 * d + 1 - opt) + (1 - flip1) * opt
    opt2 = flip2 * (opt2 - 1 + 2 * g.float()) + (1 - flip2) * opt2

    # 6. Final Selection: Choose Candidate A or B based on lowest Mean Squared Error (MSE)
    mse = ((opt - x) ** 2).sum(dim=1)
    mse2 = ((opt2 - x) ** 2).sum(dim=1)
    
    # Return the winner for each block
    return torch.where((mse < mse2)[:, None], opt, opt2)

def generate_dither(n_blocks: int, device: torch.device, seed: int = 42) -> torch.Tensor:
    """
    Generates 'Subtractive Dither' (Uniform random noise within the Voronoi cell).
    This is a crucial safety hack. Adding random noise before quantization breaks 
    up correlated rounding errors. Without this, the QA-LDLQ error-feedback loop will 
    snowball, causing catastrophic model collapse ( 1.1 Billion perplexity).
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # The dither bounds are implicitly set by the * 2.0 multiplier. 
    # If the model behaves too noisily during experiments, you can reduce this 
    # (e.g., * 1.5) to inject less noise, though 2.0 is the paper's default.
    U = torch.rand((n_blocks, 8), generator=generator, device=device, dtype=torch.float32) * 2.0
    
    # Subtracting the E8 projection perfectly bounds the noise within the lattice cell limits
    return U - encode_e8(U)