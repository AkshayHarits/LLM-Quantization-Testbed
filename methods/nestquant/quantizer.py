import torch
from core.base import BaseQuantizer
from .e8 import encode_e8, generate_dither
from .hadamard import apply_hadamard_transform
from .qa_ldlq import apply_ridge_shift, get_block_ldl, rotate_hessian

class NestQuantizer(BaseQuantizer):
    """
    WHAT: The master orchestrator for the 4-bit NestQuant pipeline.
    WHY: Inherits from BaseQuantizer but completely overrides standard 1D rounding 
    with an 8-dimensional E8 Lattice and an Optimal Brain Quantization (QA-LDLQ) 
    error-feedback loop.
    """
    def __init__(self, q=14, betas=None, eps_cons=155.0, use_dither=False, use_ldlq=False):
        super().__init__(bits=4, block_size=8)
        self.q = q
        self.use_dither = use_dither
        self.eps_cons = eps_cons
        
        # DEPARTURE FROM SEMYON REPO:
        # Added an ablation toggle (use_ldlq) to allow isolating the E8 lattice 
        # mapping from the complex Hessian error-feedback math during experiments.
        self.use_ldlq = use_ldlq 
        
        # WHAT: The discrete scaling steps for the E8 lattice.
        # WHY: The lattice needs to expand or contract to fit the specific weight distribution.
        # TWEAK HERE: These are Semyon's hardcoded "magic betas" found via 
        # massive server grid searches. If you want to experiment, you change these.
        # [4.0, 5.5, 7.0, 18.0] -> Weights. 
        # [3.47, 4.74, 6.90, 18.11] -> Activations.
        # If T5-Small is failing, we may need to scale these down.
        self.betas = betas if betas is not None else [4.0, 5.5, 7.0, 18.0]
        
        # Stateful variables injected dynamically by our Calibration Hook
        self.hessian = None 
        self.x_var = None

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        WHAT: The master pipeline for Simulated NestQuant processing.
        Transforms FP32 weights into simulated 4-bit E8 lattice points.
        """
        device = tensor.device
        W = tensor.clone().float()
        m, n = W.shape
        
        # ---------------------------------------------------------
        # 1. GAUSSIANIZATION
        # ---------------------------------------------------------
        # WHAT: Rotates weights to distribute massive outliers evenly.
        # WHY: Spreads outliers across all 8 dimensions so they don't blow past lattice bounds.
        W_shifted = apply_hadamard_transform(W, inverse=False)
        
        # ---------------------------------------------------------
        # 2. QA-LDLQ PREP (The Shift & The Feedback Matrix)
        # ---------------------------------------------------------
        # We only run this if the ablation toggle is ON and we have valid stats
        if self.use_ldlq and self.hessian is not None and self.x_var is not None:
            H_rot = rotate_hessian(self.hessian)
            
            # THE FIX: Save a pristine backup of the weights before shifting them
            W_pristine = W_shifted.clone()
            
            # Apply the intentional distortion (The "Poison")
            W_shifted_temp, H_dampened = apply_ridge_shift(W_shifted, H_rot, self.x_var, self.eps_cons)
            
            # Attempt to generate the error feedback matrix (The "Antidote")
            L = get_block_ldl(H_dampened, block_size=8)
            
            if L is None:
                # The Antidote failed to build. We MUST undo the Poison!
                W_shifted = W_pristine
                L = torch.zeros((n, n), device=device)
            else:
                # Math succeeded! Commit the shifted weights.
                W_shifted = W_shifted_temp
        else:
            # DEPARTURE FROM SEMYON REPO (Ablation Mode):
            # Blind Lattice fallback. Setting L to a zero matrix cleanly disables 
            # the error-feedback loop, allowing pure E8 baseline testing.
            L = torch.zeros((n, n), device=device)
            
        # ---------------------------------------------------------
        # 3. GREEDY BETA LOOP WITH DITHER
        # ---------------------------------------------------------
        decoded_W = torch.zeros_like(W_shifted)
        
        # We loop backwards through the columns in blocks of 8
        # WHY: To calculate rounding error of the current block and push it forward 
        # into the *unquantized* blocks on the left using matrix L.
        for c in range(n - 8, -8, -8):
            # Calculate the accumulated error feedback from right-side blocks
            # (This naturally evaluates to 0 if use_ldlq is False or L is zeroed out)
            error_feedback = (W_shifted[:, c+8:n] - decoded_W[:, c+8:n]) @ L[c+8:n, c:c+8]
            current_block = W_shifted[:, c:c+8] + error_feedback
            
            # Arrays to store the "winner" of the 4 betas for each row
            best_recon = torch.zeros_like(current_block)
            min_error = torch.full((m,), float('inf'), device=device)
            
            # Brute-force search: Test all 4 betas
            for beta_val in self.betas:
                beta = beta_val / self.q
                
                # Apply Subtractive Dither
                z = generate_dither(m, device) if self.use_dither else 0.0
                
                # Simulate Quantization
                scaled_block = (current_block / beta) + z
                lattice_points = encode_e8(scaled_block)
                recon_scaled = (lattice_points - z) * beta
                
                # Calculate MSE
                mse = ((recon_scaled - current_block) ** 2).sum(dim=1)
                
                # Save the reconstruction if it beat the previous betas
                replace_mask = mse < min_error
                best_recon[replace_mask] = recon_scaled[replace_mask]
                min_error[replace_mask] = mse[replace_mask]
                
            # Store the optimal 8D block in our final matrix
            decoded_W[:, c:c+8] = best_recon
            
        # ---------------------------------------------------------
        # 4. INVERSE GAUSSIANIZATION
        # ---------------------------------------------------------
        # Rotate the simulated quantized weights back into standard space
        W_simulated = apply_hadamard_transform(decoded_W, inverse=True)
        
        # Return the simulated FP32 weights (which have the 4-bit error baked in)
        metadata = {"simulated": True}
        return W_simulated, metadata

    def dequantize(self, q_tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Since we return the reconstructed floats immediately in `quantize` 
        to simulate perplexity rapidly, dequantize just passes them through.
        """
        return q_tensor