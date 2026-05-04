from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from core.base import BaseQuantizer, flatten_weight_rows, restore_weight_rows
from .e8 import encode_e8, generate_dither
from .hadamard import apply_hadamard_transform
from .qa_ldlq import get_ldlq_feedback_matrix, rotate_hessian


@dataclass
class ScaleSelectionResult:
    candidates: torch.Tensor
    selected: torch.Tensor
    mse: torch.Tensor
    overload: torch.Tensor
    assignments: torch.Tensor
    dp_error: float
    greedy_error: float


class NestQuantizer(BaseQuantizer):
    """

    Implemented:
    - Hadamard rotation / gaussianization.
    - Row L2 normalization to norm sqrt(n).
    - 8D block decomposition.
    - Candidate beta pool from calibration blocks.
    - First-beta assignment and dynamic programming over k selected scales.
    - Optional weight-only LDLQ-style Hessian error feedback.(not working right now)

    """

    method_name = "nestquant"

    def __init__(
        self,
        bits: int = 4,
        k: int = 4,
        m: int = 32,
        grid_type: str = "log",
        block_size: int = 64,
        q: int | None = None,
        beta_min: float | None = None,
        beta_max: float | None = None,
        use_dither: bool = False,
        use_ldlq: bool = False,
        damp_percent: float = 0.01,
    ):
        super().__init__(bits=bits, block_size=block_size, granularity="block")
        self.k = int(k)
        self.m = int(m)
        self.grid_type = grid_type
        self.q = int(q if q is not None else max(2, (2**bits) - 2))
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.use_dither = use_dither
        self.use_ldlq = use_ldlq
        self.damp_percent = damp_percent
        self.e8_dim = 8
        self._dither_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _pad_rows(self, rows: torch.Tensor) -> tuple[torch.Tensor, int]:
        pad = (-rows.shape[1]) % self.e8_dim
        if pad:
            rows = F.pad(rows, (0, pad))
        return rows, pad

    def _row_normalize(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = rows.shape[1]
        norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp(min=1e-12)
        return rows * ((n**0.5) / norms), norms

    def _row_denormalize(self, rows: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
        n = rows.shape[1]
        return rows * (norms / (n**0.5))

    def _blocks_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        return rows.reshape(-1, self.e8_dim)

    def _candidate_pool(self, blocks: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(blocks, dim=1)
        lower = self.beta_min
        upper = self.beta_max
        if lower is None:
            lower = torch.quantile((norms / self.q).clamp(min=1e-8), 0.05).item()
        if upper is None:
            upper = (torch.max(norms).item() / self.q) * 1.10 + 1e-8
        lower = max(float(lower), 1e-8)
        upper = max(float(upper), lower * 1.01)

        if self.grid_type == "linear":
            return torch.linspace(lower, upper, self.m, device=blocks.device)
        if self.grid_type == "log":
            return torch.logspace(math.log10(lower), math.log10(upper), self.m, device=blocks.device)
        raise ValueError(f"Unsupported NestQuant grid_type: {self.grid_type}")

    def _dither(self, n_blocks: int, device: torch.device) -> torch.Tensor:
        if not self.use_dither:
            return torch.zeros((n_blocks, self.e8_dim), device=device)
        key = (n_blocks, device)
        if key not in self._dither_cache:
            self._dither_cache[key] = generate_dither(n_blocks, device)
        return self._dither_cache[key]

    def _quantize_blocks_with_beta(self, blocks: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        z = self._dither(blocks.shape[0], blocks.device)
        encoded = encode_e8((blocks / beta) + z)
        return (encoded - z) * beta

    def _encode_blocks_with_beta(self, blocks: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        z = self._dither(blocks.shape[0], blocks.device)
        return encode_e8((blocks / beta) + z)

    def _is_in_finite_codebook(self, lattice_points: torch.Tensor) -> torch.Tensor:
        """
        Check codeword membership in C = E8 intersect q * V_E8.

        A point y is in q * V_E8 exactly when the nearest point in the scaled
        lattice qE8 is the origin. Since encode_e8(y / q) returns the nearest
        E8 point to y / q, y is inside the shaping Voronoi region iff that
        result is zero.
        """
        nearest_scaled_lattice_point = encode_e8(lattice_points / float(self.q))
        return torch.all(torch.abs(nearest_scaled_lattice_point) < 1e-6, dim=1)

    def _precompute_costs(self, blocks: torch.Tensor, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mse_cols = []
        overload_cols = []
        for beta in candidates:
            lattice_points = self._encode_blocks_with_beta(blocks, beta)
            recon = lattice_points * beta
            mse_cols.append(torch.sum((recon - blocks) ** 2, dim=1))
            overload_cols.append(~self._is_in_finite_codebook(lattice_points))
        return torch.stack(mse_cols, dim=1), torch.stack(overload_cols, dim=1)

    def _first_beta_assignments(self, overload: torch.Tensor, selected_indices: torch.Tensor) -> torch.Tensor:
        selected_overload = overload[:, selected_indices]
        assignments = torch.full((overload.shape[0],), selected_indices[-1].item(), device=overload.device, dtype=torch.long)
        for local_idx, candidate_idx in enumerate(selected_indices):
            fits = ~selected_overload[:, local_idx]
            unset = assignments == selected_indices[-1]
            assignments[fits & unset] = candidate_idx
        return assignments

    def _subset_cost(self, mse: torch.Tensor, overload: torch.Tensor, selected_indices: torch.Tensor) -> float:
        assignments = self._first_beta_assignments(overload, selected_indices)
        rows = torch.arange(mse.shape[0], device=mse.device)
        return torch.sum(mse[rows, assignments]).item()

    def _greedy_scales(self, mse: torch.Tensor, overload: torch.Tensor) -> torch.Tensor:
        selected = [mse.shape[1] - 1]
        while len(selected) < min(self.k, mse.shape[1]):
            best_idx = None
            best_cost = float("inf")
            for idx in range(mse.shape[1]):
                if idx in selected:
                    continue
                trial = torch.tensor(sorted(selected + [idx]), device=mse.device, dtype=torch.long)
                cost = self._subset_cost(mse, overload, trial)
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            selected.append(best_idx)
        return torch.tensor(sorted(selected), device=mse.device, dtype=torch.long)

    def _dp_select_scales(self, candidates: torch.Tensor, mse: torch.Tensor, overload: torch.Tensor) -> ScaleSelectionResult:
        n_blocks, n_candidates = mse.shape
        k = min(self.k, n_candidates)
        inf = torch.tensor(float("inf"), device=mse.device)
        dp = torch.full((n_candidates, k + 1), inf, device=mse.device)
        prev = torch.full((n_candidates, k + 1), -1, device=mse.device, dtype=torch.long)

        for i in range(n_candidates):
            covered = ~overload[:, i]
            dp[i, 1] = torch.sum(mse[covered, i])

        for j in range(2, k + 1):
            for i in range(j - 1, n_candidates):
                best_val = inf
                best_s = -1
                for s in range(j - 2, i):
                    newly_covered = overload[:, s] & (~overload[:, i])
                    val = dp[s, j - 1] + torch.sum(mse[newly_covered, i])
                    if val < best_val:
                        best_val = val
                        best_s = s
                dp[i, j] = best_val
                prev[i, j] = best_s

        valid_final = torch.where(~torch.any(overload, dim=0))[0]
        if valid_final.numel() == 0:
            final_i = n_candidates - 1
        else:
            final_costs = dp[valid_final, k]
            final_i = valid_final[torch.argmin(final_costs)].item()

        selected = []
        cur_i = int(final_i)
        cur_j = k
        while cur_j > 0 and cur_i >= 0:
            selected.append(cur_i)
            cur_i = int(prev[cur_i, cur_j].item()) if cur_j > 1 else -1
            cur_j -= 1
        selected = sorted(set(selected))
        while len(selected) < k:
            selected.insert(0, max(0, selected[0] - 1 if selected else 0))
            selected = sorted(set(selected))
        selected_indices = torch.tensor(selected[-k:], device=mse.device, dtype=torch.long)

        assignments = self._first_beta_assignments(overload, selected_indices)
        greedy_indices = self._greedy_scales(mse, overload)
        dp_error = self._subset_cost(mse, overload, selected_indices)
        greedy_error = self._subset_cost(mse, overload, greedy_indices)
        return ScaleSelectionResult(
            candidates=candidates,
            selected=candidates[selected_indices],
            mse=mse,
            overload=overload,
            assignments=assignments,
            dp_error=dp_error,
            greedy_error=greedy_error,
        )

    def _apply_ldlq_feedback(
        self,
        normalized_rows: torch.Tensor,
        decoded_rows: torch.Tensor,
        selected: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_ldlq or self.hessian is None or self.hessian.shape[-1] != normalized_rows.shape[1]:
            return decoded_rows

        H_rot = rotate_hessian(self.hessian.to(normalized_rows.device).float())
        L = get_ldlq_feedback_matrix(H_rot, block_size=self.e8_dim, damp_percent=self.damp_percent)
        if L.shape[0] > normalized_rows.shape[1]:
            L = L[: normalized_rows.shape[1], : normalized_rows.shape[1]]

        corrected = torch.zeros_like(normalized_rows)
        for c in range(normalized_rows.shape[1] - self.e8_dim, -1, -self.e8_dim):
            feedback = torch.zeros_like(normalized_rows[:, c : c + self.e8_dim])
            if c + self.e8_dim < normalized_rows.shape[1]:
                feedback = (normalized_rows[:, c + self.e8_dim :] - corrected[:, c + self.e8_dim :]) @ L[
                    c + self.e8_dim :, c : c + self.e8_dim
                ]
            block = normalized_rows[:, c : c + self.e8_dim] + feedback
            block_flat = block.reshape(-1, self.e8_dim)
            chosen = torch.empty_like(block_flat)
            assigned = torch.zeros((block_flat.shape[0],), dtype=torch.bool, device=block_flat.device)
            for beta in selected:
                lattice_points = self._encode_blocks_with_beta(block_flat, beta)
                fits = self._is_in_finite_codebook(lattice_points)
                take = fits & (~assigned)
                if torch.any(take):
                    chosen[take] = lattice_points[take] * beta
                    assigned[take] = True
            if not torch.all(assigned):
                beta = selected[-1]
                chosen[~assigned] = self._quantize_blocks_with_beta(block_flat[~assigned], beta)
            corrected[:, c : c + self.e8_dim] = chosen.reshape_as(block)
        return corrected

    def _quantize_impl(self, tensor: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        rows, original_shape = flatten_weight_rows(tensor)
        rows = rows.clone()
        padded_rows, pad = self._pad_rows(rows)

        rotated = apply_hadamard_transform(padded_rows, inverse=False)
        normalized, row_norms = self._row_normalize(rotated)
        blocks = self._blocks_from_rows(normalized)

        candidates = self._candidate_pool(blocks)
        mse, overload = self._precompute_costs(blocks, candidates)
        grow_steps = 0
        while torch.any(overload[:, -1]) and grow_steps < 8:
            candidates = candidates * 2.0
            mse, overload = self._precompute_costs(blocks, candidates)
            grow_steps += 1
        selection = self._dp_select_scales(candidates, mse, overload)

        reconstructed_blocks = torch.empty_like(blocks)
        for beta_idx, beta in enumerate(candidates):
            mask = selection.assignments == beta_idx
            if torch.any(mask):
                reconstructed_blocks[mask] = self._quantize_blocks_with_beta(blocks[mask], beta)

        decoded = reconstructed_blocks.reshape_as(normalized)
        decoded = self._apply_ldlq_feedback(normalized, decoded, selection.selected)
        denormalized = self._row_denormalize(decoded, row_norms)
        unrotated = apply_hadamard_transform(denormalized, inverse=True)
        if pad:
            unrotated = unrotated[:, :-pad]

        selected_list = [float(x) for x in selection.selected.detach().cpu().tolist()]
        assigned_betas = candidates[selection.assignments]
        high_beta_rate = torch.mean((assigned_betas > selection.selected[0]).float()).item()
        overload_rate = torch.mean(overload[:, -1].float()).item()
        greedy_error = max(selection.greedy_error, 1e-12)
        scale_index_bits = math.ceil(math.log2(max(len(selected_list), 2)))
        metadata = {
            "bpw": self.bits + (scale_index_bits / self.e8_dim),
            "q": self.q,
            "m": self.m,
            "k": self.k,
            "grid_type": self.grid_type,
            "num_scales": len(selected_list),
            "k_utilization": len(set(selected_list)) / max(self.k, 1),
            "selected_betas": selected_list,
            "overload_rate": overload_rate,
            "rescued_rate": high_beta_rate,
            "dp_error": selection.dp_error,
            "greedy_error": selection.greedy_error,
            "dp_vs_greedy_error_reduction": (greedy_error - selection.dp_error) / greedy_error,
            "ldlq_enabled": self.use_ldlq,
        }
        return restore_weight_rows(unrotated, original_shape), metadata
