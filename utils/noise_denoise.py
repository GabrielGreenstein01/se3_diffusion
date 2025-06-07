import numpy as np
import torch
import math
torch.set_default_dtype(torch.float64)

"""
LOSS
"""

def DSM_R3(pred_score, true_score, t, tol=1e-6):
    """
    Denoising Score Matching loss for score-predicting model.

    Args:
        score_pred: (B, L, 3) - predicted score ∇ log p_t(x_t | x0)
        score_true: (B, L, 3) - ground-truth score
        t:          (B, 1)    - diffusion times
        tol:        float     - small value to clamp denominator

    Returns:
        Scalar loss
    """
    loss = (pred_score - true_score) ** 2  # (N_copies, L, 3)

    # Optional weighting: λ_t = 1 / (1 - exp(-t))
    lambda_t = (1.0 / (1.0 - torch.exp(-t).clamp(min=tol))).clamp(max=10.0).unsqueeze(-1)  # (N_copies, 1, 1)
    weighted_loss = lambda_t * loss  # (N_copies, L, 3)

    return weighted_loss.sum(dim=[-1, -2]).mean()  # mean over batch

"""
R^3 DIFFUSION
"""

def noise_translations(x0, t):
    """
    Add VP-SDE (OU process) noise to batched 3D coordinates.

    Args:
        x0: Tensor of shape (N_copies, L, 3), clean coordinates
        t:  Tensor of shape (N_copies, 1), diffusion times per batch entry

    Returns:
        x_t: Tensor of shape (N_copies, L, 3), noised coordinates at time t
    """
    # Ensure correct dtype and shape
    exp_factor = torch.exp(-0.5 * t)         # shape: (N_copies, 1)
    noise_scale = torch.sqrt(1 - torch.exp(-t))  # shape: (N_copies, 1)

    # Reshape for broadcasting
    exp_factor = exp_factor.unsqueeze(-1)       # (N_copies, 1, 1)
    noise_scale = noise_scale.unsqueeze(-1)     # (N_copies, 1, 1)

    eps = torch.randn_like(x0)  # (N_copies, L, 3)

    x_t = exp_factor * x0 + noise_scale * eps # (N_copies, L, 3)
    return x_t

def score_R3(xt, x0, t, tol: float = 1e-6):
    """
    Compute the VP-SDE score ∇ log p_t(x_t | x0), safely handling small t.

    Args:
        x_t: (N_repeats, L, 3) noised translations
        x0: (N_repeats, L, 3) clean translations
        t:   (N_repeats, 1) diffusion times
        tol: small epsilon to clamp the denominator

    Returns:
        score: (B, N, 3)
    """
    denom = (1.0 - torch.exp(-t)).clamp(min=tol).unsqueeze(-1)  # (N_copies, 1, 1)
    exp_factor = torch.exp(-0.5 * t).unsqueeze(-1)              # (N_copies, 1, 1)

    score = (1.0 / denom) * (exp_factor * x0 - xt)             # (N_copies, L, 3)
    return score

"""
IGSO3
"""

def IGSO3_expansion(omegas: torch.Tensor, t: torch.Tensor, L: int = 500, tol: float = 1e-6) -> torch.Tensor:
    """
    Vectorized SO(3) heat-kernel trace over multiple t.

    Args:
        omegas: (M,) or (N,) tensor of angles in [0, pi]
        t:      (T, 1) or (N, 1) tensor of diffusion times
        L:      number of ℓ-terms to sum
        tol:    small-angle cutoff to avoid division blow-up

    Returns:
        Tensor of shape (T, M) in grid mode or (N,) in elementwise mode
    """
    l = torch.arange(L, dtype=omegas.dtype, device=omegas.device).view(1, 1, L)

    # ----- Elementwise mode: (N,), (N, 1)
    if omegas.ndim == 1 and t.ndim == 2 and omegas.shape[0] == t.shape[0]:
        N = omegas.shape[0]
        omegas = omegas.view(N, 1, 1)  # (N, 1, 1)
        t = t.view(N, 1, 1)            # (N, 1, 1)

        sin_half = torch.sin(omegas / 2)  # (N, 1, 1)
        mask = sin_half.abs() < tol

        sin_ratio = torch.where(
            mask,
            2 * l + 1,
            torch.sin(omegas * (l + 0.5)) / sin_half
        )  # (N, 1, L)

        exp_term = torch.exp(-l * (l + 1) * t / 2)  # (N, 1, L)
        terms = (2 * l + 1) * exp_term * sin_ratio  # (N, 1, L)
        return terms.sum(dim=-1).squeeze(1)  # (N,)

    # ----- Grid mode: (M,), (T, 1)
    else:
        T = t.shape[0]
        M = omegas.shape[0]
        omegas = omegas.view(1, M, 1)  # (1, M, 1)
        t = t.view(T, 1, 1)            # (T, 1, 1)

        sin_half = torch.sin(omegas / 2)  # (1, M, 1)
        mask = sin_half.abs() < tol

        sin_ratio = torch.where(
            mask,
            2 * l + 1,
            torch.sin(omegas * (l + 0.5)) / sin_half
        )  # (1, M, L)

        exp_term = torch.exp(-l * (l + 1) * t / 2)  # (T, 1, L)
        terms = (2 * l + 1) * exp_term * sin_ratio  # (T, M, L)
        return terms.sum(dim=-1)  # (T, M)

def dIGSO3_expansion(omegas: torch.Tensor, t: torch.Tensor, L: int = 1000, tol: float = 1e-6) -> torch.Tensor:
    """
    Compute ∂ω f(ω, t) for the IGSO3 (Inverse Gaussian on SO(3)) series expansion.

    Args:
        omegas: (M,) or (N,) tensor of rotation angles ω ∈ [0, π]
        t:      (T, 1) or (N, 1) tensor of diffusion times
        L:      Number of terms ℓ = 0…L-1 to include in the truncation
        tol:    Small threshold to guard against division-by-zero near ω=0

    Returns:
        Tensor of shape (T, M) in grid mode or (N,) in elementwise mode
    """
    l = torch.arange(L, device=omegas.device, dtype=omegas.dtype).view(1, 1, L)

    # ----- Elementwise mode: (N,), (N, 1)
    if omegas.ndim == 1 and t.ndim == 2 and omegas.shape[0] == t.shape[0]:
        N = omegas.shape[0]
        omegas = omegas.view(N, 1, 1)  # (N, 1, 1)
        t = t.view(N, 1, 1)            # (N, 1, 1)

        sin_half = torch.sin(omegas / 2)  # (N, 1, 1)
        cos_half = torch.cos(omegas / 2)  # (N, 1, 1)

        num = torch.sin(omegas * (l + 0.5))  # (N, 1, L)
        dnum = (l + 0.5) * torch.cos(omegas * (l + 0.5)) * sin_half  # (N, 1, L)
        dden = 0.5 * num * cos_half                                   # (N, 1, L)

        d_ratio = (dnum - dden) / sin_half.clamp(min=tol)**2  # (N, 1, L)

        weights = (2 * l + 1) * torch.exp(-l * (l + 1) * t / 2)  # (N, 1, L)
        d_terms = weights * d_ratio  # (N, 1, L)

        return d_terms.sum(dim=-1).squeeze(1)  # (N,)

    # ----- Grid mode: (M,), (T, 1)
    else:
        T = t.shape[0]
        M = omegas.shape[0]
        omegas = omegas.view(1, M, 1)  # (1, M, 1)
        t = t.view(T, 1, 1)            # (T, 1, 1)

        sin_half = torch.sin(omegas / 2)  # (1, M, 1)
        cos_half = torch.cos(omegas / 2)  # (1, M, 1)

        num = torch.sin(omegas * (l + 0.5))  # (1, M, L)
        dnum = (l + 0.5) * torch.cos(omegas * (l + 0.5)) * sin_half  # (1, M, L)
        dden = 0.5 * num * cos_half                                   # (1, M, L)

        d_ratio = (dnum - dden) / sin_half.clamp(min=tol)**2  # (1, M, L)

        weights = (2 * l + 1) * torch.exp(-l * (l + 1) * t / 2)  # (T, 1, L)
        d_terms = weights * d_ratio  # (T, M, L)

        return d_terms.sum(dim=-1)  # (T, M)


def invert_cdf(cdf: torch.Tensor, omega_grid: torch.Tensor, K: int = 1, tol: float = 1e-6):
    """
    Vectorized inverse CDF: sample K angles per row of batched CDF.

    Args:
        cdf:         (N_copies, M) CDF per batch
        omega_grid:  (M) shared omega values
        K:           number of samples per row (e.g. L)
    Returns:
        omega_samples: (N_copies, K)
    """
    N, M = cdf.shape
    u = torch.rand(N, K, device=cdf.device)  # (N_copies, K)

    # For each u[i, j], find the smallest index k s.t. cdf[i, k - 1] <= u[i, j] < cdf[i, k] for i = 0,...,N_copies-1 and j = 0,...,K-1
    idx = torch.searchsorted(cdf, u, right=True).clamp(1, M - 1)  # (N_copies, K)
    row = torch.arange(N, device=cdf.device).unsqueeze(-1)  # (N_copies, 1)

    # cdf_[i,j] = cdf[ row[i,1], idx[i,j] - 1] for i=0,...,N_copies-1, j=0,...,K-1
    cdf_lo, cdf_hi = cdf[row, idx - 1], cdf[row, idx] # (N_copies, K)

    # omega_values[i, j] = omega_grid[idx[i, j] - 1] for i=0,...,N_copies-1, j=0,...,K-1
    omega_lo, omega_hi = omega_grid[idx - 1], omega_grid[idx] # (N_copies, K)

    w = (u - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=tol) # (N_copies, K)
    return omega_lo + w * (omega_hi - omega_lo)  # (N_copies, K)

def noise_rotations(R0, t, omega_grid, tol: float = 1e-6):
    """
    Apply IGSO3 noise to each rotation matrix in R0.

    Args:
        R0:         (N_copies, L, 3, 3)
        t:          (N_copies, 1)
        omega_grid: (M)
        tol: float

    Returns:
        R_noised: (N_copies, L, 3, 3)
    """
    N, L, _, _ = R0.shape
    device = R0.device

    d_omegas = omega_grid[1:] - omega_grid[:-1] # (M-1) Used for numerical integration
    d_omegas = d_omegas.unsqueeze(0) # (1, M-1)

    # --- Step 1: Precompute IGSO3 PDF ---
    p_unnormalized = IGSO3_expansion(omega_grid, t)  # (N_copies, M)
    integrand = (p_unnormalized[:, :-1] + p_unnormalized[:, 1:]) / 2  # (N_copies, M-1)
    Z = torch.sum( integrand * d_omegas, dim=-1, keepdim=True)  # (N_copies, 1)
    p_normalized = p_unnormalized / Z  # (N_copies, M)

    # --- Step 2: Build CDF over omega ---
    cdf = torch.zeros_like(p_normalized) # (N_copies, M)
    integrand = (p_normalized[:, :-1] + p_normalized[:, 1:]) / 2 # (N_copies, M-1)
    cdf[:, 1:] = torch.cumsum(integrand * d_omegas, dim=-1) # (N_copies, 1)
    normalized_cdf = cdf / cdf[:, -1].unsqueeze(-1).clamp(min=tol)  # (N_copies, M) / (N_copies, 1) ==> (N_copies, M)

    # Sample L angles from CDF
    omega_samples = invert_cdf(cdf, omega_grid, K=L)                # (N_copies, L)

    # Sample axis u ∈ S² and scale by ω
    u = torch.randn(N, L, 3, device=device) # (N_copies, L, 3)
    u = u / u.norm(dim=-1, keepdim=True).clamp(min=tol)
    k = u * omega_samples.unsqueeze(-1)                       # (N_copies, L, 3)

    K = vec2skewsym(k)                                        # (N_copies, L, 3, 3)

    dR = expmap(K)                                                  # (N_copies, L, 3, 3)

    # Apply noise: R_noised =  R0 @ dR
    R_noised = torch.einsum('...ik,...kj -> ...ij', R0, dR)                              # (N_copies, L, 3, 3)

    return R_noised

def score_SO3(R: torch.Tensor,
              R0: torch.Tensor,
              t: torch.Tensor,
              tol: float = 1e-6) -> torch.Tensor:
    """
    Compute the score s_t(R|R0) = ∇_R log p_{t|0}(R|R0) on SO(3)
    using a precomputed table of ∂_ω log f(ω, t) over (omega, t) grid.

    Args:
        R         (N_copies, L, 3, 3): Noised rotations
        R0        (N_copies, L, 3, 3): Clean rotations
        t         (N_copies, 1):       Diffusion times
        tol       (float):            Stability threshold for small ω

    Returns:
        Tensor of shape (N_copies, L, 3, 3): Tangent vectors (scores) in T_SO(3)
    """
    N, L = R.shape[:2]

    # 1. Relative rotation Δ = R0ᵀ R
    delta = torch.einsum('...ki,...kj->...ij', R0, R)  # (N_copies, L, 3, 3)

    # 2. Log map and rotation angle
    log_delta = vec2skewsym(logmap(delta, tol=tol))   # (N_copies, L, 3, 3)
    omega = rot2angle(delta)                          # (N_copies, L)

    # 3. Reshape t to match (N_copies * L, 1) manually (instead of repeat_interleave)
    t_expanded = t.unsqueeze(-1).expand(-1, L, -1).reshape(-1, 1)  # (N_copies * L, 1)
    omega_flat = omega.reshape(-1)                               # (N_copies * L,)

    # 4. Radial score ∂ω log f(ω,t); f cancels out so Z is not needed
    f_val = IGSO3_expansion(omega_flat, t_expanded)    # (N_copies * L,)
    df_val = dIGSO3_expansion(omega_flat, t_expanded)  # (N_copies * L,)
    dlogf = df_val / f_val.clamp(min=tol)                            # (N_copies * L,)

    # 5. Compute scale = (∂ log f / ∂ω) / ω
    scale = dlogf / omega_flat.clamp(min=tol)                        # (N_copies * L,)
    scale = scale.view(N, L, 1, 1)                            # (N_copies, L, 1, 1)

    # 6. Final score vector: s = R @ (scale * log(Δ))
    return torch.einsum('...ik,...kj->...ij', R, scale * log_delta)  # (N_copies, L, 3, 3)

"""
SO(3) UTILITY FUNCTIONS
"""

def vec2skewsym(v: torch.Tensor) -> torch.Tensor:
    """
    v: (..., 3)
    returns K: (..., 3, 3) where K @ w == v × w
    """
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)

    # build each row as a stack, then stack the rows
    row0 = torch.stack([ zero,  -z,   y ], dim=-1)
    row1 = torch.stack([  z,   zero, -x ], dim=-1)
    row2 = torch.stack([ -y,    x,  zero], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)

def skewsym2vec(K: torch.Tensor) -> torch.Tensor:
    """
    Convert skew-symmetric matrix/matrices K ∈ so(3) to axis vector k ∈ R^3.

    Args:
        K: Tensor of shape (..., 3, 3), each slice K[..., i, j] skew-symmetric:
           K[..., i, j] = -K[..., j, i].

    Returns:
        k: Tensor of shape (..., 3), where
           k[..., 0] = K[..., 2, 1],
           k[..., 1] = K[..., 0, 2],
           k[..., 2] = K[..., 1, 0].
    """
    kx = K[..., 2, 1]
    ky = K[..., 0, 2]
    kz = K[..., 1, 0]
    return torch.stack([kx, ky, kz], dim=-1)

def rot2angle(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix or batch of rotation matrices to rotation angles.

    Args:
        R: tensor of shape [..., 3, 3], each a valid SO(3) matrix.

    Returns:
        theta: tensor of shape [...] giving the rotation angle in radians ∈ [0, π].
    """
    # 1) Cosine term via trace
    #    θ = acos( (tr(R) - 1) / 2 )
    cos_theta = (torch.einsum('...ii', R) - 1.0) / 2.0
    cos_theta = cos_theta.clamp(-1.0, 1.0)

    # 2) Sine term via Frobenius norm of the skew part
    #    ‖R - Rᵀ‖_F = 2√2 |sin θ|  ⇒  |sin θ| = ‖R - Rᵀ‖_F / (2√2)
    M = R - R.transpose(-2, -1)                 # [...,3,3]
    sq = torch.einsum('...ij,...ij->...', M, M) # sum of squares → [...]
    sin_theta = torch.sqrt(sq) / (2.0 * math.sqrt(2))

    # 3) Recover θ robustly with atan2
    theta = torch.atan2(sin_theta, cos_theta)

    return theta

def expmap(K: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """
    Exponential map so(3) → SO(3) via Rodrigues’ formula,
    using torch.sinc to handle θ → 0 and θ = π robustly.

    Computes:
        R = I
          + (sinθ/θ)·K
          + ((1−cosθ)/θ²)·K²
    where K is skew‐symmetric, k = skewsym2vec(K), and θ = ‖k‖.

    Args:
        K:   Tensor of shape (..., 3, 3), skew‐symmetric so that K @ w = axis × w.
             (Typically produced by vec2skewsym from a 3-vector.)

    Returns:
        R:   Tensor of shape (..., 3, 3), valid rotation matrices in SO(3).
    """
    # 1) Extract axis-angle vector k ∈ ℝ³ and its norm θ
    k = skewsym2vec(K)                              # (..., 3)
    theta = k.norm(dim=-1, keepdim=True)            # (..., 1)

    # 2) Compute sinθ/θ via sinc: sinc(x) = sin(πx)/(πx)
    #    so sinc(theta/π) = sinθ/θ, with sinc(0)=1 exactly
    sin_term = torch.sinc(theta / torch.pi)[..., None]  # (..., 1, 1)

    # 3) Compute (1 - cosθ)/θ², guarding tiny denominators via isclose
    theta_sq = theta * theta                          # (..., 1)
    theta_sq_safe = theta_sq.clamp_min(tol)  # anything < tol is now tol
    cos_term = ((1 - torch.cos(theta)) / theta_sq_safe)[..., None]  # (..., 1, 1)

    # 4) Build K² and broadcast identity
    K2 = K @ K                                        # (..., 3, 3)
    I = torch.eye(3, device=K.device, dtype=K.dtype)
    I = I.expand(*K.shape[:-2], 3, 3)                 # (..., 3, 3)

    # 5) Assemble Rodrigues' formula
    return I + sin_term * K + cos_term * K2

def shifted_expmap(R0: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Apply an exponential‐map update on SO(3):

        R = R0 · exp( R0ᵀ K )

    Args:
        R0:    [..., 3, 3] current rotation matrix (∈ SO(3))
        K:     [..., 3, 3] a skew‐symmetric “twist” matrix

    Returns:
        R:     [..., 3, 3] updated rotation after applying the exp‐map
    """
    # pull the twist into R0’s local frame: so3 = R0ᵀ · K
    so3 = R0.transpose(-2, -1) @ K
    so3 = (so3 - torch.transpose(so3, -2, -1))/2 # included for numerical stability

    # map that to SO(3) and push back into the world frame
    return R0 @ expmap(so3)

def logmap(R: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """
    Inverse Rodrigues map: SO(3) → R^3 axis-angle (k = θ·û).
    Uses:
      - skewsym2vec for vee(R - Rᵀ)
      - rot2angle for θ via trace
    Branches:
      • 0 ≤ θ < π−tol:  k = (θ/(2 sinθ)) ω = 0.5·ω/sinc(θ/π)
      • θ ≈ π:         k_i = ±π·√((R_ii+1)/2), sign via dot(ω,k)
    """
    # 1) raw S^-1(R - Rᵀ)
    omegas = skewsym2vec(R - R.transpose(-2, -1))        # (...,3)
    # 2) angle θ
    theta  = rot2angle(R).unsqueeze(-1)                  # (...,1)

    # 3) mask for π-case
    near_pi = theta.squeeze(-1) > (torch.pi - tol)       # (...,)

    # 4) allocate
    k = torch.empty_like(omegas)                         # (...,3)

    # 5) general 0 ≤ θ < π−tol
    if (~near_pi).any():
        k[~near_pi] = omegas[~near_pi]  * (0.5 / torch.sinc(theta[~near_pi] / torch.pi))

    # 6) π-case
    if near_pi.any():
        R_pi   = R[near_pi]                              # (m,3,3)
        diag = torch.einsum('...ii->...i', R_pi)
        u = torch.sqrt(((diag + 1) / 2).clamp_min(0.0))  # (m, 3)
        k_i   = u * torch.pi                            # (m,3)

        # sign via dot-product for stability
        signs = torch.sign((omegas[near_pi] * k_i).sum(dim=1, keepdim=True)).clamp(min=tol) # (m,1)
        k[near_pi] = k_i * signs                        # broadcast

    return k
