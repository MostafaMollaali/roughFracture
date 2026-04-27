
"""
Synthetic Self-Affine Fracture Generation
==========================================
Standalone Python implementation following Algorithm 1 in the paper appendix
(Stigsson 2025 / Casagrande 2018 / Bandis-Barton 1983 framework).

F_slide uses the Casagrande (2018) wedge criterion: tan(φ_b + β).
  The alternative max(tan β − tan φ_b, 0) gives thresholds of 50–78° which
  never trigger for typical fracture roughness. The LaTeX text was incorrect
  and should be updated to match this implementation.

References
----------
Stigsson et al. 2025 (EESci), Stigsson 2019 (JRC), Casagrande et al. 2018,
Barton & Choubey 1977, Barton & Bandis 1980, Bandis et al. 1983.
"""

# ============================================================
# Imports
# ============================================================
import math
from pathlib import Path

import numpy as np
import pyvista as pv
from numpy.fft import fftfreq, irfft2, rfftfreq

# ============================================================
# Algorithm 1 – Synthetic self-affine sheared fracture workflow
# (text-aligned with the LaTeX algorithm block)
# ============================================================
#
# Input:
#   JRC, sigma_n, (dx, dy), (lx_user, ly_user), b0, [lambda_min, lambda_max],
#   shear_dir in {x, y}, shear_offset Δs, Δβ (b_step_deg), η (closure_frac),
#   k_n (kn_joint), (phi_b, c_intact), guards (cos_eps, eps_nz), m_max, N_big.
#
# Output: (z_lower, z_upper, a_closed(x,y;sigma_n))
#
# Step A – Map JRC → (H, σ_δh,target)            [get_H_and_sigma1mm_from_JRC]
# Step B – Snap grid, validate wavelength band    [grid snapping + checks]
# Step C – Synthesize mated walls                 [generate_correlated_surfaces]
#   C1: build periodic z_big on N_big grid        [Eq. PSD: A(K)∝K^{-(H+1)}]
#   C2: band-limit to K ∈ [K_min, K_max]
#   C3: rescale to σ_δh,target at 1-mm lag        [Eq. discrete calibration]
#   C4: center-crop → n_x × n_y; de-mean; optional detrend/taper
#   C5: z_lower = z0 - b0/2,     z_upper = z0 + b0/2       [Eq. mated walls]
# Step D – Stress-dependent strength anchors      [get_phi_and_jcs_table]
# Step E – Casagrande-type shear damage on z_d    [casagrande_shear]
#   for m = 1..m_max:
#     compute β_fs                                [compute_apparent_dip_signed]
#     β* ← floor(max(β_fs)/Δβ)·Δβ
#     while β* > 0:
#       A(β*) = {(i,j) : β_fs ≥ β* > 0}         [Eq. active set]
#       compute σ_N,cf = min(σ_n/cos(β), JCS)    [Eq. local normal stress]
#       F_shear = Σ [c + σ_N,cf·tan(φ)] · A_cell [Eq. F_shear]
#       F_slide = Σ [σ_N,cf·max(tan(β)−tan(φ_b),0)] · A_cell  [Eq. F_slide]
#       if F_slide ≥ F_shear:
#         clip |∂z/∂s| ≤ tan(β*); Poisson correct  [clip_slope_in_direction]
#         updated ← True; break
#       β* ← β* − Δβ
#     if not updated: break
# Step F – Offset, midplane aperture, closure
#   periodic shear offset Δs                      [np.roll]
#   compute Δz_c, n_m, a                          [compute_midplane_aperture]
#   Δb_m = η·b0; Δb_n via Bandis; a_closed        [apply_bandis_normal_closure]
# ============================================================


# ============================================================
# Default configuration
# ============================================================
CFG_DEFAULT = {
    # Domain + grid
    "lx_user": 0.10,          # [m]
    "ly_user": 0.10,          # [m]
    "dx": 1.0e-3,             # [m] fixed 1 mm
    "dy": 1.0e-3,             # [m] fixed 1 mm
    # Roughness band
    "lambda_min": 4.0e-3,     # [m]  must be >= 4·dx
    "lambda_max": 0.05,       # [m]  must be <= min(lx, ly)
    # Fracture geometry
    "b0": 2.0e-4,             # [m] initial mechanical aperture
    # Strength / friction
    "phi_basic_deg": 30.0,    # [deg] base friction angle φ_b
    # Effective asperity cohesion at mm scale. Intact-rock cohesion (28 MPa) applies
    # to field-scale fractures; at 1 mm grid scale breakage never triggers at
    # 0.2–10 MPa. Use an effective value calibrated to produce realistic contact area.
    "cohesion_intact_mpa": 0.5,   # [MPa] effective mm-scale asperity cohesion
    # Numerical guards
    "cos_eps": 1e-6,          # lower bound for cos(β) to avoid blow-up
    "tan_clip_deg": 89.5,     # clip angle (deg) for tan() argument
    # Damage controls
    "damage_wall": "upper",   # "upper" or "lower"
    "direction": "x",         # shear direction "x" or "y"
    "b_step_deg": 0.5,        # Δβ increment [deg]
    "max_outer": 50,          # m_max
    "verbose": False,
    "debug_casagrande": False,      # print shear-activation summary per case
    "debug_contact_stages": False,  # print no-damage vs full-pipeline contact deltas
    # Normal stress cases [MPa]
    "sigmas_mpa": [0.2, 2.0, 10.0],
    # Contact / closure
    "contact_tol": 1e-9,
    "kn_joint": 2e10,         # [Pa/m]  k_n in Bandis formula
    "closure_frac": 0.9,      # η: Δb_m = η · b0
    # Surface generation
    "jrc": [4.0, 7.0, 10.0],
    "seed": 123,
    "n_seeds": 10,
    # Stigsson (2019/2025) anchor table for JRC → (H, σ_δh at 1-mm lag)
    "jrc_anch":       np.array([4.0, 7.0, 10.0]),
    "h_anch":         np.array([0.7, 0.8, 0.9]),
    "sig1mm_anch_mm": np.array([0.0969, 0.1440, 0.1910]),  # [mm]
    # Post-processing (disabled by default to preserve periodic consistency
    # with FFT Poisson solve + periodic shear roll).
    "detrend": False,
    "edge_taper_nodes": 0,
    "N_big": 2049,            # oversized FFT grid for center-crop
}


def build_cfg(user: dict | None = None) -> dict:
    """Merge user overrides into defaults; derive nx, ny, lx, ly."""
    cfg = {**CFG_DEFAULT}
    if user:
        cfg.update(user)
    # Backward compatibility for older configs.
    if user and ("cohesion_intact_mpa" not in user) and ("cohesion_joint_mpa" in user):
        cfg["cohesion_intact_mpa"] = cfg["cohesion_joint_mpa"]
    cfg["nx"] = int(round(cfg["lx_user"] / cfg["dx"])) + 1
    cfg["ny"] = int(round(cfg["ly_user"] / cfg["dy"])) + 1
    cfg["lx"] = cfg["dx"] * (cfg["nx"] - 1)
    cfg["ly"] = cfg["dy"] * (cfg["ny"] - 1)
    cfg["phi_basic_rad"] = math.radians(cfg["phi_basic_deg"])
    cfg["cohesion_intact_pa"] = cfg["cohesion_intact_mpa"] * 1e6
    # Keep legacy alias so old downstream code does not break.
    cfg["cohesion_joint_pa"] = cfg["cohesion_intact_pa"]
    _validate_cfg(cfg)
    return cfg


def _validate_cfg(cfg: dict) -> None:
    lam_max_hard = min(cfg["lx"], cfg["ly"])
    lam_min_safe = 4.0 * max(cfg["dx"], cfg["dy"])
    if cfg["lambda_max"] > lam_max_hard:
        raise ValueError(
            f"lambda_max={cfg['lambda_max']} m exceeds min(lx,ly)={lam_max_hard} m."
        )
    if cfg["lambda_min"] < lam_min_safe:
        raise ValueError(
            f"lambda_min={cfg['lambda_min']} m < 4·dx={lam_min_safe} m."
        )
    if cfg["lambda_min"] >= cfg["lambda_max"]:
        raise ValueError("lambda_min must be < lambda_max.")


# ============================================================
# Step A – Map JRC → (H, σ_δh,target)
# Eq. (dh_definition), (self_affine_scaling)
# ============================================================

def get_H_and_sigma1mm_from_JRC(jrc: float, cfg: dict) -> tuple[float, float]:
    """
    Linear interpolation within Stigsson anchor table.
    Raises ValueError outside [jrc_min, jrc_max].
    Returns (H, sigma_dh_target [m]).
    """
    j = float(jrc)
    jrc_anch = cfg["jrc_anch"]
    h_anch   = cfg["h_anch"]
    sig_anch = cfg["sig1mm_anch_mm"]

    j_min, j_max = float(jrc_anch.min()), float(jrc_anch.max())
    if j < j_min or j > j_max:
        raise ValueError(
            f"JRC={j:g} outside Stigsson anchor range [{j_min:g}, {j_max:g}]."
        )
    H         = float(np.interp(j, jrc_anch, h_anch))
    sig_1mm_m = float(np.interp(j, jrc_anch, sig_anch)) * 1e-3  # mm → m
    return H, sig_1mm_m


# ============================================================
# Step B helpers – grid snapping and band constraints
# Eqs. (grid_snapping_n), (grid_snapping_L),
#      (band_constraints_lambda), (band_constraints_K)
# ============================================================
# Grid snapping is done in build_cfg above.
# Band constraints are validated in _validate_cfg above.


# ============================================================
# Step C1–C4 – Spectral synthesis helpers
# ============================================================

def _calculate_sigma_dh(surface: np.ndarray) -> float:
    """
    Std of height differences at adjacent nodes (1-mm lag).
    Eq. (discrete_calibration).
    """
    dzdx = surface[1:, :] - surface[:-1, :]
    dzdy = surface[:, 1:] - surface[:, :-1]
    return float(np.std(np.concatenate([dzdx.ravel(), dzdy.ravel()])))


def detrend_plane(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Remove best-fit plane z = ax + by + c (least squares)."""
    nx, ny = z.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(nx * ny)])
    coef, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    return z - (coef[0] * X + coef[1] * Y + coef[2])


def apply_edge_taper(z: np.ndarray, n: int) -> np.ndarray:
    """
    Cosine taper over n nodes at each boundary edge.
    Reduces tilted-edge artifacts from FFT cropping.
    n=0 disables.
    """
    if n <= 0:
        return z
    nx, ny = z.shape
    n = int(min(n, (nx - 1) // 2, (ny - 1) // 2))
    if n <= 0:
        return z
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n)))
    w = np.ones((nx, ny))
    w[:n,  :] *= ramp[:, None]
    w[-n:, :] *= ramp[::-1, None]
    w[:,  :n] *= ramp[None, :]
    w[:, -n:] *= ramp[None, ::-1]
    return z * w


def generate_surface_stigsson(
    jrc: float,
    cfg: dict,
    seed: int | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Step C: generate one periodic self-affine height field.

    Spectral construction (Eqs. psd_2d, amp_scaling):
      S_2D(K) ∝ K^{-(2H+2)}  →  A(K) ∝ K^{-(H+1)},  A(0) = 0
    Band-limited to K ∈ [K_min, K_max].
    Synthesized on N_big grid, calibrated (Eq. discrete_calibration),
    center-cropped, de-meaned, detrended, tapered.

    Returns (x, y, z) with z of shape (nx, ny).
    """
    nx, ny   = int(cfg["nx"]), int(cfg["ny"])
    lx, ly   = float(cfg["lx"]), float(cfg["ly"])
    dx, dy   = float(cfg["dx"]), float(cfg["dy"])
    N_big    = int(cfg.get("N_big", 2049))

    if lambda_min is None:
        lambda_min = float(cfg["lambda_min"])
    if lambda_max is None:
        lambda_max = float(cfg["lambda_max"])

    if nx > N_big or ny > N_big:
        raise ValueError(f"nx×ny={nx}×{ny} exceeds N_big={N_big}.")

    H, sigma_dh_target = get_H_and_sigma1mm_from_JRC(jrc, cfg)

    # Wavenumber limits (Eq. band_constraints_K)
    k_min_grid = 2.0 * np.pi / max((N_big - 1) * dx, (N_big - 1) * dy)
    k_max_grid = np.pi / min(dx, dy)
    k_min_user = 2.0 * np.pi / lambda_max
    k_max_user = 2.0 * np.pi / lambda_min
    k_min = max(k_min_grid, k_min_user)
    k_max = min(k_max_grid, k_max_user)
    if k_min >= k_max:
        raise ValueError(f"Empty k-band: k_min={k_min:.3e}, k_max={k_max:.3e}")

    # Build isotropic amplitude spectrum on N_big grid
    # Eq. amp_scaling: A(K) ∝ K^{-(H+1)}
    kx = 2.0 * np.pi * fftfreq(N_big, d=dx)
    ky = 2.0 * np.pi * rfftfreq(N_big, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX**2 + KY**2)

    alpha = H + 1.0           # amplitude exponent
    K_nz  = K.copy()
    K_nz[0, 0] = 1.0          # avoid division by zero at DC
    amplitude = K_nz ** (-alpha)
    amplitude[0, 0] = 0.0     # zero DC (zero mean)
    amplitude *= (K >= k_min) & (K <= k_max)  # band-limit

    # Random phases → complex Fourier coefficients
    rng = np.random.default_rng(seed)
    phase  = rng.uniform(0.0, 2.0 * np.pi, size=amplitude.shape)
    Zhat   = amplitude * (np.cos(phase) + 1j * np.sin(phase))
    Zhat[0, 0] = 0.0

    # Inverse FFT → real-space field on N_big grid
    z_big = irfft2(Zhat, s=(N_big, N_big))
    z_big -= float(np.mean(z_big))

    # Calibrate to target σ_δh at 1-mm lag (Eq. discrete_calibration)
    sigma_num = _calculate_sigma_dh(z_big)
    if sigma_num <= 0.0:
        raise RuntimeError("sigma_num = 0; check spectrum/band settings.")
    z_big *= sigma_dh_target / sigma_num

    # Center-crop to target domain
    i0 = (N_big - nx) // 2
    j0 = (N_big - ny) // 2
    z  = z_big[i0:i0 + nx, j0:j0 + ny]

    # Optional post-processing
    if cfg.get("detrend", True):
        z = detrend_plane(z, dx, dy)
    n_taper = int(cfg.get("edge_taper_nodes", 0))
    if n_taper > 0:
        z = apply_edge_taper(z, n_taper)

    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    return x, y, z


# ============================================================
# Step C5 – Mated walls construction
# Eq. (mated_walls)
# ============================================================

def generate_correlated_surfaces(
    jrc: float,
    cfg: dict,
    seed: int | None = None,
    lambda_min: float | None = None,
    lambda_max: float | None = None,
    b0: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two perfectly mated walls from one rough surface z0 [Eq. mated_walls]:
      z_lower = z0 - b0/2
      z_upper = z0 + b0/2
    Both walls follow the same topography; gap = b0 everywhere when mated.
    When b0 = 0 the walls coincide exactly (zero aperture).
    Returns (x, y, z_lower, z_upper).
    """
    if b0 is None:
        b0 = float(cfg["b0"])
    x, y, z = generate_surface_stigsson(jrc, cfg, seed, lambda_min, lambda_max)
    z0      = z - z.mean()
    z_lower = z0 - 0.5 * b0
    z_upper = z0 + 0.5 * b0
    return x, y, z_lower, z_upper


# ============================================================
# Step D – Stress-dependent φ(σ_n) and JCS(σ_n)
# Eq. (phi_jcs_anchors) with cubic smoothstep interpolation
# ============================================================

def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def get_phi_and_jcs_table(sigma_n_mpa: float) -> tuple[float, float]:
    """
    Barton-type anchor interpolation:
      σ_n = [0.2, 2, 20] MPa → φ = [60, 50, 30]°, JCS = [209, 153, 97] MPa.
    Raises ValueError outside [0.2, 20] MPa.
    Returns (phi_rad, jcs_pa).
    """
    s = float(sigma_n_mpa)
    s0, s1, s2 = 0.2, 2.0, 20.0
    if s < s0 or s > s2:
        raise ValueError(
            f"sigma_n={s:g} MPa outside Stigsson/Forsmark anchor range [{s0},{s2}] MPa."
        )
    if s <= s1:
        w = _smoothstep((s - s0) / (s1 - s0))
        phi_deg = 60.0 + (50.0 - 60.0) * w
        jcs_mpa = 209.0 + (153.0 - 209.0) * w
    else:
        w = _smoothstep((s - s1) / (s2 - s1))
        phi_deg = 50.0 + (30.0 - 50.0) * w
        jcs_mpa = 153.0 + (97.0 - 153.0) * w
    return np.deg2rad(phi_deg), jcs_mpa * 1e6


# ============================================================
# Step E helpers – apparent dip, forces, slope clipping
# ============================================================

def compute_apparent_dip_signed(
    surface: np.ndarray,
    dx: float,
    dy: float,
    direction: str = "x",
) -> np.ndarray:
    """
    Signed apparent dip β_fs [deg] per cell in the shear direction.
    Eq. (beta_fs_x): cell average of forward-difference slopes.
    Shape: (nx-1, ny-1).
    """
    z = surface
    if direction == "x":
        # average slope over cell in x: (z[i+1,j]+z[i+1,j+1]-z[i,j]-z[i,j+1])/(2·dx)
        num = (z[1:, :-1] + z[1:, 1:] - z[:-1, :-1] - z[:-1, 1:]) * 0.5
        slope = num / dx
    elif direction == "y":
        num = (z[:-1, 1:] + z[1:, 1:] - z[:-1, :-1] - z[1:, :-1]) * 0.5
        slope = num / dy
    else:
        raise ValueError("direction must be 'x' or 'y'")
    return np.degrees(np.arctan(slope))


def calculate_forces_facetwise(
    beta_deg: np.ndarray,
    active_mask: np.ndarray,
    sigma_n_pa: float,
    phi: float,         # stress-dependent friction angle [rad]
    phi_b: float,       # base friction angle [rad]
    jcs_pa: float,
    a_cell: float,
    cfg: dict,
) -> tuple[float, float]:
    """
    Aggregate shear and sliding forces over the active set A(β*).

    F_shear = Σ_{A} [c + σ_N,cf · tan(φ)] · A_cell         (Eq. F_shear)
    F_slide = Σ_{A} [σ_N,cf · tan(φ_b + β)] · A_cell        (Casagrande 2018)

    Wedge/asperity shear criterion:
      tan(φ_b + β) = force per unit σ_N needed to push material OVER an asperity
      of angle β against base friction φ_b. Trigger when this exceeds the asperity
      breaking strength c + σ_N·tan(φ).

    This gives realistic β_thresholds of 1–25° depending on σ_n.
    The alternative max(tan β − tan φ_b, 0) gives thresholds of 50–78°
    which never trigger damage for typical fracture roughness — physically wrong.
    The LaTeX text requires correction to match this formula.
    """
    facing_mask = (beta_deg > 0.0) & active_mask
    if not np.any(facing_mask):
        return 0.0, 0.0

    tan_clip_deg = float(cfg["tan_clip_deg"])
    if ("cohesion_joint_mpa" in cfg) and (
        ("cohesion_intact_mpa" not in cfg) or
        (float(cfg["cohesion_joint_mpa"]) != float(cfg["cohesion_intact_mpa"]))
    ):
        c_mpa = float(cfg["cohesion_joint_mpa"])
    else:
        c_mpa = float(cfg.get("cohesion_intact_mpa", 0.5))
    c_pa = c_mpa * 1e6

    beta_rad = np.deg2rad(beta_deg[facing_mask])

    # Ensemble contact-area stress concentration (Stigsson approach).
    # Project active facet areas onto horizontal; concentration = A_domain / A_contact.
    # This is physically correct and does not diverge at steep β (unlike σ_n / cos β).
    A_domain   = a_cell * float(beta_deg.size)
    A_cf_total = float(np.sum(a_cell * np.cos(beta_rad)))
    eps_a      = float(cfg.get("eps", 1e-12))
    sigma_N    = min(sigma_n_pa * A_domain / max(A_cf_total, eps_a), jcs_pa)

    # Shear resistance (Eq. F_shear) — sum over all N_facing cells
    n_facing  = int(np.sum(facing_mask))
    F_shear   = float(n_facing) * a_cell * (c_pa + sigma_N * math.tan(phi))

    # Sliding driving — Casagrande wedge criterion: tan(φ_b + β)
    angle     = np.clip(phi_b + beta_rad,
                        -np.deg2rad(tan_clip_deg),
                         np.deg2rad(tan_clip_deg))
    F_slide   = float(np.sum(a_cell * sigma_N * np.tan(angle)))

    return F_shear, F_slide


def clip_slope_in_direction(
    surface: np.ndarray,
    dx: float,
    dy: float,
    direction: str,
    beta_deg: np.ndarray,
    active_mask: np.ndarray,
    b_star_deg: float,
) -> np.ndarray:
    """
    Clip |∂z/∂s| ≤ tan(β*) on nodes connected to active cells,
    then recover a consistent height field via a periodic Poisson solve.
    Eq. (slope_clip).
    """
    if not np.any(active_mask):
        return surface.copy()

    z = surface.copy()
    nx, ny = z.shape
    tan_b_star = math.tan(math.radians(b_star_deg))

    # Central-difference gradient on nodes
    gx = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2.0 * dx)
    gy = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2.0 * dy)

    # Node mask: union of four corners of each active cell
    active_nodes = np.zeros_like(z, dtype=bool)
    active_nodes[:-1, :-1] |= active_mask
    active_nodes[1:,  :-1] |= active_mask
    active_nodes[:-1, 1:]  |= active_mask
    active_nodes[1:,  1:]  |= active_mask

    gx_clip = gx.copy()
    gy_clip = gy.copy()

    # Clip only uphill (positive) slopes at active nodes.
    # Downhill slopes at the same nodes are not facing facets and must not be clipped.
    if direction == "x":
        m = active_nodes
        gx_clip[m] = np.minimum(gx[m], tan_b_star)
    else:
        m = active_nodes
        gy_clip[m] = np.minimum(gy[m], tan_b_star)

    delta_gx = gx_clip - gx
    delta_gy = gy_clip - gy

    if np.allclose(delta_gx, 0.0) and np.allclose(delta_gy, 0.0):
        return z

    # Periodic Poisson solve: ∇²(Δz) = div(δg)
    # In Fourier: -K²·ΔZ_k = i·KX·DGx_k + i·KY·DGy_k
    # → ΔZ_k = -(i·KX·DGx_k + i·KY·DGy_k) / K²
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    DGx_k  = np.fft.fft2(delta_gx)
    DGy_k  = np.fft.fft2(delta_gy)
    RHS_k  = 1j * KX * DGx_k + 1j * KY * DGy_k

    denom  = KX**2 + KY**2
    denom[0, 0] = np.inf       # regularize DC (zero-mean correction)
    DeltaZ_k    = -RHS_k / denom
    DeltaZ_k[0, 0] = 0.0

    delta_z = np.real(np.fft.ifft2(DeltaZ_k))
    return z + delta_z


# ============================================================
# Step E – Casagrande shear damage (outer loop)
# Algorithm 1, Step E
# ============================================================

def casagrande_shear(
    surface: np.ndarray,
    sigma_n_mpa: float,
    cfg: dict,
    direction: str | None = None,
    b_step_deg: float | None = None,
    max_outer: int | None = None,
    verbose: bool | None = None,
) -> np.ndarray:
    """
    Casagrande-type iterative shear damage on one fracture wall.

    For each outer iteration m:
      1. Compute β_fs per cell (Eq. beta_fs_x).
      2. Set β* = floor(max(β_fs)/Δβ)·Δβ.
      3. Sweep β* downward:
         a. A(β*) = {(i,j) : β_fs ≥ β*}         (Eq. active_set)
         b. Compute σ_N,cf, F_shear, F_slide      (Eqs. sigmaN_eff, F_shear, F_slide)
         c. If F_slide ≥ F_shear:                 (Eq. update_trigger)
              clip slopes; apply Poisson correction (Eq. slope_clip)
              mark updated; break inner loop
         d. else β* ← β* − Δβ
      4. If no update in entire sweep: stop.

    Returns damaged surface z_d.
    """
    if direction  is None: direction  = cfg.get("direction",  "x")
    if b_step_deg is None: b_step_deg = float(cfg.get("b_step_deg", 0.5))
    if max_outer  is None: max_outer  = int(cfg.get("max_outer",  50))
    if verbose    is None: verbose    = bool(cfg.get("verbose",  False))
    debug_casa = bool(cfg.get("debug_casagrande", False))

    dx = float(cfg["dx"])
    dy = float(cfg["dy"])

    sigma_n = sigma_n_mpa * 1e6
    phi, jcs = get_phi_and_jcs_table(sigma_n_mpa)
    phi_b = float(cfg["phi_basic_rad"])
    A_cell = dx * dy

    z = surface.copy()
    z_initial = surface.copy()
    beta_init_max = float(np.max(compute_apparent_dip_signed(z, dx, dy, direction)))
    update_events = 0
    updated_nodes_accum = 0
    triggered_cells_accum = 0
    max_dz_event = 0.0
    outer_with_updates = 0

    for outer in range(max_outer):
        beta_deg = compute_apparent_dip_signed(z, dx, dy, direction)
        beta_max = float(np.max(beta_deg))

        if verbose:
            print(f"  Outer {outer}: beta_max={beta_max:.2f}°")

        if beta_max < 1e-3:
            if verbose:
                print(f"  Outer {outer}: max β_fs < 0.001°, stopping.")
            break

        # β* starts at floor(beta_max / Δβ)·Δβ
        b_star = math.floor(beta_max / b_step_deg) * b_step_deg
        any_updated = False

        while b_star > 0.0:
            # Active set A(β*): Eq. (active_set)
            active_mask = (beta_deg >= b_star)
            if not np.any(active_mask):
                b_star -= b_step_deg
                continue

            F_shear, F_slide = calculate_forces_facetwise(
                beta_deg=beta_deg,
                active_mask=active_mask,
                sigma_n_pa=sigma_n,
                phi=phi,
                phi_b=phi_b,
                jcs_pa=jcs,
                a_cell=A_cell,
                cfg=cfg,
            )

            if verbose:
                ratio = (F_slide / F_shear) if F_shear > 0 else 0.0
                if ratio > 0.9:
                    print(
                        f"    β*={b_star:.2f}°: F_shear={F_shear:.3e}, "
                        f"F_slide={F_slide:.3e}, ratio={ratio:.3f}"
                    )

            # Update trigger (Eq. update_trigger): F_slide ≥ F_shear
            if F_slide >= F_shear and F_shear > 0.0:
                z_new = clip_slope_in_direction(
                    surface=z,
                    dx=dx, dy=dy,
                    direction=direction,
                    beta_deg=beta_deg,
                    active_mask=active_mask,
                    b_star_deg=b_star,
                )
                if np.allclose(z_new, z):
                    b_star -= b_step_deg
                    continue
                dz_field = np.abs(z_new - z)
                dz = float(np.max(dz_field))
                changed_nodes = int(np.count_nonzero(dz_field > 1e-14))
                if verbose:
                    print(f"      Updated at β*={b_star:.2f}°, max|Δz|={dz:.3e} m")
                z = z_new
                any_updated = True
                update_events += 1
                updated_nodes_accum += changed_nodes
                triggered_cells_accum += int(np.sum(active_mask))
                max_dz_event = max(max_dz_event, dz)
                beta_deg = compute_apparent_dip_signed(z, dx, dy, direction)
                # Stay at same β* to check if further clipping is needed
            else:
                b_star -= b_step_deg

        if not any_updated:
            if verbose:
                print(f"  Outer {outer}: no updates, converged.")
            break
        outer_with_updates += 1

    if debug_casa:
        beta_final_max = float(np.max(compute_apparent_dip_signed(z, dx, dy, direction)))
        dz_total = np.abs(z - z_initial)
        changed_nodes_total = int(np.count_nonzero(dz_total > 1e-14))
        changed_pct = 100.0 * changed_nodes_total / float(z.size)
        dz_rms = float(np.sqrt(np.mean(dz_total**2)))
        dz_max = float(np.max(dz_total))
        print(
            f"  Casagrande activation: σ_n={sigma_n_mpa:.2f} MPa | "
            f"events={update_events}, outer_iters_with_updates={outer_with_updates}, "
            f"triggered_cells_total={triggered_cells_accum}"
        )
        print(
            f"    β_max: {beta_init_max:.3f}° -> {beta_final_max:.3f}° | "
            f"changed nodes={changed_nodes_total}/{z.size} ({changed_pct:.2f}%)"
        )
        print(
            f"    |Δz| max_total={dz_max:.3e} m, rms_total={dz_rms:.3e} m, "
            f"max_per_event={max_dz_event:.3e} m, changed_nodes_accum={updated_nodes_accum}"
        )

    return z


# ============================================================
# Step F.1 – Periodic shear offset
# Eq. (shear_offset)
# ============================================================

def apply_shear_offset(z: np.ndarray, direction: str, shift: int = 1) -> np.ndarray:
    """
    Periodic shear displacement of `shift` grid cells in `direction`.
    Default shift = 1 → Δs = dx (for x) or dy (for y).
    Eq. (shear_offset).
    """
    if direction == "x":
        return np.roll(z, shift=shift, axis=0)
    elif direction == "y":
        return np.roll(z, shift=shift, axis=1)
    else:
        raise ValueError("direction must be 'x' or 'y'")


def peak_shear_displacement_cells(
    jrc: float,
    jcs_mpa: float,
    sigma_n_mpa: float,
    L_m: float,
    cell_size: float,
) -> int:
    """
    Peak shear displacement (Barton & Bandis 1982, Eq. 1):
      u_peak = (JRC / 500) * (JCS / sigma_n)^0.33 * L   [same units as L]

    Returns displacement rounded to nearest grid cell (minimum 1).
    """
    u_peak = (jrc / 500.0) * (jcs_mpa / max(sigma_n_mpa, 1e-3)) ** 0.33 * L_m
    return max(1, int(round(u_peak / cell_size)))


# ============================================================
# Step F.2 – Midplane aperture
# Eqs. (centroid_sep), (midplane_normal), (midplane_aperture)
# ============================================================

def compute_midplane_aperture(
    z_lower: np.ndarray,
    z_upper: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """
    Midplane-based local aperture on cell centres.

    1. Centroid separation: Δz_c = zU_c - zL_c    (Eq. centroid_sep)
    2. Midplane normal n_m = (n_L + n_U) / |n_L + n_U|   (Eq. midplane_normal)
    3. a = max(Δz_c / n_{m,z}, 0)                 (Eq. midplane_aperture)

    Returns a of shape (nx-1, ny-1).
    """
    if z_lower.shape != z_upper.shape:
        raise ValueError("z_lower and z_upper shapes must match.")
    nx, ny = z_lower.shape
    if nx < 2 or ny < 2:
        raise ValueError("Need at least 2×2 nodes.")

    # Cell-centroid elevations (average of four nodal corners)
    zL_c = 0.25 * (z_lower[:-1, :-1] + z_lower[1:, :-1]
                   + z_lower[:-1, 1:] + z_lower[1:, 1:])
    zU_c = 0.25 * (z_upper[:-1, :-1] + z_upper[1:, :-1]
                   + z_upper[:-1, 1:] + z_upper[1:, 1:])
    dz_c = zU_c - zL_c      # Eq. (centroid_sep)

    # Surface normals via central-difference slopes on cell centres
    def _normals(z):
        sx = (z[1:, :-1] + z[1:, 1:] - z[:-1, :-1] - z[:-1, 1:]) / (2.0 * dx)
        sy = (z[:-1, 1:] + z[1:, 1:] - z[:-1, :-1] - z[1:, :-1]) / (2.0 * dy)
        # Upward unit normal: n = (-∂z/∂x, -∂z/∂y, 1) / |...|
        norm = np.sqrt(sx**2 + sy**2 + 1.0)
        return -sx / norm, -sy / norm, 1.0 / norm

    nxL, nyL, nzL = _normals(z_lower)
    nxU, nyU, nzU = _normals(z_upper)

    # Midplane normal: sum and re-normalise  (Eq. midplane_normal)
    nxM = nxL + nxU
    nyM = nyL + nyU
    nzM = nzL + nzU
    normM = np.sqrt(nxM**2 + nyM**2 + nzM**2)
    normM = np.where(normM == 0.0, 1.0, normM)
    nzM  /= normM

    # Aperture: max(Δz_c · n_{m,z}, 0)  (Eq. midplane_aperture)
    # Geometric fact: for two parallel tilted planes, the perpendicular aperture
    # equals the vertical centroid separation scaled by the z-component of the
    # midplane normal (cos of tilt angle). Dividing inflated apertures at steep
    # tilts and caused a false singularity.
    a = dz_c * nzM
    return np.maximum(a, 0.0)


# ============================================================
# Step F.3 – Barton–Bandis normal closure
# Eq. (bandis_closure), (closed_aperture)
# ============================================================

def bandis_params_from_jrc(
    jrc: float,
    jcs_mpa: float,
    b0_m: float,
    cfg: dict,
) -> tuple[float, float]:
    """
    JRC/JCS-dependent Bandis stiffness and maximum closure.

    Initial normal stiffness (Barton et al. 1985, Eq. 9):
      k_ni = -7.15 + 1.75·JRC + 0.02·(JCS/E0)   [GPa/m]
      where E0 = b0 in µm.

    Maximum closure (Bandis et al. 1983, Table 5 fit):
      Δb_m = η · b0   (η from cfg, or derived below)
      Literature shows Δb_m ≈ b0 / (1 + 0.02·JCS·b0_mm^{-0.5})
      — a simple JCS-dependent damping; use cfg["closure_frac"] as fallback.

    Returns (kn_joint [Pa/m], delta_bm [m]).
    """
    b0_mm = b0_m * 1e3                               # m → mm  (Barton uses mm)

    # k_ni (Barton et al. 1985, Eq. 9): E0 must be in mm, JCS in MPa → GPa/m
    kni_gpa = -7.15 + 1.75 * jrc + 0.02 * (jcs_mpa / max(b0_mm, 1e-3))
    kni_gpa = float(np.clip(kni_gpa, 1.0, 100.0))   # clamp to [1, 100] GPa/m
    kn_pa_m = kni_gpa * 1e9                          # GPa/m → Pa/m

    # Maximum closure fraction η (Bandis et al. 1983, Table 5 fit):
    # rougher joints close proportionally more; bounded to [0.4, 0.9]
    eta = float(np.clip(0.5 + 0.03 * jrc, 0.4, 0.9))
    delta_bm = eta * b0_m

    return kn_pa_m, delta_bm


def apply_bandis_normal_closure(
    a: np.ndarray,
    sigma_n_mpa: float,
    cfg: dict,
    b0: float | None = None,
    kn_joint: float | None = None,
    closure_frac: float | None = None,
    jrc: float | None = None,
    jcs_mpa: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Hyperbolic normal closure (Bandis et al. 1983):
      Δb_n = (σ_n / (σ_n + k_n · Δb_m)) · Δb_m       (Eq. bandis_closure)
      a_closed = max(a − Δb_n, 0)                      (Eq. closed_aperture)

    If jrc and jcs_mpa are provided, k_n and Δb_m are derived from
    Barton et al. (1985) / Bandis et al. (1983) instead of cfg constants.

    Returns (a_closed, delta_b_n [m]).
    """
    if b0 is None: b0 = float(cfg["b0"])
    sigma_n = float(sigma_n_mpa) * 1e6     # [Pa]

    if jrc is not None and jcs_mpa is not None:
        # Physics-based stiffness and closure from JRC/JCS
        kn_joint, delta_bm = bandis_params_from_jrc(jrc, jcs_mpa, b0, cfg)
    else:
        # Fallback: cfg constants (backward-compatible)
        if kn_joint   is None: kn_joint    = float(cfg["kn_joint"])
        if closure_frac is None: closure_frac = float(cfg.get("closure_frac", 0.9))
        delta_bm = closure_frac * b0

    if delta_bm <= 0.0 or sigma_n <= 0.0:
        return np.maximum(a, 0.0), 0.0

    denom     = sigma_n + kn_joint * delta_bm
    delta_b_n = (sigma_n / denom) * delta_bm   # [m]
    a_closed  = np.maximum(a - delta_b_n, 0.0)
    return a_closed, float(delta_b_n)


# ============================================================
# Top-level runner: Algorithm 1 in full
# ============================================================

def run_one_case(
    jrc: float,
    sigma_n_mpa: float,
    cfg: dict,
    seed: int | None = None,
) -> dict:
    """
    Execute Algorithm 1 (Steps A–F) for one (JRC, σ_n, seed) case.

    Returns a dict with:
      x, y, z_lower, z_upper       – wall geometry
      z_lower_d / z_upper_d         – damaged wall
      a_raw, a_closed               – aperture fields (nx-1, ny-1)
      delta_b_n                     – scalar Bandis closure [m]
      contact                       – binary contact indicator
      k_parallel_plate_proxy        – LCL/parallel-plate proxy permeability [m²]
    """
    if seed is None:
        seed = int(cfg.get("seed", 123))

    # ---- Steps A–C: generate mated walls ----------------------------
    x, y, z_lower, z_upper = generate_correlated_surfaces(
        jrc, cfg, seed=seed,
        lambda_min=cfg.get("lambda_min"),
        lambda_max=cfg.get("lambda_max"),
        b0=cfg.get("b0"),
    )

    # ---- Step D: strength anchors (inside casagrande_shear) ---------

    # ---- Step E: Casagrande shear damage ----------------------------
    damage_wall = cfg.get("damage_wall", "upper")
    direction   = cfg.get("direction",   "x")

    if damage_wall == "upper":
        z_dmg   = casagrande_shear(z_upper, sigma_n_mpa, cfg, direction=direction)
        z_lower_d, z_upper_d = z_lower, z_dmg
    else:
        z_dmg   = casagrande_shear(z_lower, sigma_n_mpa, cfg, direction=direction)
        z_lower_d, z_upper_d = z_dmg, z_upper

    # ---- Step F.1: periodic shear offset ----------------------------
    # Fixed one-cell shift, consistent with the paper (Stigsson 2025).
    z_upper_shifted = apply_shear_offset(z_upper_d, direction, shift=1)

    # ---- Step F.2: midplane aperture --------------------------------
    a_raw = compute_midplane_aperture(
        z_lower_d, z_upper_shifted, float(cfg["dx"]), float(cfg["dy"])
    )

    # ---- Step F.3: Bandis normal closure ----------------------------
    # Pass jrc + jcs so k_n and Δb_m are derived from Barton/Bandis empirics.
    _, jcs_pa_val = get_phi_and_jcs_table(sigma_n_mpa)
    debug_contact = bool(cfg.get("debug_contact_stages", False))
    if debug_contact:
        z_upper_shifted_nodmg = apply_shear_offset(z_upper, direction, shift=1)
        a_raw_nodmg = compute_midplane_aperture(
            z_lower, z_upper_shifted_nodmg, float(cfg["dx"]), float(cfg["dy"])
        )
        a_closed_nodmg, _ = apply_bandis_normal_closure(
            a_raw_nodmg, sigma_n_mpa, cfg,
            jrc=jrc, jcs_mpa=jcs_pa_val / 1e6,
        )

    a_closed, delta_b_n = apply_bandis_normal_closure(
        a_raw, sigma_n_mpa, cfg,
        jrc=jrc, jcs_mpa=jcs_pa_val / 1e6,
    )

    # ---- Contact indicator and permeability -------------------------
    tol     = float(cfg.get("contact_tol", 0.0))
    contact = (a_closed <= tol).astype(np.uint8)
    if debug_contact:
        ic_raw_nodmg = 100.0 * float(np.mean(a_raw_nodmg <= tol))
        ic_raw_full  = 100.0 * float(np.mean(a_raw <= tol))
        ic_cl_nodmg  = 100.0 * float(np.mean(a_closed_nodmg <= tol))
        ic_cl_full   = 100.0 * float(np.mean(a_closed <= tol))
        print(
            f"  Contact stages: JRC={jrc:.1f}, σ_n={sigma_n_mpa:.2f} MPa | "
            f"raw(no dmg)={ic_raw_nodmg:.3f}%, raw(full)={ic_raw_full:.3f}%"
        )
        print(
            f"    closed(+Bandis only)={ic_cl_nodmg:.3f}%, "
            f"closed(full)={ic_cl_full:.3f}%, "
            f"ΔCasagrande={ic_cl_full - ic_cl_nodmg:.3f} %-points"
        )
    a_pos   = np.maximum(a_closed, 0.0)
    # Clamp aperture before cubic law so contact cells (a=0) don't yield k=0,
    # which can produce a singular flow-solver system matrix.
    a_min  = float(cfg.get("a_min_flow", 1e-8))   # [m] residual flow path
    a_flow = np.maximum(a_pos, a_min)

    # Optional roughness correction to cubic law (Zimmerman & Bodvarsson 1996, Eq. 4):
    #   T = (a³/12) · (1 − 1.5·(σ_a/ā)²)³  valid for CV < 0.4 (open cells only).
    # Disabled by default: all roughness in this model is explicitly resolved on the
    # 1-mm grid (λ_min = 4 mm > dx), so there is no sub-grid roughness to correct for.
    # Enable via cfg["roughness_correction"] = True only if sub-grid roughness is present.
    if cfg.get("roughness_correction", False):
        open_a = a_pos[a_pos > 0]
        if len(open_a) > 0 and open_a.std() / open_a.mean() < 0.4:
            cv   = float(open_a.std() / open_a.mean())
            corr = max((1.0 - 1.5 * cv**2) ** 3, 0.1)
        else:
            corr = 1.0   # CV too high or no open cells: skip correction
    else:
        corr = 1.0
    k_parallel_plate_proxy = corr * a_flow**2 / 12.0  # LCL/parallel-plate proxy

    # Reynolds number diagnostic (reference pressure gradient: ΔP/L from cfg)
    dP   = float(cfg.get("delta_p", 200.0))   # [Pa]  default OGS benchmark value
    L    = float(cfg["lx"])                    # [m]   domain length in flow direction
    re   = compute_reynolds(a_closed, grad_p=dP / L)

    return {
        "x": x, "y": y,
        "z_lower": z_lower, "z_upper": z_upper,
        "z_lower_d": z_lower_d, "z_upper_d": z_upper_shifted,
        "a_raw": a_raw,
        "a_closed": a_closed,
        "delta_b_n": delta_b_n,
        "contact": contact,
        "k_parallel_plate_proxy": k_parallel_plate_proxy,
        # Legacy alias retained for compatibility with existing OGS projects.
        "k_frac": k_parallel_plate_proxy,
        "reynolds": re,
    }


# ============================================================
# Reynolds number diagnostic
# ============================================================

def compute_reynolds(
    a_closed: np.ndarray,
    grad_p: float,
    rho: float = 1000.0,   # [kg/m³] water density
    mu: float  = 1e-3,     # [Pa·s]  dynamic viscosity
) -> dict:
    """
    Estimate local Reynolds number for each cell using the cubic-law
    Darcy velocity and the hydraulic diameter D_h = 2w.

    Physics background
    ------------------
    Geometry-specific interpretation (no universal fracture cutoff):
      Re < ~4        cubic law tends to be most reliable
      ~4 <= Re < 100 inertial/nonlinear effects grow
      Re >= 100      strong nonlinearity likely

    Formula
    -------
    Darcy velocity (cubic law):  v = (w² / 12μ) · |∇P|
    Hydraulic diameter:          D_h = 2w          [parallel plates]
    Reynolds number:             Re = ρ · v · D_h / μ = 2ρ · v · w / μ

    Parameters
    ----------
    a_closed  : closed aperture field [m], shape (nx-1, ny-1)
    grad_p    : magnitude of pressure gradient [Pa/m]
    rho, mu   : fluid properties

    Returns
    -------
    dict with:
      Re        – local Re field (same shape as a_closed)
      Re_max    – maximum Re
      Re_mean   – mean Re over open cells (a_closed > 0)
      Re_p95    – 95th-percentile Re
      lcl_reliable_frac    – fraction of cells with Re < ~4
      transition_frac      – fraction with ~4 ≤ Re < 100
      nonlinear_frac       – fraction with Re ≥ 100
      w_max     – maximum aperture [m]
    """
    w = np.maximum(a_closed, 0.0)              # non-negative aperture [m]
    v = (w**2 / (12.0 * mu)) * grad_p         # Darcy velocity [m/s]
    Re = (2.0 * rho * v * w) / mu             # Re = 2ρvw/μ  (D_h = 2w)

    open_mask = w > 0.0
    Re_open   = Re[open_mask]

    lcl_reliable_frac = float(np.mean(Re < 4.0))
    transition_frac   = float(np.mean((Re >= 4.0) & (Re < 100.0)))
    nonlinear_frac    = float(np.mean(Re >= 100.0))

    return {
        "Re":                Re,
        "Re_max":            float(Re.max()),
        "Re_mean":           float(Re_open.mean()) if Re_open.size > 0 else 0.0,
        "Re_p95":            float(np.percentile(Re_open, 95)) if Re_open.size > 0 else 0.0,
        "lcl_reliable_frac": lcl_reliable_frac,
        "transition_frac":   transition_frac,
        "nonlinear_frac":    nonlinear_frac,
        # Legacy aliases for existing scripts.
        "darcy_valid_frac":  lcl_reliable_frac,
        "forchheimer_frac":  transition_frac,
        "turbulent_frac":    nonlinear_frac,
        "w_max":             float(w.max()),
    }


def print_reynolds_report(re_dict: dict, label: str = "") -> None:
    """Print a structured Reynolds number validity report."""
    print(f"\n{'─'*55}")
    if label:
        print(f"  Re diagnostic: {label}")
    print(f"  w_max          = {re_dict['w_max']*1e3:.3f} mm")
    print(f"  Re_max         = {re_dict['Re_max']:.1f}")
    print(f"  Re_mean (open) = {re_dict['Re_mean']:.2f}")
    print(f"  Re_p95  (open) = {re_dict['Re_p95']:.2f}")
    print(f"  LCL-reliable  (Re < ~4)         : {re_dict['lcl_reliable_frac']*100:.1f}% of cells")
    print(f"  Transition    (~4 ≤ Re < 100)   : {re_dict['transition_frac']*100:.1f}% of cells")
    print(f"  Nonlinear     (Re ≥ 100)        : {re_dict['nonlinear_frac']*100:.1f}% of cells")
    if re_dict['Re_max'] > 4.0:
        print("  ⚠  Re_max > ~4: inertial effects likely in wider channels")
    if re_dict['Re_max'] > 100.0:
        print("  ⚠  Re_max > 100: strong nonlinearity likely")
    print(f"{'─'*55}")


# ============================================================
# VTU export helper
# ============================================================

def build_quad_mesh_vtu(x_nodes, y_nodes, z0: float = 0.0):
    """Build a 2D structured quad UnstructuredGrid (pyvista)."""
    x_nodes = np.asarray(x_nodes)
    y_nodes = np.asarray(y_nodes)
    nx, ny  = len(x_nodes), len(y_nodes)

    Xn, Yn = np.meshgrid(x_nodes, y_nodes, indexing="ij")
    pts = np.column_stack([Xn.ravel(), Yn.ravel(), np.full(nx * ny, z0)])

    n_cells = (nx - 1) * (ny - 1)
    cells   = np.empty((n_cells, 5), dtype=np.int64)
    k = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            p0 = i + j * nx
            cells[k] = [4, p0, p0 + 1, p0 + nx + 1, p0 + nx]
            k += 1
    celltypes = np.full(n_cells, 9, dtype=np.uint8)   # VTK_QUAD = 9
    grid = pv.UnstructuredGrid(cells.ravel(), celltypes, pts)

    x_cells = 0.5 * (x_nodes[:-1] + x_nodes[1:])
    y_cells = 0.5 * (y_nodes[:-1] + y_nodes[1:])
    return grid, x_cells, y_cells


def export_vtu(result: dict, vtu_path: str | Path, cfg: dict) -> str:
    """
    Save aperture / contact / permeability fields to VTU.
    Fields (cell-wise):
      aperture_raw, aperture_closed, contact, aperture_pos, k_parallel_plate_proxy
    """
    x, y = result["x"], result["y"]
    grid, _, _ = build_quad_mesh_vtu(x, y, z0=0.0)

    grid.cell_data["aperture_raw"]    = result["a_raw"].ravel(order="C")
    grid.cell_data["aperture_closed"] = result["a_closed"].ravel(order="C")
    grid.cell_data["contact"]         = result["contact"].ravel(order="C")
    grid.cell_data["aperture_pos"]    = np.maximum(result["a_closed"], 0.0).ravel(order="C")
    grid.cell_data["k_parallel_plate_proxy"] = result["k_parallel_plate_proxy"].ravel(order="C")
    # Keep legacy field name for existing OGS project files.
    grid.cell_data["k_frac"] = result["k_parallel_plate_proxy"].ravel(order="C")

    vtu_path = Path(vtu_path)
    vtu_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(str(vtu_path))
    return str(vtu_path)


# ============================================================
# Main: parameter sweep (JRC × σ_n × seeds)
# ============================================================

def main():
    cfg = build_cfg()

    JRC_LIST   = cfg["jrc"]
    SIGMAS_MPA = cfg["sigmas_mpa"]
    N_SEEDS    = int(cfg.get("n_seeds", 10))
    SEED_BASE  = int(cfg.get("seed", 123))
    SEED_LIST  = [SEED_BASE + i for i in range(N_SEEDS)]
    OUT_ROOT   = Path(cfg.get("out_root", "_out/results"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Grid: {cfg['nx']}×{cfg['ny']} nodes, "
          f"lx={cfg['lx']:.3f} m, ly={cfg['ly']:.3f} m")
    print(f"Sweep: JRC={JRC_LIST}, σ_n={SIGMAS_MPA} MPa, seeds {SEED_LIST}\n")

    for jrc in JRC_LIST:
        for seed in SEED_LIST:
            for sigma_n in SIGMAS_MPA:
                print(f"JRC={jrc}  seed={seed}  σ_n={sigma_n} MPa ... ", end="", flush=True)

                res = run_one_case(jrc, sigma_n, cfg, seed=seed)

                mean_a   = float(res["a_closed"].mean())
                contact_pct = float(np.mean(res["contact"]) * 100.0)
                print(f"<a>={mean_a:.3e} m, contact={contact_pct:.1f}%, "
                      f"Δb_n={res['delta_b_n']:.3e} m")
                print_reynolds_report(
                    res["reynolds"],
                    label=f"JRC={jrc} seed={seed} σ_n={sigma_n} MPa"
                )

                case_dir = OUT_ROOT / f"JRC_{jrc}" / f"seed_{seed}" / f"sigma_{sigma_n}MPa"
                export_vtu(res, case_dir / "fracture_aperture_k.vtu", cfg)

    print(f"\nDone. Output under: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()
