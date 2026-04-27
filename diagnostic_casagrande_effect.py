"""
Diagnostic Check 3: Casagrande effect summary
=============================================
Quantifies how Casagrande shear damage changes:
  1) roughness (Δσ_dh),
  2) steep asperities (Δβ_p99),
  3) closed-contact area contribution after Bandis closure,
  4) geometry-change magnitude (max |Δz|).
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

_BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_BASE))

from rough_fracture_synthesis import (
    build_cfg,
    generate_correlated_surfaces,
    casagrande_shear,
    compute_apparent_dip_signed,
    _calculate_sigma_dh,
    apply_shear_offset,
    compute_midplane_aperture,
    apply_bandis_normal_closure,
)


OUT = _BASE / "_out" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

cfg = build_cfg()
JRC_LIST = [4.0, 7.0, 10.0]
SIGMAS = [0.2, 2.0, 10.0]
SEEDS = list(range(123, 133))
dx, dy = float(cfg["dx"]), float(cfg["dy"])
tol = float(cfg.get("contact_tol", 0.0))
direction = str(cfg.get("direction", "x"))


def _new_metric_store() -> dict:
    return {j: {s: [] for s in SIGMAS} for j in JRC_LIST}


dsdh_um = _new_metric_store()      # sigma_dh_after - sigma_dh_before [um]
dbeta_deg = _new_metric_store()    # beta_p99_after - beta_p99_before [deg]
dcontact_pp = _new_metric_store()  # full - bandis_only [percentage points]
dzmax_um = _new_metric_store()     # max |z_dmg - z| [um]

print("Running Check 3: Casagrande effect summary …")
for jrc in JRC_LIST:
    print(f"  JRC {jrc:.0f} …", end=" ", flush=True)
    for sigma_n in SIGMAS:
        for seed in SEEDS:
            _, _, zL, zU = generate_correlated_surfaces(jrc, cfg, seed=seed)

            beta0 = compute_apparent_dip_signed(zU, dx, dy, direction)
            sd0 = _calculate_sigma_dh(zU)

            zU_d = casagrande_shear(
                surface=zU,
                sigma_n_mpa=float(sigma_n),
                cfg=cfg,
                direction=direction,
            )

            beta1 = compute_apparent_dip_signed(zU_d, dx, dy, direction)
            sd1 = _calculate_sigma_dh(zU_d)

            zU_s = apply_shear_offset(zU, direction, shift=1)
            zU_ds = apply_shear_offset(zU_d, direction, shift=1)
            a_nodmg = compute_midplane_aperture(zL, zU_s, dx, dy)
            a_full = compute_midplane_aperture(zL, zU_ds, dx, dy)
            a_nodmg_cl, _ = apply_bandis_normal_closure(a_nodmg, float(sigma_n), cfg)
            a_full_cl, _ = apply_bandis_normal_closure(a_full, float(sigma_n), cfg)

            ic_nodmg = 100.0 * float(np.mean(a_nodmg_cl <= tol))
            ic_full = 100.0 * float(np.mean(a_full_cl <= tol))

            dsdh_um[jrc][sigma_n].append((sd1 - sd0) * 1e6)
            dbeta_deg[jrc][sigma_n].append(
                float(np.percentile(beta1, 99) - np.percentile(beta0, 99))
            )
            dcontact_pp[jrc][sigma_n].append(ic_full - ic_nodmg)
            dzmax_um[jrc][sigma_n].append(float(np.max(np.abs(zU_d - zU)) * 1e6))
    print("done")


def _to_mean_std(metric: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.zeros((len(JRC_LIST), len(SIGMAS)))
    std = np.zeros((len(JRC_LIST), len(SIGMAS)))
    for i, j in enumerate(JRC_LIST):
        for k, s in enumerate(SIGMAS):
            arr = np.asarray(metric[j][s], dtype=float)
            mean[i, k] = float(np.mean(arr))
            std[i, k] = float(np.std(arr))
    return mean, std


M1, S1 = _to_mean_std(dsdh_um)
M2, S2 = _to_mean_std(dbeta_deg)
M3, S3 = _to_mean_std(dcontact_pp)
M4, S4 = _to_mean_std(dzmax_um)

fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.8), constrained_layout=True)

v1 = float(np.max(np.abs(M1))) or 1.0
v2 = float(np.max(np.abs(M2))) or 1.0
v3 = float(np.max(np.abs(M3))) or 1.0

panels = [
    ("(a)", axs[0, 0], M1, S1, "RdBu_r",
     colors.TwoSlopeNorm(vcenter=0.0, vmin=-v1, vmax=v1),
     r"$\Delta \sigma_{\delta h} = \sigma_{\delta h,\mathrm{after}} - \sigma_{\delta h,\mathrm{before}}$ [$\mu$m]"),
    ("(b)", axs[0, 1], M2, S2, "RdBu_r",
     colors.TwoSlopeNorm(vcenter=0.0, vmin=-v2, vmax=v2),
     r"$\Delta \beta_{p99} = \beta_{p99,\mathrm{after}} - \beta_{p99,\mathrm{before}}$ [deg]"),
    ("(c)", axs[1, 0], M3, S3, "RdBu",
     colors.TwoSlopeNorm(vcenter=0.0, vmin=-v3, vmax=v3),
     r"$\Delta I_c = I_{c,\mathrm{full}} - I_{c,\mathrm{Bandis-only}}$ [pp]"),
    ("(d)", axs[1, 1], M4, S4, "magma",
     colors.Normalize(vmin=float(np.min(M4)), vmax=float(np.max(M4)) if float(np.max(M4)) > float(np.min(M4)) else float(np.min(M4)) + 1.0),
     r"$\max |\Delta z|$ per seed [$\mu$m]"),
]

for panel_label, ax, mat, std, cmap, norm, title in panels:
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(f"{panel_label} {title}", fontsize=10)
    ax.set_xticks(np.arange(len(SIGMAS)))
    ax.set_xticklabels([f"{s:g}" for s in SIGMAS])
    ax.set_yticks(np.arange(len(JRC_LIST)))
    ax.set_yticklabels([f"{j:g}" for j in JRC_LIST])
    ax.set_xlabel(r"$\sigma_n$ [MPa]")
    ax.set_ylabel("JRC")

    # Cell boundaries for a cleaner "table-like" academic look.
    ax.set_xticks(np.arange(-.5, len(SIGMAS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(JRC_LIST), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=1.0, alpha=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i,
                f"{mat[i, j]:.2f}\n±{std[i, j]:.2f}",
                ha="center", va="center", fontsize=8, color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.8),
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

fig.suptitle(
    "Check 3 — Casagrande Damage Metrics by JRC and Normal Stress",
    fontsize=13, y=0.995,
)
fig.text(
    0.5, 0.005,
    f"Cells report mean ± standard deviation across N={len(SEEDS)} seeds (not min/max). "
    f"For panel (d), max|Δz| is computed per seed over the full grid, then summarized as mean ± std. "
    f"Configuration: b0={cfg['b0']*1e3:.2f} mm, shear direction={direction}.",
    ha="center", va="bottom", fontsize=9,
)

out_png = OUT / "check3_casagrande_effect_summary.png"
fig.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"  → saved {out_png.name}")
print(f"\nAll diagnostics saved to: {OUT}/")
