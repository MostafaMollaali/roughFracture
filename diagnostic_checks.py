"""
Diagnostic checks for rough_fracture_synthesis.py
===================================================
Run this script to verify:
  1. Calibration drift  – sigma_dh before vs after crop/detrend/taper
  2. Contact area decomposition – contribution of each pipeline stage
     (mated walls | + offset | + Bandis | + Casagrande damage)

Outputs saved to _diagnostics/
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

_BASE = Path(__file__).parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(_BASE))
from rough_fracture_synthesis import (
    build_cfg,
    get_H_and_sigma1mm_from_JRC,
    _calculate_sigma_dh,
    generate_surface_stigsson,
    generate_correlated_surfaces,
    casagrande_shear,
    apply_shear_offset,
    compute_midplane_aperture,
    apply_bandis_normal_closure,
)

# ── Global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":          False,          # use mathtext (no LaTeX install needed)
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "serif"],
    "mathtext.fontset":     "dejavuserif",
    "axes.labelsize":       10,
    "axes.titlesize":       10,
    "xtick.labelsize":      8.5,
    "ytick.labelsize":      8.5,
    "legend.fontsize":      8.5,
    "figure.dpi":           150,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "axes.grid":            True,
    "grid.color":           "0.88",
    "grid.linewidth":       0.6,
    "grid.linestyle":       "-",
})

OUT = _BASE / "_out" / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

cfg       = build_cfg()
JRC_LIST  = [4.0, 7.0, 10.0]
SIGMAS    = [0.2, 2.0, 10.0]
SEEDS     = list(range(123, 133))          # 10 seeds
JRC_CLR   = ["#2166ac", "#d6604d", "#1a9641"]   # colorblind-safe blue/red/green
SIGMA_DMG = 2.0                            # MPa used during Casagrande damage
dx, dy    = float(cfg["dx"]), float(cfg["dy"])
tol       = float(cfg.get("contact_tol", 0.0))

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 – Calibration drift
# ─────────────────────────────────────────────────────────────────────────────
print("Running Check 1: calibration drift …")

drift_data = {}
for jrc in JRC_LIST:
    _, target = get_H_and_sigma1mm_from_JRC(jrc, cfg)
    post_vals = []
    for seed in SEEDS:
        _, _, z = generate_surface_stigsson(jrc, cfg, seed=seed)
        post_vals.append(_calculate_sigma_dh(z))
    drift_data[jrc] = (target, np.array(post_vals))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8),
                         gridspec_kw={"wspace": 0.38})
fig.suptitle(
    r"Calibration check: $\sigma_{\delta h}$ at 1$\,$mm lag"
    "\nbefore and after centre-crop, detrending, and edge tapering"
    f"  ($N={len(SEEDS)}$ seeds)",
    fontsize=10, y=1.02,
)

# ── Left: grouped bar ────────────────────────────────────────────────────────
ax = axes[0]
x_pos = np.arange(len(JRC_LIST))
bar_w = 0.32
targets = [drift_data[j][0] * 1e6 for j in JRC_LIST]
means   = [drift_data[j][1].mean() * 1e6 for j in JRC_LIST]
stds    = [drift_data[j][1].std()  * 1e6 for j in JRC_LIST]

ax.bar(x_pos - bar_w / 2, targets, bar_w,
       color="0.82", edgecolor="0.3", linewidth=0.8,
       label=r"Target ($N_\mathrm{big}$ grid)")
ax.bar(x_pos + bar_w / 2, means, bar_w,
       color=JRC_CLR, edgecolor="0.2", linewidth=0.8, alpha=0.88,
       label="Post crop / detrend / taper")
ax.errorbar(x_pos + bar_w / 2, means, yerr=stds,
            fmt="none", color="0.15", capsize=3.5, linewidth=0.9, zorder=5)

ax.set_xticks(x_pos)
ax.set_xticklabels([r"$\mathrm{JRC}\ 4$", r"$\mathrm{JRC}\ 7$", r"$\mathrm{JRC}\ 10$"])
ax.set_ylabel(r"$\sigma_{\delta h}\ [\mu\mathrm{m}]$")
ax.set_title(r"(a)$\;$ Mean $\sigma_{\delta h}$ before and after post-processing")
ax.set_ylim(0, 230)
ax.yaxis.set_minor_locator(MultipleLocator(10))
ax.legend(framealpha=0.9, edgecolor="0.7")

# annotate drift %
for xi, (t, m) in enumerate(zip(targets, means)):
    drift = 100 * (m - t) / t
    ax.text(xi + bar_w / 2, m + stds[xi] + 4,
            rf"${drift:+.1f}\%$", ha="center", va="bottom",
            fontsize=7.5, color="0.25")

# ── Right: per-seed drift scatter ────────────────────────────────────────────
ax = axes[1]
tol_band = 5.0
ax.axhspan(-tol_band, tol_band, color="0.93", zorder=0)
ax.axhline(0,          color="0.3",   linewidth=0.9)
ax.axhline(+tol_band,  color="tomato", linewidth=1.0, linestyle="--",
           label=r"$\pm 5\%$ tolerance")
ax.axhline(-tol_band,  color="tomato", linewidth=1.0, linestyle="--")

jrc_x = {4.0: 0, 7.0: 1, 10.0: 2}
for i, jrc in enumerate(JRC_LIST):
    target, post_vals = drift_data[jrc]
    drift_pct = 100 * (post_vals - target) / target
    xi = jrc_x[jrc]
    # individual seeds
    ax.scatter(np.full(len(SEEDS), xi) + np.random.default_rng(0).uniform(-0.08, 0.08, len(SEEDS)),
               drift_pct, color=JRC_CLR[i], alpha=0.65, s=28, zorder=3,
               label=rf"$\mathrm{{JRC}}\ {jrc:.0f}$")
    # mean bar
    ax.hlines(drift_pct.mean(), xi - 0.22, xi + 0.22,
              color=JRC_CLR[i], linewidth=2.2, zorder=4)

ax.set_xticks([0, 1, 2])
ax.set_xticklabels([r"$\mathrm{JRC}\ 4$", r"$\mathrm{JRC}\ 7$", r"$\mathrm{JRC}\ 10$"])
ax.set_ylabel(r"Drift $[\%]$")
ax.set_title(r"(b)$\;$ Per-seed drift (mean $=$ thick bar)")
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-10, 12)
ax.yaxis.set_minor_locator(MultipleLocator(1))
ax.legend(framealpha=0.9, edgecolor="0.7", ncol=2)

fig.savefig(OUT / "check1_calibration_drift.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)

for jrc in JRC_LIST:
    target, post_vals = drift_data[jrc]
    d = 100 * (post_vals.mean() - target) / target
    print(f"  JRC {jrc:.0f}: target={target*1e6:.2f} µm | "
          f"post={post_vals.mean()*1e6:.2f} µm | drift={d:+.1f}%")
print("  → saved check1_calibration_drift.png")

# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 – Contact area decomposition
# ─────────────────────────────────────────────────────────────────────────────
print("\nRunning Check 2: contact area decomposition …")

stages = [
    r"Mated walls" + "\n" + r"(no offset)",
    r"$+$ Offset" + "\n" + r"(no closure)",
    r"$+$ Bandis" + "\n" + r"(no damage)",
    r"Full pipeline" + "\n" + r"($+$ Casagrande)",
]

stage_contacts = {jrc: {s: [[] for _ in stages] for s in SIGMAS} for jrc in JRC_LIST}

for jrc in JRC_LIST:
    print(f"  JRC {jrc:.0f} …", end=" ", flush=True)
    for seed in SEEDS:
        x, y, zL, zU = generate_correlated_surfaces(jrc, cfg, seed=seed)

        a0 = compute_midplane_aperture(zL, zU, dx, dy)

        zU_s  = apply_shear_offset(zU, cfg["direction"], shift=1)
        a1    = compute_midplane_aperture(zL, zU_s, dx, dy)

        zU_d   = casagrande_shear(zU, SIGMA_DMG, cfg)
        zU_ds  = apply_shear_offset(zU_d, cfg["direction"], shift=1)
        a3_raw = compute_midplane_aperture(zL, zU_ds, dx, dy)

        for sigma_n in SIGMAS:
            a2_cl, _ = apply_bandis_normal_closure(a1,      float(sigma_n), cfg)
            a3_cl, _ = apply_bandis_normal_closure(a3_raw,  float(sigma_n), cfg)

            stage_contacts[jrc][sigma_n][0].append(100.0 * np.mean(a0    <= tol))
            stage_contacts[jrc][sigma_n][1].append(100.0 * np.mean(a1    <= tol))
            stage_contacts[jrc][sigma_n][2].append(100.0 * np.mean(a2_cl <= tol))
            stage_contacts[jrc][sigma_n][3].append(100.0 * np.mean(a3_cl <= tol))

    print("done")

# ── Global y-axis ceiling (shared across all subplots) ───────────────────────
global_max = max(
    np.mean(stage_contacts[jrc][s][3]) + np.std(stage_contacts[jrc][s][3])
    for jrc in JRC_LIST for s in SIGMAS
)
Y_MAX = np.ceil((global_max + 4) / 5) * 5   # round up to next multiple of 5

# ── Stage bar colours (light → dark, same hue) ───────────────────────────────
stage_colors = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]

fig = plt.figure(figsize=(13, 9))
fig.suptitle(
    "Contact area decomposition by pipeline stage\n"
    r"$I_c = (a_\mathrm{closed} \leq 0)$, "
    rf"$N={len(SEEDS)}$ seeds, "
    r"$b_0 = 0.2\,$mm, $\Delta s = 1\,$mm",
    fontsize=11, y=1.01,
)

gs = gridspec.GridSpec(
    len(JRC_LIST), len(SIGMAS), figure=fig,
    hspace=0.52, wspace=0.28,
    left=0.07, right=0.97, top=0.93, bottom=0.09,
)

for row, jrc in enumerate(JRC_LIST):
    for col, sigma_n in enumerate(SIGMAS):
        ax = fig.add_subplot(gs[row, col])

        vals  = [np.array(stage_contacts[jrc][sigma_n][k]) for k in range(len(stages))]
        means = [v.mean() for v in vals]
        stds  = [v.std()  for v in vals]

        bars = ax.bar(range(len(stages)), means,
                      color=stage_colors, edgecolor="0.25",
                      linewidth=0.7, width=0.6, zorder=3)
        ax.errorbar(range(len(stages)), means, yerr=stds,
                    fmt="none", color="0.15", capsize=3.5,
                    linewidth=0.9, zorder=4)

        # value labels above error bars
        for k, (m, s) in enumerate(zip(means, stds)):
            ypos = m + s + Y_MAX * 0.02
            ax.text(k, ypos, rf"${m:.1f}\%$",
                    ha="center", va="bottom",
                    fontsize=7, color="0.2")

        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(stages, fontsize=7.5)
        ax.set_ylim(0, Y_MAX)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_minor_locator(MultipleLocator(5))

        # y-label only on left column
        if col == 0:
            ax.set_ylabel(r"Contact area, $I_c\ [\%]$", fontsize=9)
        else:
            ax.set_yticklabels([])

        # panel label: (a), (b), …
        panel = chr(ord("a") + row * len(SIGMAS) + col)
        ax.set_title(
            rf"$({panel})\;$ $\mathrm{{JRC}}\,{jrc:.0f}$, "
            rf"$\sigma_n = {sigma_n}\,\mathrm{{MPa}}$",
            fontsize=9, pad=4,
        )

        # highlight full-pipeline bar with a subtle border
        bars[-1].set_edgecolor("0.1")
        bars[-1].set_linewidth(1.3)

fig.savefig(OUT / "check2_contact_decomposition.png",
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("  → saved check2_contact_decomposition.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Full-pipeline contact area [%] ===")
header = f"{'':12}" + "".join(f"  σ_n={s} MPa" for s in SIGMAS)
print(header)
for jrc in JRC_LIST:
    row_str = f"JRC {jrc:.0f}      "
    for sigma_n in SIGMAS:
        v = stage_contacts[jrc][sigma_n][3]
        row_str += f"  {np.mean(v):5.1f} ± {np.std(v):.1f}"
    print(row_str)

print(f"\nAll diagnostics saved to: {OUT}/")
