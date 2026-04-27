# %%
# Auto-generated from OGSrun_synthesis_roughFracture.ipynb
# Generated on 2026-04-27

# %% cell 0
from __future__ import annotations

# stdlib
import json
import os
import sys
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any
from xml.etree.ElementTree import Element
try:
    import pandas as pd
except ModuleNotFoundError:  # optional; only needed for verify dataframe output
    pd = None

# third-party: numerics / scientific
import numpy as np
from numpy.random import default_rng
from scipy.spatial import cKDTree
from scipy.stats import truncnorm

# third-party: visualization / IO
import matplotlib.pyplot as plt
import pyvista as pv
import imageio.v2 as iio  # keep v2 for broad compatibility
import matplotlib as mpl

# progress bars
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb

# domain-specific
import ogstools as ot

# notebook display
from IPython.display import Image, Markdown, display

# %% cell 1
user_parameters_rough = {
    "prefix": "roughFracture_synthesis_LF",
    "t_end": "1000",
    "initial_dt": ".1",
    "minimum_dt": "1e-3",
    "maximum_dt": "100",
    "specific_body_force": "0 0 -9.81",
    # BC mode: "fixed_pressure" (Dirichlet left/right) or "fixed_flux" (Neumann left + Dirichlet right)
    "bc_mode": "fixed_flux",
    "initial_pressure": {"type":"Constant","value":100000},     # "initial_pressure": {"expression":"1000*9.81*(z-0.015) + 1e5","type":"Function"},
    "outlet_pressure": {"type":"Constant","value":100000},
    "inlet_pressure": {"type":"Constant","value":100200},
    # Used only when bc_mode="fixed_flux". With equation_balance_type="mass",
    # units are kg/(m^2 s), and positive value means inflow of mass.
    # Start conservative; increase only if needed.
    # Quick scaling: q_new = q_old * (Δp_target / Δp_observed).
    "inlet_specific_flux": 0.03,
    "porosity_value": "1.0",
    "fluid_density": {
        "type": "Linear",
        "reference_value": "1000",
        "variable_name": "liquid_phase_pressure",
        "reference_condition": "1e5",
        "slope": "4.5e-7"
    },
    "equation_balance_type": "mass",
    'permeability_mesh_field': 'permeability_cubic',
    "fracture_thickness_mesh_field": "aperture_closed",
    "user_w_min_aperture": "1e-8"
}

JRC_LIST = [4, 7, 10]
sigmas_mpa = [0.2, 2.0, 10.0]

# %% cell 2 [markdown]
# ## prj file helpers

# %% cell 3
def _xpath_param(name: str) -> str:
    return f".//parameters/parameter[name='{name}']"

def _ensure_parameter(project, name: str) -> str:
    """
    Ensure <parameter><name>name</name> exists. Return its base xpath.
    """
    base = _xpath_param(name)
    if project.tree.find(base) is None:
        project.add_element(".//parameters", "parameter")
        last = ".//parameters/parameter[last()]"
        project.add_element(last, "name", name)
    return base

def _clear_children(project, base_xpath: str, tags: list[str]) -> None:
    """
    Remove possibly-existing child tags under base_xpath.
    """
    for t in tags:
        project.remove_element(f"{base_xpath}/{t}")

def _set_parameter_constant_value(project, name: str, value: Any) -> None:
    base = _ensure_parameter(project, name)
    _clear_children(project, base, ["type", "value", "values", "field_name", "expression"])
    project.add_element(base, "type", "Constant")
    project.add_element(base, "value", str(value))

def _set_parameter_constant_values(project, name: str, values_text: str) -> None:
    base = _ensure_parameter(project, name)
    _clear_children(project, base, ["type", "value", "values", "field_name", "expression"])
    project.add_element(base, "type", "Constant")
    project.add_element(base, "values", str(values_text))

def _set_parameter_mesh_element(project, name: str, field_name: str) -> None:
    base = _ensure_parameter(project, name)
    _clear_children(project, base, ["type", "value", "values", "field_name", "expression"])
    project.add_element(base, "type", "MeshElement")
    project.add_element(base, "field_name", str(field_name))

def _set_parameter_function(project, name: str, expression: str) -> None:
    base = _ensure_parameter(project, name)
    _clear_children(project, base, ["type", "value", "values", "field_name", "expression"])
    project.add_element(base, "type", "Function")
    project.add_element(base, "expression", str(expression))

def _update_pressure_param(project, xml_name: str, spec: Any) -> None:
    """
    xml_name: parameter name in prj, e.g. 'p0', 'p_left', 'p_right'
    spec:
      - {"type":"Constant","value": ...}
      - {"expression":"...", "type":"Function"} or just "..."
    """
    if isinstance(spec, dict) and spec.get("type") == "Constant":
        _set_parameter_constant_value(project, xml_name, spec["value"])
    else:
        expr = spec if not isinstance(spec, dict) else spec.get("expression", "")
        _set_parameter_function(project, xml_name, expr)

def _ensure_pressure_bc_container(project) -> str:
    """Ensure boundary_conditions exists under pressure process variable."""
    bc_xpath = ".//process_variables/process_variable[name='pressure']/boundary_conditions"
    if project.tree.find(bc_xpath) is None:
        pv_xpath = ".//process_variables/process_variable[name='pressure']"
        project.add_element(pv_xpath, "boundary_conditions")
    return bc_xpath


def _reset_pressure_boundary_conditions(project, bc_defs: list[dict[str, str]]) -> None:
    """
    Replace all pressure BCs with bc_defs.

    Each item in bc_defs needs keys: type, mesh, parameter.
    """
    bc_xpath = _ensure_pressure_bc_container(project)

    while project.tree.find(f"{bc_xpath}/boundary_condition") is not None:
        project.remove_element(f"{bc_xpath}/boundary_condition")

    for bc in bc_defs:
        project.add_element(bc_xpath, "boundary_condition")
        last = f"{bc_xpath}/boundary_condition[last()]"
        project.add_element(last, "type", str(bc["type"]))
        project.add_element(last, "mesh", str(bc["mesh"]))
        project.add_element(last, "parameter", str(bc["parameter"]))


def _configure_pressure_boundary_mode(project, params: dict) -> None:
    """
    Configure pressure BCs by mode:
      - fixed_pressure (default): Dirichlet left/right (p_left, p_right)
      - fixed_flux: Neumann on left (q_left), Dirichlet on right (p_right)

    Notes:
      - For fixed_flux, provide params['inlet_specific_flux'].
      - In LiquidFlow with equation_balance_type="mass", units are kg/(m^2 s),
        and positive value corresponds to inflow of mass.
    """
    mode = str(params.get("bc_mode", "fixed_pressure")).strip().lower()

    if mode in {"fixed_pressure", "pressure", "dirichlet"}:
        if "inlet_pressure" in params:
            _update_pressure_param(project, "p_left", params["inlet_pressure"])
        if "outlet_pressure" in params:
            _update_pressure_param(project, "p_right", params["outlet_pressure"])

        _reset_pressure_boundary_conditions(project, [
            {"type": "Dirichlet", "mesh": "left",  "parameter": "p_left"},
            {"type": "Dirichlet", "mesh": "right", "parameter": "p_right"},
        ])
        return

    if mode in {"fixed_flux", "flux", "neumann"}:
        if "inlet_specific_flux" not in params:
            raise ValueError(
                "bc_mode='fixed_flux' requires 'inlet_specific_flux' in user_parameters."
            )

        _set_parameter_constant_value(project, "q_left", params["inlet_specific_flux"])

        # outlet pressure acts as pressure reference in fixed-flux mode
        if "outlet_pressure" in params:
            _update_pressure_param(project, "p_right", params["outlet_pressure"])

        _reset_pressure_boundary_conditions(project, [
            {"type": "Neumann",   "mesh": "left",  "parameter": "q_left"},
            {"type": "Dirichlet", "mesh": "right", "parameter": "p_right"},
        ])
        return

    raise ValueError(
        f"Unsupported bc_mode='{mode}'. Use 'fixed_pressure' or 'fixed_flux'."
    )


def _ensure_density_property(project) -> str:
    """
    Ensure:
      .//media/medium/phases/phase[type='AqueousLiquid']/properties/property[name='density']
    exists and return xpath to that <property>.
    """
    prop_xpath = ".//media/medium/phases/phase[type='AqueousLiquid']/properties/property[name='density']"
    if project.tree.find(prop_xpath) is None:
        # create <property> inside AqueousLiquid/properties
        parent = ".//media/medium/phases/phase[type='AqueousLiquid']/properties"
        project.add_element(parent, "property")
        last = f"{parent}/property[last()]"
        project.add_element(last, "name", "density")
    return prop_xpath

def _update_density_property(project, density_spec: dict) -> None:
    prop_xpath = _ensure_density_property(project)

    # wipe old content except <name>
    _clear_children(project, prop_xpath, [
        "type", "value", "reference_value", "independent_variable",
    ])

    dtype = density_spec.get("type", "Constant")

    if dtype == "Constant":
        project.add_element(prop_xpath, "type", "Constant")
        project.add_element(prop_xpath, "value", str(density_spec.get("value", "1000")))
    elif dtype == "Linear":
        project.add_element(prop_xpath, "type", "Linear")
        project.add_element(prop_xpath, "reference_value", str(density_spec.get("reference_value", "1000")))
        # independent_variable subtree
        project.add_element(prop_xpath, "independent_variable")
        iv = f"{prop_xpath}/independent_variable"
        project.add_element(iv, "variable_name", str(density_spec.get("variable_name", "liquid_phase_pressure")))
        project.add_element(iv, "reference_condition", str(density_spec.get("reference_condition", "1e5")))
        project.add_element(iv, "slope", str(density_spec.get("slope", "4.5e-7")))
    else:
        raise ValueError("fluid_density.type must be 'Constant' or 'Linear'")

def _set_equation_balance_type(project, value: str) -> None:
    """
    Ensure <equation_balance_type> exists under the first process and set it.
    value must be 'mass' or 'volume'.
    """
    proc_xpath = ".//processes/process[1]"
    node = project.tree.find(f"{proc_xpath}/equation_balance_type")

    if node is None:
        project.add_element(proc_xpath, "equation_balance_type")
        node = project.tree.find(f"{proc_xpath}/equation_balance_type")

    # node is an Element; set its text
    node.text = str(value)


def update_project_parameters(project: "ot.Project", params: dict) -> None:
    # simple direct text replacements
    update_map = {
        "prefix": ".//time_loop/output/prefix",
        "inlet_concentration_value": ".//parameters/parameter[name='c_bottom']/value",
        "porosity_value": ".//parameters/parameter[name='constant_porosity_parameter']/value",
        "decay_Si_value": ".//parameters/parameter[name='decay']/value",
        "t_end": ".//time_loop/processes/process/time_stepping/t_end",
        "initial_dt": ".//time_loop/processes/process/time_stepping/initial_dt",
        "minimum_dt": ".//time_loop/processes/process/time_stepping/minimum_dt",
        "maximum_dt": ".//time_loop/processes/process/time_stepping/maximum_dt",
        "specific_body_force": ".//processes/process/specific_body_force",
        "longitudinal_dispersivity_value": ".//media/medium/properties/property[name='longitudinal_dispersivity']/value",
        "transversal_dispersivity_value": ".//media/medium/properties/property[name='transversal_dispersivity']/value",
    }

    for key, xpath in update_map.items():
        if key in params:
            project.replace_text(str(params[key]), xpath=xpath)

    # pressure IC and pressure/flux BC mode
    if "initial_pressure" in params:
        _update_pressure_param(project, "p0", params["initial_pressure"])

    _configure_pressure_boundary_mode(project, params)

    # permeability (either constant scalar -> tensor values OR mesh field)
    has_val   = "permeability_value" in params
    has_field = "permeability_mesh_field" in params
    if has_val and has_field:
        raise ValueError("Specify only one of 'permeability_value' or 'permeability_mesh_field'.")

    if has_val:
        perm_val = float(params["permeability_value"])
        vals = f"{perm_val} 0 0\n0 {perm_val} 0\n0 0 {perm_val}"
        _set_parameter_constant_values(project, "kappa1_frac", vals)
    elif has_field:
        _set_parameter_mesh_element(project, "kappa1_frac", params["permeability_mesh_field"])

    # fracture thickness (constant OR mesh field)
    has_wval   = "fracture_thickness_value" in params
    has_wfield = "fracture_thickness_mesh_field" in params
    if has_wval and has_wfield:
        raise ValueError("Specify only one of 'fracture_thickness_value' or 'fracture_thickness_mesh_field'.")

    if has_wval:
        _set_parameter_constant_value(project, "fracture_thickness_const", params["fracture_thickness_value"])
    elif has_wfield:
        _set_parameter_mesh_element(project, "fracture_thickness_const", params["fracture_thickness_mesh_field"])

    # fluid density (create or update)
    if "fluid_density" in params:
        _update_density_property(project, params["fluid_density"])

    # equation_balance_type
    # - if user explicitly provides it: respect it
    # - otherwise: choose automatically from density model
    if "equation_balance_type" in params:
        _set_equation_balance_type(project, str(params["equation_balance_type"]))
    else:
        # auto rule: Constant density -> volume, Linear density -> mass
        dens = params.get("fluid_density", {})
        dens_type = dens.get("type", "Constant") if isinstance(dens, dict) else "Constant"
        _set_equation_balance_type(project, "mass" if dens_type == "Linear" else "volume")

    
    # numerical stabilization (remove then optionally add)
    main_proc_xpath = ".//processes/process[1]"
    project.remove_element(f"{main_proc_xpath}/numerical_stabilization")

    if "numerical_stabilization" in params:
        stab = params["numerical_stabilization"]
        stype = stab.get("type")
        if not stype:
            raise ValueError("'type' is required in numerical_stabilization")

        project.add_element(main_proc_xpath, "numerical_stabilization")
        ns = f"{main_proc_xpath}/numerical_stabilization[last()]"
        project.add_element(ns, "type", stype)

        if stype == "FullUpwind":
            project.add_element(ns, "cutoff_velocity", str(stab.get("cutoff_velocity", "0.0")))
        elif stype == "IsotropicDiffusion":
            if "tuning_parameter" not in stab or "cutoff_velocity" not in stab:
                raise ValueError("'tuning_parameter' and 'cutoff_velocity' required for IsotropicDiffusion")
            project.add_element(ns, "tuning_parameter", str(stab["tuning_parameter"]))
            project.add_element(ns, "cutoff_velocity", str(stab["cutoff_velocity"]))
        elif stype == "FluxCorrectedTransport":
            pass
        else:
            raise ValueError(f"Unsupported stabilization type: {stype}")


# %% cell 4 [markdown]
# ## run simulation helpers

# %% cell 5
def find_2d_vtu_for_case(jrc: float, sigma: float, root="_out") -> Path:
    root = Path(root)
    s_tag = str(float(sigma)).replace(".", "p")

    # folder is JRC_4.0, JRC_7.0, ...
    folder = root / jrc_dir_name(jrc)

    # filename is joint_JRC4_sigma_0p2MPa.vtu
    fname = f"joint_JRC{int(jrc)}_sigma_{s_tag}MPa.vtu"
    p = folder / fname

    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return p

def jrc_dir_name(jrc) -> str:
        return f"JRC_{float(jrc):.1f}"

def _run_cmd(cmd, cwd: Path):
    subprocess.run(cmd, cwd=str(cwd), check=True)

def _set_project_meshes(project: "ot.Project", mesh_names: list[str]) -> None:
    meshes_xpath = ".//meshes"
    if project.tree.find(meshes_xpath) is None:
        project.add_element(".//OpenGeoSysProject", "meshes")

    while project.tree.find(".//meshes/mesh") is not None:
        project.remove_element(".//meshes/mesh")

    for m in mesh_names:
        project.add_element(meshes_xpath, "mesh", str(m))

def _ensure_aperture_and_cubic_perm(
    mesh_path: Path,
    ap_name: str = "aperture_closed",
    perm_name: str = "permeability_cubic",
    w_min=None,
):
    mesh = pv.read(str(mesh_path))

    if ap_name not in mesh.cell_data:
        raise KeyError(f"Cell field '{ap_name}' not found in: {mesh_path}")

    w = np.asarray(mesh.cell_data[ap_name], float)
    if w_min is not None:
        w = np.maximum(w, float(w_min))

    k = (w**2) / 12.0
    mesh.cell_data[ap_name] = w
    mesh.cell_data[perm_name] = k

    if "MaterialIDs" not in mesh.cell_data:
        mesh.cell_data["MaterialIDs"] = np.zeros(mesh.n_cells, dtype=np.int32)

    mesh.save(str(mesh_path))

def _extract_and_split_boundaries_2d(mesh_dir: Path, mesh_name_vtu: str):
    mesh_dir = Path(mesh_dir)
    original_file = mesh_name_vtu
    boundary_file = "boundaries.vtu"

    mesh = pv.read(str(mesh_dir / original_file))
    x_min, x_max, y_min, y_max, *_ = mesh.bounds

    h = max((x_max - x_min), (y_max - y_min)) / max(np.sqrt(mesh.n_points), 1.0)
    tol = 0.75 * h

    _run_cmd(["NodeReordering", "-i", original_file, "-o", original_file, "-m", "1"], cwd=mesh_dir)
    _run_cmd(["ExtractBoundary", "-i", original_file, "-o", boundary_file], cwd=mesh_dir)

    bnd = pv.read(str(mesh_dir / boundary_file))
    cc = bnd.cell_centers().points
    bnd.cell_data["cx"] = cc[:, 0]
    bnd.cell_data["cy"] = cc[:, 1]

    def pick_by_center(surf_, scalar, target, tol_):
        return surf_.threshold([target - tol_, target + tol_], scalars=scalar)

    left   = pick_by_center(bnd, "cx", x_min, tol)
    right  = pick_by_center(bnd, "cx", x_max, tol)
    bottom = pick_by_center(bnd, "cy", y_min, tol)
    top    = pick_by_center(bnd, "cy", y_max, tol)

    left.save(str(mesh_dir / "left.vtu"))
    right.save(str(mesh_dir / "right.vtu"))
    bottom.save(str(mesh_dir / "bottom.vtu"))
    top.save(str(mesh_dir / "top.vtu"))

    parts = ["left.vtu", "right.vtu", "bottom.vtu", "top.vtu"]
    _run_cmd(["identifySubdomains", "-m", original_file, *parts], cwd=mesh_dir)
    _run_cmd(["checkMesh", "-v", "-p", original_file], cwd=mesh_dir)

def _run_one_ogs_case(
    mesh_dir: Path,
    prj_template: Path,
    out_dir: Path,
    user_parameters: dict,
    main_mesh_name: str,
    boundary_meshes: list[str],
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = ot.Project(
        input_file=Path(prj_template),
        output_file=out_dir / Path(prj_template).name.replace(".prj", "_final.prj"),
    )

    _set_project_meshes(project, [main_mesh_name, *boundary_meshes])
    update_project_parameters(project, user_parameters)
    project.write_input()
    project.run_model(args=f"-o {out_dir} -m {mesh_dir}", logfile=out_dir / "run.log")


def _jrc_dir_name(jrc) -> str:
    # matches your actual folders: JRC_4.0, JRC_7.0, JRC_10.0
    return f"JRC_{float(jrc):.1f}"

def _resolve_fracture_root_dir(fracture_root_dir: str | Path) -> Path:
    """
    Prefer requested fracture_root_dir, then try common legacy locations.
    """
    cands = [
        Path(fracture_root_dir),
        Path("_out_OGS/synthesis/results"),
        Path("_out/results"),
    ]
    seen = set()
    uniq = []
    for c in cands:
        cc = c.resolve() if c.is_absolute() else c
        key = str(cc)
        if key not in seen:
            uniq.append(c)
            seen.add(key)

    for c in uniq:
        if c.exists():
            return c

    raise FileNotFoundError(
        "No fracture source directory found. Checked: "
        + ", ".join(str(c) for c in uniq)
        + ". Run synthesis first or set fracture_root_dir."
    )


def workflow_run_jrc_sigma(
    jrc_list,
    sigmas_mpa,
    user_parameters: dict,
    fracture_root_dir: str = "_out_OGS/synthesis/results",
    fracture_mesh_pattern: str = "joint_JRC{jrc}_sigma_{s_tag}MPa.vtu",
    prj_template: str = "roughFracture_synthesis_LF.prj",
    base_run_dir: str = "_out_OGS/runs",
    boundary_meshes=None,
    w_min=1e-8,
):
    fracture_root_dir = _resolve_fracture_root_dir(fracture_root_dir)
    prj_template = Path(prj_template)
    base_run_dir = Path(base_run_dir)
    base_run_dir.mkdir(parents=True, exist_ok=True)

    if boundary_meshes is None:
        boundary_meshes = ["left.vtu", "right.vtu", "bottom.vtu", "top.vtu"]

    w_min_case = float(user_parameters.get("w_min", w_min))
    n_found = 0
    n_missing = 0

    for jrc in jrc_list:
        jrc_int = int(float(jrc))
        jrc_folder = _jrc_dir_name(jrc)

        for sigma in sigmas_mpa:
            s_tag = str(float(sigma)).replace(".", "p")

            src_mesh = fracture_root_dir / jrc_folder / fracture_mesh_pattern.format(
                jrc=jrc_int, s_tag=s_tag
            )

            if not src_mesh.exists():
                n_missing += 1
                print(f"skip JRC={jrc_int}, sigma={sigma}: missing {src_mesh}")
                continue
            n_found += 1

            case_dir = base_run_dir / f"JRC_{jrc_int}" / f"sigma_{s_tag}"
            mesh_dir = case_dir / "mesh"
            out_dir  = case_dir / "out"
            mesh_dir.mkdir(parents=True, exist_ok=True)
            out_dir.mkdir(parents=True, exist_ok=True)

            dst_mesh = mesh_dir / src_mesh.name
            shutil.copy2(src_mesh, dst_mesh)

            _ensure_aperture_and_cubic_perm(dst_mesh, w_min=w_min_case)
            _extract_and_split_boundaries_2d(mesh_dir, dst_mesh.name)

            params = dict(user_parameters)
            params["prefix"] = f"{params.get('prefix','roughFracture')}_JRC{jrc_int}_sigma_{s_tag}"
            params["permeability_mesh_field"] = "permeability_cubic"
            params["fracture_thickness_mesh_field"] = "aperture_closed"

            _run_one_ogs_case(
                mesh_dir=mesh_dir,
                prj_template=prj_template,
                out_dir=out_dir,
                user_parameters=params,
                main_mesh_name=dst_mesh.name,
                boundary_meshes=boundary_meshes,
            )

            print(f"✅ done JRC={jrc_int}, sigma={sigma} -> {case_dir}")

    if n_found == 0:
        raise RuntimeError(
            f"No source fracture meshes found under {fracture_root_dir}. "
            "Run synthesis first (write meshes), then rerun this workflow cell."
        )
    print(f"workflow summary: found={n_found}, missing={n_missing}, source={fracture_root_dir}")


# %% cell 6 [markdown]
# # run all simulations

# %% cell 7
def run_all_simulations():
    """Run OGS cases for all configured JRC and sigma values."""
    workflow_run_jrc_sigma(
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        user_parameters=user_parameters_rough,
        fracture_root_dir="_out_OGS/synthesis/results",
        fracture_mesh_pattern="joint_JRC{jrc}_sigma_{s_tag}MPa.vtu",
        prj_template=f"{user_parameters_rough['prefix']}.prj",
        base_run_dir="_out_OGS/runs",
    )

# %% cell 8 [markdown]
# # Post-processing

# %% cell 9
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
import ogstools as ot

# ============================================================
# PLOTTING PIPELINE (JRC × sigma)
#   - supports grids for: v_mag, pressure, aperture_closed, contact
#   - optional velocity arrows overlay on ANY background scalar
#   - works when aperture/contact exist ONLY in the INPUT .vtu mesh
# ============================================================

# -----------------------------
# SETTINGS
# -----------------------------
BASE_RUN_DIR = Path("_out_OGS/runs")
OUT_IMG_DIR  = Path("_out_OGS/plots_fields")
OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)

# recommended colormaps
CMAP_VMAG = "jet"
CMAP_P    = "turbo"
CMAP_AP   = "viridis"
CMAP_CT   = "gray"

opacity = 0.5

use_cell_centers = True
downsample_every = 3
arrow_scale_frac = 0.12
min_mag = 1e-14
arrow_geom = pv.Arrow(tip_length=0.35, tip_radius=0.14, shaft_radius=0.05)

# ---- streamline settings (used when use_streamlines=True) ----
use_streamlines    = True    # True → streamline tubes + direction arrows; False → legacy glyphs
n_stream_seeds     = 50      # seed points along inlet edge (x = x_min)
stream_use_tubes   = True    # False = draw thin polyline streamlines
stream_tube_r_frac = 0.001   # tube radius as fraction of mesh diagonal (~0.85 mm on 100×100 mm domain)
stream_tube_min_radius = 2.0e-5  # [m] lower bound so tube filter does not degenerate
stream_tube_n_sides = 8           # lower = lighter/faster
stream_line_width  = 1.5     # used when stream_use_tubes=False
boundary_strip_pct = 0.03    # exclude outer 3 % of domain for percentile clim
stream_clim_pct    = (2, 98) # percentile range for streamline v_mag color scale
n_stream_arrows    = 0        # direction arrows along streamlines (0 = off)
stream_arr_frac    = 0.012   # arrow size as fraction of mesh diagonal
streamline_tube_opacity  = 1.0
streamline_arrow_opacity = 1.0
streamline_prefer_local_clim = True  # clearer per-case streamlines over strict cross-case comparability

PRESSURE_CANDIDATES = ("pressure", "LiquidFlow_pressure", "p")

# candidates for input-mesh scalars (cell data often)
APERTURE_CANDIDATES = ("aperture_closed", "aperture", "w", "aperture_clamped")
CONTACT_CANDIDATES  = ("contact_closed", "is_contact", "contact_flag")


# -----------------------------
# Helpers
# -----------------------------
def mag(v):
    v = np.asarray(v)
    m = np.linalg.norm(v, axis=1)
    m[m <= 0] = min_mag
    return m


def _s_tag(sigma: float) -> str:
    return str(float(sigma)).replace(".", "p")

def _resolve_existing_dir(path_like: str | Path) -> Path:
    """
    Resolve a directory robustly across script vs notebook CWD differences.
    Priority:
      1) given path as-is
      2) cwd / path
      3) script_dir / path (if __file__ available)
    """
    p = Path(path_like)
    cands = [p]
    if not p.is_absolute():
        cands.append(Path.cwd() / p)
        try:
            cands.append(Path(__file__).resolve().parent / p)
        except NameError:
            pass

    seen = set()
    uniq = []
    for c in cands:
        k = str(c.resolve()) if c.exists() else str(c)
        if k not in seen:
            uniq.append(c)
            seen.add(k)

    for c in uniq:
        if c.exists():
            return c
    return p


def find_pvd_for_jrc_sigma(jrc: int, sigma: float, base_run_dir=BASE_RUN_DIR) -> Path:
    s_tag = _s_tag(sigma)
    out_dir = Path(base_run_dir) / f"JRC_{int(jrc)}" / f"sigma_{s_tag}" / "out"
    pvds = sorted(out_dir.glob("*.pvd"))
    if not pvds:
        raise FileNotFoundError(f"No .pvd found in: {out_dir}")
    return pvds[0]


def count_available_ogs_cases(jrc_list, sigmas_mpa, base_run_dir=BASE_RUN_DIR):
    n_ok = 0
    missing = []
    for jrc in jrc_list:
        for sigma in sigmas_mpa:
            try:
                _ = find_pvd_for_jrc_sigma(jrc, sigma, base_run_dir=base_run_dir)
                _ = find_input_vtu_for_case(jrc, sigma, base_run_dir=base_run_dir)
                n_ok += 1
            except FileNotFoundError as e:
                missing.append(str(e))
    return n_ok, missing


def find_input_vtu_for_case(jrc: int, sigma: float, base_run_dir=BASE_RUN_DIR) -> Path:
    """
    Expected run layout created by your workflow:
      _out_OGS/runs/JRC_{jrc}/sigma_{s_tag}/mesh/*.vtu   (main mesh)
    We pick the first .vtu that is NOT a boundary mesh.
    """
    s_tag = _s_tag(sigma)
    mesh_dir = Path(base_run_dir) / f"JRC_{int(jrc)}" / f"sigma_{s_tag}" / "mesh"
    vtus = sorted(mesh_dir.glob("*.vtu"))
    if not vtus:
        raise FileNotFoundError(f"No input .vtu found in: {mesh_dir}")
    # skip boundary meshes if present
    skip = {"left.vtu", "right.vtu", "top.vtu", "bottom.vtu", "boundaries.vtu"}
    for p in vtus:
        if p.name not in skip:
            return p
    return vtus[0]


def _find_first_name(candidates, names):
    for n in candidates:
        if n in names:
            return n
    return None


def _finite_vals(a):
    a = np.asarray(a, float).ravel()
    a = a[np.isfinite(a)]
    return a


def _finite_stats(arr: np.ndarray):
    a = np.asarray(arr, dtype=float).ravel()
    n = a.size
    n_nan = np.count_nonzero(~np.isfinite(a))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "nan_frac": 1.0, "n": n}
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "mean": float(np.mean(a)),
        "nan_frac": float(n_nan / max(n, 1)),
        "n": int(n),
    }


def _convert_cell_to_point_if_needed(mesh: pv.DataSet) -> pv.DataSet:
    # Some OGS outputs put scalars in cell data (pressure often). Convert for robust sampling.
    if hasattr(mesh, "cell_data") and len(mesh.cell_data.keys()) > 0:
        return mesh.cell_data_to_point_data(pass_cell_data=False)
    return mesh


def load_last_output_mesh(pvd_file: Path) -> pv.DataSet:
    series = ot.MeshSeries(str(pvd_file))
    mesh = series[len(series) - 1]  # last timestep
    return _convert_cell_to_point_if_needed(mesh)


def sample_output_on_surface(output_mesh: pv.DataSet, surf: pv.PolyData) -> pv.PolyData:
    """
    Sample output_mesh onto surf and copy ONLY arrays that match
    surf.n_points (point data) or surf.n_cells (cell data).
    This avoids copying field-data arrays (e.g., length = #timesteps).
    """
    sampled = output_mesh.sample(surf)
    out = surf.copy(deep=True)

    # point data (most important for plotting on a surface)
    for name in sampled.point_data.keys():
        arr = np.asarray(sampled.point_data[name])
        if arr.shape[0] == out.n_points:
            out.point_data[name] = arr

    # cell data (optional; keep if it matches)
    for name in sampled.cell_data.keys():
        arr = np.asarray(sampled.cell_data[name])
        if arr.shape[0] == out.n_cells:
            out.cell_data[name] = arr

    return out



def build_glyphs_from_output(output_mesh: pv.DataSet, surf: pv.PolyData):
    """Legacy arrow-glyph overlay (used when use_streamlines=False)."""
    if use_cell_centers:
        pts = surf.cell_centers()
    else:
        pts = pv.PolyData(surf.points)

    sampled = output_mesh.sample(pts)

    def _get_v(sd):
        for store in (sd.point_data, sd.cell_data):
            if "v" in store.keys():
                return np.asarray(store["v"])
        if "v" in sd.array_names:
            return np.asarray(sd["v"])
        return None

    v = _get_v(sampled)
    if v is None or v.shape[0] != pts.n_points:
        pts = pv.PolyData(surf.points)
        sampled = output_mesh.sample(pts)
        v = _get_v(sampled)
        if v is None or v.shape[0] != pts.n_points:
            return None

    pts = pts.copy(deep=True)
    pts.point_data["v"] = v
    pts.point_data["v_mag"] = mag(v)

    if downsample_every and downsample_every > 1 and pts.n_points > 0:
        keep = np.arange(pts.n_points)[::downsample_every]
        pts = pts.extract_points(keep, include_cells=False)

    diag  = float(output_mesh.length)
    vmax  = float(np.max(pts.point_data["v_mag"])) if pts.n_points else 1.0
    factor = (diag * arrow_scale_frac) / max(vmax, min_mag)
    return pts.glyph(orient="v", scale="v_mag", factor=factor, geom=arrow_geom)


def _interior_percentile_clim(v_mag_all: np.ndarray, points: np.ndarray, bounds) -> tuple:
    """Return (lo, hi) = percentile color limits from interior points only."""
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    sx, sy = dx * boundary_strip_pct, dy * boundary_strip_pct
    mask = (
        (points[:, 0] > bounds[0] + sx) & (points[:, 0] < bounds[1] - sx) &
        (points[:, 1] > bounds[2] + sy) & (points[:, 1] < bounds[3] - sy)
    )
    interior = v_mag_all[mask]
    base = interior if interior.size > 20 else v_mag_all
    return (float(np.percentile(base, stream_clim_pct[0])),
            float(np.percentile(base, stream_clim_pct[1])))


def build_streamlines_from_output(output_mesh: pv.DataSet):
    """
    Trace velocity streamlines across the fracture surface.
    - Converts UnstructuredGrid → PolyData (extract_surface) for surface integration
    - Seeds slightly inside the inlet edge (x = x_min + 1 %)
    - Uses surface_streamlines=True so integration stays on the 2-D fracture plane
    - Returns (stream_geom, dir_arrows, interior_clim, used_tubes)
    """
    # ── convert to PolyData for surface streamlines ───────────────────────────
    if isinstance(output_mesh, pv.PolyData):
        surf_mesh = output_mesh.copy(deep=False)
    else:
        surf_mesh = output_mesh.extract_surface()

    # ensure v in point_data
    if "v" not in surf_mesh.point_data.keys():
        if "v" in surf_mesh.cell_data.keys():
            surf_mesh = surf_mesh.cell_data_to_point_data()
        elif "v" in surf_mesh.array_names:
            surf_mesh.point_data["v"] = surf_mesh["v"]
        else:
            return None, None, None, False

    v_all  = np.asarray(surf_mesh.point_data["v"])
    vm_all = mag(v_all)
    surf_mesh.point_data["v_mag"] = vm_all

    bounds   = surf_mesh.bounds
    int_clim = _interior_percentile_clim(vm_all, np.asarray(surf_mesh.points), bounds)
    diag     = float(surf_mesh.length)
    tube_r   = max(diag * stream_tube_r_frac, float(stream_tube_min_radius))
    arr_size = diag * stream_arr_frac

    # ── seed points: slightly inside inlet (x = x_min + 1 %) ─────────────────
    dx     = bounds[1] - bounds[0]
    x_in   = bounds[0] + dx * 0.01
    y_seed = np.linspace(bounds[2], bounds[3], n_stream_seeds + 2)[1:-1]
    # z = 0 works for near-flat fractures; seeds are projected onto surface internally
    seed_pts = np.column_stack([np.full(len(y_seed), x_in),
                                y_seed,
                                np.zeros(len(y_seed))])
    source = pv.PolyData(seed_pts)

    # ── trace streamlines on surface ──────────────────────────────────────────
    try:
        sl = surf_mesh.streamlines_from_source(
            source,
            vectors="v",
            surface_streamlines=True,
            max_steps=5000,
            initial_step_length=diag * 0.004,
            terminal_speed=min_mag * 10,
            integration_direction="forward",
        )
    except Exception as exc:
        print(f"  [streamlines] failed: {exc}")
        return None, None, int_clim, False

    if sl is None or sl.n_points == 0:
        print("  [streamlines] returned 0 points")
        return None, None, int_clim, False

    # ── colour by v_mag ───────────────────────────────────────────────────────
    sl_v  = np.asarray(sl.point_data.get("v", np.zeros((sl.n_points, 3))))
    sl_vm = np.linalg.norm(sl_v, axis=1)
    sl_vm[sl_vm <= 0] = min_mag
    sl.point_data["v_mag"] = sl_vm

    # ── streamline geometry (tubes or plain lines) ───────────────────────────
    used_tubes = False
    if stream_use_tubes:
        try:
            tubes = sl.tube(radius=tube_r, n_sides=int(stream_tube_n_sides), capping=False)
            if tubes is None or tubes.n_points == 0:
                raise RuntimeError("tube() produced empty geometry")
            tree   = cKDTree(sl.points)
            _, idx = tree.query(tubes.points)
            tubes.point_data["v_mag"] = sl_vm[idx]
            used_tubes = True
        except Exception as exc:
            print(f"  [streamlines] tube conversion failed ({exc}); using polylines.")
            tubes = sl
    else:
        tubes = sl

    # ── periodic direction arrows ─────────────────────────────────────────────
    dir_arrows = None
    if n_stream_arrows > 0 and sl.n_points > 1:
        step  = max(1, sl.n_points // n_stream_arrows)
        keep  = np.arange(0, sl.n_points, step)
        a_pts = pv.PolyData(sl.points[keep])
        a_pts.point_data["v"]     = sl_v[keep]
        a_pts.point_data["v_mag"] = sl_vm[keep]
        try:
            dir_arrows = a_pts.glyph(
                orient="v", scale=False, factor=arr_size, geom=arrow_geom
            )
            _tree = cKDTree(sl.points)
            _, _i = _tree.query(dir_arrows.points)
            dir_arrows.point_data["v_mag"] = sl_vm[_i]
        except Exception:
            dir_arrows = None

    return tubes, dir_arrows, int_clim, used_tubes



def _clean_pressure(pvals):
    p = np.asarray(pvals, float)
    # remove bogus non-positive values sometimes introduced by sampling
    p[p <= 0.0] = np.nan
    return p


def render_case_to_png(
    jrc: int,
    sigma: float,
    png_file: Path,
    scalar_name: str,           # "v_mag" | "pressure" | "aperture_closed" | "contact_closed"
    clim,
    cmap_name: str,
    base_run_dir=BASE_RUN_DIR,
    window_size=(2600, 1800),
    scale=2,
    zoom_factor=1.0,
    show_axes=False,
    add_pyvista_title=False,
    title="",
    arrows_on: str | None = None,       # None | "pressure" | "aperture_closed" | "contact_closed" | "v_mag"
    arrow_cmap: str = CMAP_VMAG,         # arrows colored by v_mag
    arrow_clim=None,                     # pass global clim_v for consistent arrow colors
):
    """
    Renders ONE case to PNG with optional velocity arrows overlay.
    Background scalar is chosen by scalar_name.
    If scalar_name is aperture/contact, values are read from INPUT mesh (.vtu in case/mesh).
    If scalar_name is v_mag/pressure, values are read from OUTPUT mesh (.pvd last timestep).
    Arrows (if enabled) are always derived from OUTPUT velocity 'v' and colored by |v|.
    """
    jrc = int(jrc)
    sigma = float(sigma)

    pvd = find_pvd_for_jrc_sigma(jrc, sigma, base_run_dir=base_run_dir)
    out_mesh = load_last_output_mesh(pvd)

    # ---- background surface comes from INPUT mesh (stable geometry & has aperture/contact)
    in_vtu = find_input_vtu_for_case(jrc, sigma, base_run_dir=base_run_dir)
    in_mesh = pv.read(str(in_vtu))
    surf = in_mesh.extract_surface()

    want_arrows = arrows_on is not None and arrows_on.lower() != "off"

    # ---- get background scalar values on surf
    bg = surf.copy(deep=True)
    scalar_preference = "point"
    bg_opacity = float(opacity)
    if want_arrows:
        # Keep background slightly translucent so tubes/arrows remain readable.
        bg_opacity = min(bg_opacity, 0.40)

    if scalar_name in ("aperture_closed", "contact_closed"):
        # Use INPUT surface cell data directly (no interpolation) for robust, physical maps.
        # Interpolating contact to points turns binary 0/1 into fuzzy values and looks non-physical.
        scalar_preference = "cell"
        bg_opacity = 0.50 if (scalar_name == "contact_closed" and want_arrows) else (1.0 if scalar_name == "contact_closed" else bg_opacity)

        if scalar_name == "aperture_closed":
            name = _find_first_name(APERTURE_CANDIDATES, surf.cell_data.keys())
            if name is None:
                raise KeyError(f"No aperture field found in input surface {in_vtu}. Have: {list(surf.cell_data.keys())}")
            bg.cell_data["aperture_closed"] = np.asarray(surf.cell_data[name], float)

        else:
            name = _find_first_name(CONTACT_CANDIDATES, surf.cell_data.keys())
            if name is None:
                raise KeyError(f"No contact field found in input surface {in_vtu}. Have: {list(surf.cell_data.keys())}")
            bg.cell_data["contact_closed"] = (np.asarray(surf.cell_data[name], float) > 0.5).astype(float)

    elif scalar_name in ("pressure", "v_mag"):
        # sample from output mesh
        bg = sample_output_on_surface(out_mesh, bg)

        if scalar_name == "pressure":
            pname = _find_first_name(PRESSURE_CANDIDATES, bg.array_names)
            if pname is None:
                raise KeyError(f"No pressure found in output {pvd}. Have: {bg.array_names}")
            bg["pressure"] = _clean_pressure(bg[pname])

        else:  # v_mag
            if "v" not in bg.array_names:
                raise KeyError(f"No velocity 'v' found in output {pvd}. Have: {bg.array_names}")
            bg["v_mag"] = mag(bg["v"])

    else:
        raise ValueError(f"Unsupported scalar_name='{scalar_name}'")

    # ---- per-case interior clim for pressure (global clim is useless: ΔP spans
    #      orders of magnitude across cases — each case needs its own scale)
    if scalar_name == "pressure":
        p_vals = np.asarray(bg[scalar_name] if scalar_name in bg.array_names
                            else bg.point_data.get(scalar_name, []), float)
        if p_vals.size > 0:
            b_ = bg.bounds; pts_ = np.asarray(bg.points)
            sx_ = (b_[1]-b_[0]) * boundary_strip_pct
            sy_ = (b_[3]-b_[2]) * boundary_strip_pct
            mask_ = ((pts_[:,0]>b_[0]+sx_)&(pts_[:,0]<b_[1]-sx_)&
                     (pts_[:,1]>b_[2]+sy_)&(pts_[:,1]<b_[3]-sy_))
            p_int = p_vals[mask_]; p_int = p_int[np.isfinite(p_int)]
            if p_int.size > 10:
                clim = (float(np.nanpercentile(p_int, 2)),
                        float(np.nanpercentile(p_int, 98)))

    # ---- build flow overlay (streamlines or legacy glyphs)
    do_arrows  = want_arrows
    glyphs     = None   # legacy glyphs (when use_streamlines=False)
    sl_tubes   = None   # streamline tubes
    sl_arrows  = None   # direction arrows on streamlines
    sl_clim    = None   # interior percentile clim
    sl_used_tubes = False

    if do_arrows:
        if use_streamlines:
            sl_tubes, sl_arrows, sl_clim, sl_used_tubes = build_streamlines_from_output(out_mesh)
        else:
            glyphs = build_glyphs_from_output(out_mesh, bg)

    # ---- render
    p = pv.Plotter(off_screen=True, window_size=window_size)
    p.set_background("white")

    if add_pyvista_title and title:
        p.add_text(title, font_size=28, position="upper_left")

    p.add_mesh(
        bg,
        scalars=scalar_name,
        cmap=cmap_name,
        clim=clim,
        opacity=bg_opacity,
        preference=scalar_preference,
        show_edges=False,
        show_scalar_bar=False,
    )

    # legacy glyph arrows
    if glyphs is not None:
        p.add_mesh(
            glyphs,
            scalars="v_mag",
            cmap=arrow_cmap,
            clim=arrow_clim,
            lighting=False,
            opacity=1.0,
            show_scalar_bar=False,
        )

    # streamline tubes + direction arrows
    if streamline_prefer_local_clim and sl_clim is not None:
        _sl_clim = sl_clim
    else:
        _sl_clim = arrow_clim if arrow_clim is not None else sl_clim
    if sl_tubes is not None:
        sl_kwargs = dict(
            scalars="v_mag",
            cmap=arrow_cmap,
            clim=_sl_clim,
            opacity=streamline_tube_opacity,
            show_scalar_bar=False,
            lighting=False,
        )
        if not sl_used_tubes:
            sl_kwargs["render_lines_as_tubes"] = False
            sl_kwargs["line_width"] = float(stream_line_width)
        p.add_mesh(sl_tubes, **sl_kwargs)
    if sl_arrows is not None:
        p.add_mesh(
            sl_arrows,
            scalars="v_mag",
            cmap=arrow_cmap,
            clim=_sl_clim,
            lighting=False,
            opacity=streamline_arrow_opacity,
            show_scalar_bar=False,
        )

    p.enable_parallel_projection()
    p.reset_camera(bounds=bg.bounds)
    p.camera.zoom(float(zoom_factor))

    if show_axes:
        p.show_axes()

    p.screenshot(str(png_file), scale=scale)
    p.close()


def compute_global_clim(
    jrc_list,
    sigmas_mpa,
    scalar_name: str,
    base_run_dir=BASE_RUN_DIR,
):
    if scalar_name == "contact_closed":
        return (0.0, 1.0)

    vmins, vmaxs = [], []
    for jrc in jrc_list:
        for sigma in sigmas_mpa:
            try:
                pvd = find_pvd_for_jrc_sigma(jrc, sigma, base_run_dir=base_run_dir)
                out_mesh = load_last_output_mesh(pvd)

                in_vtu = find_input_vtu_for_case(jrc, sigma, base_run_dir=base_run_dir)
                in_mesh = pv.read(str(in_vtu))
                surf = in_mesh.extract_surface()

                if scalar_name in ("aperture_closed", "contact_closed"):
                    if scalar_name == "aperture_closed":
                        name = _find_first_name(APERTURE_CANDIDATES, surf.cell_data.keys())
                        if name is None:
                            raise KeyError(f"aperture missing in {in_vtu}")
                        vals = np.asarray(surf.cell_data[name], float)
                    else:
                        name = _find_first_name(CONTACT_CANDIDATES, surf.cell_data.keys())
                        if name is None:
                            raise KeyError(f"contact missing in {in_vtu}")
                        vals = (np.asarray(surf.cell_data[name], float) > 0.5).astype(float)

                elif scalar_name == "pressure":
                    sampled = out_mesh.sample(surf)
                    name = _find_first_name(PRESSURE_CANDIDATES, sampled.array_names)
                    if name is None:
                        raise KeyError(f"pressure missing in {pvd}")
                    p_all = np.asarray(_clean_pressure(sampled[name]), float)
                    # pool interior values: exclude boundary strip AND NaN
                    b_ = surf.bounds; pts_ = np.asarray(surf.points)
                    sx_ = (b_[1]-b_[0]) * boundary_strip_pct
                    sy_ = (b_[3]-b_[2]) * boundary_strip_pct
                    mask_ = (
                        (pts_[:,0] > b_[0]+sx_) & (pts_[:,0] < b_[1]-sx_) &
                        (pts_[:,1] > b_[2]+sy_) & (pts_[:,1] < b_[3]-sy_)
                    )
                    p_int = p_all[mask_]
                    p_int = p_int[np.isfinite(p_int)]
                    if p_int.size > 10:
                        vmins.append(float(np.nanpercentile(p_int, stream_clim_pct[0])))
                        vmaxs.append(float(np.nanpercentile(p_int, stream_clim_pct[1])))
                    continue   # skip generic min/max append

                elif scalar_name == "v_mag":
                    sampled = out_mesh.sample(surf)
                    if "v" not in sampled.array_names:
                        raise KeyError(f"v missing in {pvd}")
                    v_all = mag(sampled["v"])
                    # use interior percentile to avoid outlet/boundary spikes
                    lo, hi = _interior_percentile_clim(
                        v_all, np.asarray(surf.points), surf.bounds
                    )
                    vmins.append(lo)
                    vmaxs.append(hi)
                    continue   # skip the generic min/max append below

                else:
                    raise ValueError(scalar_name)

                vals = _finite_vals(vals)
                if vals.size == 0:
                    continue
                vmins.append(float(np.min(vals)))
                vmaxs.append(float(np.max(vals)))
            except FileNotFoundError:
                continue

    if not vmins:
        raise RuntimeError(
            f"No valid values found to compute clim for '{scalar_name}'. "
            f"No readable OGS outputs under {base_run_dir}. "
            "Run workflow_run_jrc_sigma first."
        )
    return (float(np.min(vmins)), float(np.max(vmaxs)))


def print_case_stats(jrc_list, sigmas_mpa, base_run_dir=BASE_RUN_DIR):
    print("\n=== PER-CASE STATS (last timestep) ===")
    for jrc in jrc_list:
        for sigma in sigmas_mpa:
            try:
                pvd = find_pvd_for_jrc_sigma(jrc, sigma, base_run_dir=base_run_dir)
                out_mesh = load_last_output_mesh(pvd)

                in_vtu = find_input_vtu_for_case(jrc, sigma, base_run_dir=base_run_dir)
                in_mesh = pv.read(str(in_vtu))
                surf = in_mesh.extract_surface()

                # pressure
                sp = out_mesh.sample(surf)
                pname = _find_first_name(PRESSURE_CANDIDATES, sp.array_names)
                p_stats = _finite_stats(_clean_pressure(sp[pname])) if pname else None

                # v_mag
                v_stats = _finite_stats(mag(sp["v"])) if "v" in sp.array_names else None

                # aperture
                aname = _find_first_name(APERTURE_CANDIDATES, surf.cell_data.keys())
                a_stats = _finite_stats(surf.cell_data[aname]) if aname else None

                # contact
                cname = _find_first_name(CONTACT_CANDIDATES, surf.cell_data.keys())
                c_stats = _finite_stats((np.asarray(surf.cell_data[cname], float) > 0.5).astype(float)) if cname else None

                def fmt(s, kind):
                    if s is None:
                        return "missing"
                    if kind == "pressure":
                        return f"min={s['min']:.2f}, max={s['max']:.2f}, mean={s['mean']:.2f}"
                    return f"min={s['min']:.6g}, max={s['max']:.6g}, mean={s['mean']:.6g}"

                print(
                    f"JRC={int(jrc)}, σn={float(sigma):g} MPa | "
                    f"p: {fmt(p_stats,'pressure')} | "
                    f"|v|: {fmt(v_stats,'v')} | "
                    f"w: {fmt(a_stats,'w')} | "
                    f"contact: {fmt(c_stats,'c')}"
                )

            except FileNotFoundError as e:
                print(f"JRC={int(jrc)}, σn={float(sigma):g} MPa | missing: {e}")


def render_all_jrc_sigma(
    jrc_list,
    sigmas_mpa,
    scalar_name: str,
    cmap_name: str,
    base_run_dir=BASE_RUN_DIR,
    out_img_dir=OUT_IMG_DIR,
    window_size=(2600, 1800),
    scale=2,
    zoom_factor=1.0,
    add_pyvista_title=False,
    arrows_on: str | None = None,     # None/off to disable; else overlay arrows
    arrow_clim=None,
):
    out_dir = Path(out_img_dir) / scalar_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # global clim for the background scalar
    clim = compute_global_clim(jrc_list, sigmas_mpa, scalar_name, base_run_dir=base_run_dir)
    print(f"Global color scale for {scalar_name}: {clim}")

    png_map = {}
    for jrc in jrc_list:
        for sigma in sigmas_mpa:
            try:
                # verify case exists
                _ = find_pvd_for_jrc_sigma(jrc, sigma, base_run_dir=base_run_dir)
                _ = find_input_vtu_for_case(jrc, sigma, base_run_dir=base_run_dir)
            except FileNotFoundError as e:
                print(f"skip JRC={int(jrc)}, sigma={sigma}: {e}")
                png_map[(int(jrc), float(sigma))] = None
                continue

            s_tag = _s_tag(sigma)
            png_file = out_dir / f"JRC{int(jrc)}_sigma_{s_tag}__{scalar_name}.png"
            title = rf"JRC={int(jrc)} | $\sigma_n={float(sigma):g}\ \mathrm{{MPa}}$"

            try:
                render_case_to_png(
                    jrc=int(jrc),
                    sigma=float(sigma),
                    png_file=png_file,
                    scalar_name=scalar_name,
                    clim=clim,
                    cmap_name=cmap_name,
                    base_run_dir=base_run_dir,
                    window_size=window_size,
                    scale=scale,
                    zoom_factor=zoom_factor,
                    add_pyvista_title=add_pyvista_title,
                    title=title,
                    arrows_on=arrows_on,
                    arrow_clim=arrow_clim,
                )
                png_map[(int(jrc), float(sigma))] = png_file
                print("saved:", png_file)
            except Exception as _e:
                print(f"  [render] FAILED JRC={int(jrc)}, sigma={sigma}: {_e}")
                png_map[(int(jrc), float(sigma))] = None

    # ensure all keys exist
    for jrc in jrc_list:
        for sigma in sigmas_mpa:
            png_map.setdefault((int(jrc), float(sigma)), None)

    return png_map, clim


def plot_grid_with_shared_colorbar(
    png_map,
    jrc_list,
    sigmas_mpa,
    clim,
    cmap_name="jet",
    figsize=None,
    cbar_label="",
    title_fontsize=18,
    cbar_fontsize=18,
    cbar_ticksize=14,
    cbar_height=0.10,
    hspace=0.02,
    wspace=0.02,
    scalar_title="",
):
    nrows = len(jrc_list)
    ncols = len(sigmas_mpa)

    if figsize is None:
        figsize = (6 * ncols, 4.8 * nrows + 1.0)

    fig = plt.figure(figsize=figsize)

    height_ratios = [1.0] * nrows + [cbar_height]
    gs = fig.add_gridspec(
        nrows=nrows + 1,
        ncols=ncols,
        height_ratios=height_ratios,
        hspace=hspace,
        wspace=wspace,
    )

    for i, jrc in enumerate(jrc_list):
        for j, sigma in enumerate(sigmas_mpa):
            ax = fig.add_subplot(gs[i, j])
            key = (int(jrc), float(sigma))
            png = png_map.get(key, None)

            ax.set_title(
                rf"JRC={int(jrc)} | $\sigma_n={float(sigma):g}\ \mathrm{{MPa}}$",
                fontsize=title_fontsize
            )

            if png is None:
                ax.text(0.5, 0.5, f"(missing {scalar_title})", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            ax.imshow(plt.imread(png))
            ax.axis("off")

    cax = fig.add_subplot(gs[-1, :])
    norm = mpl.colors.Normalize(vmin=float(clim[0]), vmax=float(clim[1]))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.get_cmap(cmap_name))
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=cbar_fontsize)
    cbar.ax.tick_params(labelsize=cbar_ticksize)
    cax.yaxis.set_visible(False)
    return fig


# ============================================================
# RUN (EXAMPLE)
# ============================================================
# Define your lists before running:
# JRC_LIST = [4, 7, 10]
# sigmas_mpa = [0.2, 2.0, 10.0]
def run_all_plots(base_run_dir: str | Path | None = None, out_img_dir: str | Path | None = None):
    """Render all field grids from existing OGS outputs."""
    base_run_dir = _resolve_existing_dir(BASE_RUN_DIR if base_run_dir is None else base_run_dir)
    out_img_dir = Path(OUT_IMG_DIR if out_img_dir is None else out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    n_ok_cases, missing_cases = count_available_ogs_cases(JRC_LIST, sigmas_mpa, base_run_dir=base_run_dir)
    if n_ok_cases == 0:
        msg = "\n".join(missing_cases[:5])
        raise RuntimeError(
            f"No OGS cases found under {base_run_dir}. "
            "Run workflow_run_jrc_sigma first. Example missing entries:\n" + msg
        )
    print(f"Found {n_ok_cases} OGS cases under {base_run_dir}")

    print_case_stats(JRC_LIST, sigmas_mpa, base_run_dir=base_run_dir)

    # global clim for arrows (velocity magnitude), so arrows stay consistent across backgrounds
    clim_v = compute_global_clim(JRC_LIST, sigmas_mpa, "v_mag", base_run_dir=base_run_dir)

    # -----------------------------
    # 1) v_mag grid (with arrows ON/OFF)
    # -----------------------------
    png_map_v, clim_bg_v = render_all_jrc_sigma(
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        scalar_name="v_mag",
        cmap_name=CMAP_VMAG,
        base_run_dir=base_run_dir,
        out_img_dir=out_img_dir,
        window_size=(2800, 2000),
        scale=2,
        zoom_factor=1.4,
        add_pyvista_title=False,
        arrows_on=None,          # set to something non-None to overlay arrows
        arrow_clim=clim_v,
    )

    fig_v = plot_grid_with_shared_colorbar(
        png_map=png_map_v,
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        clim=clim_bg_v,
        cmap_name=CMAP_VMAG,
        figsize=(6 * len(sigmas_mpa), 5 * len(JRC_LIST)),
        cbar_label=r"Darcy velocity $|\mathbf{v}|$ [m/s]",
        cbar_height=0.04,
        scalar_title="v_mag",
    )
    fig_v.savefig(out_img_dir / "GRID_v_mag.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # 2) pressure grid (OPTIONAL arrows over pressure)
    # -----------------------------
    png_map_p, clim_p = render_all_jrc_sigma(
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        scalar_name="pressure",
        cmap_name=CMAP_P,
        base_run_dir=base_run_dir,
        out_img_dir=out_img_dir,
        window_size=(2800, 2000),
        scale=2,
        zoom_factor=1.4,
        add_pyvista_title=False,
        arrows_on="pressure",    # "pressure" OR None to disable
        arrow_clim=clim_v,
    )

    fig_p = plot_grid_with_shared_colorbar(
        png_map=png_map_p,
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        clim=clim_p,
        cmap_name=CMAP_P,
        figsize=(6 * len(sigmas_mpa), 5 * len(JRC_LIST)),
        cbar_label=r"Pressure $p$ [Pa]",
        cbar_height=0.04,
        scalar_title="pressure",
    )
    fig_p.savefig(out_img_dir / "GRID_pressure.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # 3) aperture_closed grid (arrows ON/OFF)
    # -----------------------------
    png_map_w, clim_w = render_all_jrc_sigma(
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        scalar_name="aperture_closed",
        cmap_name=CMAP_AP,
        base_run_dir=base_run_dir,
        out_img_dir=out_img_dir,
        window_size=(2800, 2000),
        scale=2,
        zoom_factor=1.4,
        add_pyvista_title=False,
        arrows_on="aperture_closed",  # overlay velocity arrows on aperture map
        arrow_clim=clim_v,
    )

    fig_w = plot_grid_with_shared_colorbar(
        png_map=png_map_w,
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        clim=clim_w,
        cmap_name=CMAP_AP,
        figsize=(6 * len(sigmas_mpa), 5 * len(JRC_LIST)),
        cbar_label=r"Aperture $w$ [m]",
        cbar_height=0.04,
        scalar_title="aperture_closed",
    )
    fig_w.savefig(out_img_dir / "GRID_aperture_closed.png", dpi=300, bbox_inches="tight")
    plt.show()

    # -----------------------------
    # 4) contact grid (arrows ON/OFF)
    # -----------------------------
    png_map_c, clim_c = render_all_jrc_sigma(
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        scalar_name="contact_closed",
        cmap_name=CMAP_CT,
        base_run_dir=base_run_dir,
        out_img_dir=out_img_dir,
        window_size=(2800, 2000),
        scale=2,
        zoom_factor=1.4,
        add_pyvista_title=False,
        arrows_on="contact_closed",  # overlay velocity arrows on contact map
        arrow_clim=clim_v,
    )

    fig_c = plot_grid_with_shared_colorbar(
        png_map=png_map_c,
        jrc_list=JRC_LIST,
        sigmas_mpa=sigmas_mpa,
        clim=clim_c,
        cmap_name=CMAP_CT,
        figsize=(6 * len(sigmas_mpa), 5 * len(JRC_LIST)),
        cbar_label=r"Contact indicator $I_c$ [-]",
        cbar_height=0.04,
        scalar_title="contact_closed",
    )
    fig_c.savefig(out_img_dir / "GRID_contact.png", dpi=300, bbox_inches="tight")
    plt.show()


def _running_in_ipykernel() -> bool:
    """True when executed from Jupyter/VSCode interactive kernel."""
    return ("ipykernel" in sys.modules) or ("JPY_PARENT_PID" in os.environ)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run OGS rough-fracture simulations and/or plotting."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run OGS simulations (JRC x sigma cases).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Run post-processing plots from existing OGS outputs.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run mesh/field verification report on existing OGS outputs.",
    )
    parser.add_argument(
        "--stream-mode",
        choices=("lines", "tubes"),
        default=None,
        help="Override streamline rendering mode for plotting only.",
    )
    # parse_known_args keeps script robust if extra args are injected by wrappers
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown args: {unknown}")

    if not args.run and not args.plot and not args.verify:
        print("No action selected.")
        print("  --run    : run OGS simulations (JRC × sigma)")
        print("  --plot   : generate field plots from existing OGS outputs")
        print("  --verify : verify mesh/field outputs")
        print("\nExample:  python OGSrun_synthesis_roughFracture.py --plot")
        return

    if args.run:
        run_all_simulations()
    if args.plot:
        if args.stream_mode is not None:
            global stream_use_tubes
            stream_use_tubes = (args.stream_mode == "tubes")
            print(f"Using stream mode: {'tubes' if stream_use_tubes else 'lines'}")
        run_all_plots()
    if args.verify:
        run_verify()


if __name__ == "__main__":
    if _running_in_ipykernel():
        print("Interactive kernel detected. main() not auto-run.")
        print("Run cell-by-cell and call run_all_simulations() / run_all_plots() explicitly.")
    else:
        main()

# %% cell 9b [markdown]
# # Notebook-style execution cells

# %% cell 9c
# Run simulations (optional):
# run_all_simulations()

# %% cell 9d
# Plot only (requires existing _out_OGS/runs):
# stream_use_tubes = False  # or True
# run_all_plots()

# %% cell 9e
# Verify only:
# run_verify()

# %% cell 10
# # Use different colormaps per variable (recommended)
# CMAP_VMAG = "jet"
# CMAP_P    = "turbo"

# opacity = 0.5

# use_cell_centers = True
# downsample_every = 2
# arrow_scale_frac = 0.04
# min_mag = 1e-14
# arrow_geom = pv.Arrow(tip_length=0.3, tip_radius=0.08, shaft_radius=0.02)

# PRESSURE_CANDIDATES = ("pressure", "LiquidFlow_pressure", "p")

# # ============================================================
# # RUN
# # ============================================================

# print_case_stats(JRC_LIST, sigmas_mpa, base_run_dir=BASE_RUN_DIR)

# # 1) Darcy velocity magnitude grid
# png_map_v, clim_v = render_all_jrc_sigma(
#     jrc_list=JRC_LIST,
#     sigmas_mpa=sigmas_mpa,
#     scalar_name="v_mag",
#     cmap_name=CMAP_VMAG,
#     base_run_dir=BASE_RUN_DIR,
#     out_img_dir=OUT_IMG_DIR,
#     window_size=(2800, 2000),
#     scale=2,
#     zoom_factor=1.4,
#     add_pyvista_title=False,
# )

# fig_v = plot_grid_with_shared_colorbar(
#     png_map=png_map_v,
#     jrc_list=JRC_LIST,
#     sigmas_mpa=sigmas_mpa,
#     clim=clim_v,
#     cmap_name=CMAP_VMAG,
#     figsize=(6 * len(sigmas_mpa), 5 * len(JRC_LIST)),
#     cbar_label=r"Darcy velocity $|\mathbf{v}|$ [m/s]",
#     cbar_height=0.04,
#     scalar_title="v_mag",
# )
# fig_v.savefig(OUT_IMG_DIR / "GRID_v_mag.png", dpi=300, bbox_inches="tight")
# plt.show()

# # 2) Pressure grid
# png_map_p, clim_p = render_all_jrc_sigma(
#     jrc_list=JRC_LIST,
#     sigmas_mpa=sigmas_mpa,
#     scalar_name="pressure",
#     cmap_name=CMAP_P,
#     base_run_dir=BASE_RUN_DIR,
#     out_img_dir=OUT_IMG_DIR,
#     window_size=(2800, 2000),
#     scale=2,
#     zoom_factor=1.4,
#     add_pyvista_title=False,
# )

# fig_p = plot_grid_with_shared_colorbar(
#     png_map=png_map_p,
#     jrc_list=JRC_LIST,
#     sigmas_mpa=sigmas_mpa,
#     clim=clim_p,
#     cmap_name=CMAP_P,
#     figsize=(6 * len(sigmas_mpa), 5 * len(JRC_LIST)),
#     cbar_label=r"Pressure $p$ [Pa]",
#     cbar_height=0.04,
#     scalar_title="pressure",
# )
# fig_p.savefig(OUT_IMG_DIR / "GRID_pressure.png", dpi=300, bbox_inches="tight")
# plt.show()

# %% cell 11
# -----------------------------
# Utilities
# -----------------------------
def _safe_norm(arr) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected vector array shape (N,3), got {arr.shape}")
    return np.linalg.norm(arr, axis=1)

def _pick_first_existing_field(mesh, candidates, where="cell_data"):
    data = getattr(mesh, where)
    for name in candidates:
        if name in data:
            return name
    return None

def _fmt3(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "   -    "
    return f"{x:.3e}"

def _fmtf(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  -  "
    return f"{x:.{nd}f}"

def _tag_sigma_dirname(dir_name: str) -> str:
    # "sigma_0p2" -> "0.2"
    if dir_name.startswith("sigma_"):
        s = dir_name.replace("sigma_", "").replace("p", ".")
        return s
    return dir_name

def _parse_case_from_pvd(pvd: Path):
    """
    Supports both layouts:
      A) _out_OGS/runs/sigma_0p2/out/*.pvd
      B) _out_OGS/runs/JRC_7/sigma_0p2/out/*.pvd
    """
    out_dir = pvd.parent          # .../out
    sigma_dir = out_dir.parent    # .../sigma_0p2
    sigma_tag = sigma_dir.name
    sigma_str = _tag_sigma_dirname(sigma_tag)

    jrc = None
    if sigma_dir.parent.name.startswith("JRC_"):
        # .../JRC_7/sigma_0p2/out
        try:
            jrc = float(sigma_dir.parent.name.replace("JRC_", ""))
        except Exception:
            jrc = None

    # pretty label for printing
    case_label = []
    if jrc is not None:
        case_label.append(f"JRC={jrc:g}")
    case_label.append(f"σn={sigma_str} MPa")
    case_label.append(pvd.stem)
    return jrc, sigma_str, " | ".join(case_label)

# -----------------------------
# Main verifier
# -----------------------------
def verify_all_cases(
    base_run_dir="_out_OGS/runs",
    aperture_candidates=("aperture_closed", "aperture_mid", "aperture_raw"),
    contact_field="contact_closed",
    vec_field="v",
    max_print=9999,          # limit printed cases
    return_dataframe=True,   # also return a DataFrame for sorting/filtering
):
    base = Path(base_run_dir)

    # Find pvds in both layouts
    pvds = sorted(base.rglob("out/*.pvd"))
    if not pvds:
        raise FileNotFoundError(f"No .pvd found under: {base}/**/out/*.pvd")

    print(f"Found {len(pvds)} case(s) under {base}")

    rows = []
    printed = 0

    for pvd in pvds:
        jrc, sigma_str, case_title = _parse_case_from_pvd(pvd)

        try:
            series = ot.MeshSeries(str(pvd))
            mesh = series[len(series) - 1]  # last timestep

            # ---- v on points
            has_v_pt = vec_field in mesh.point_data
            v_pt_min = v_pt_mean = v_pt_max = np.nan
            if has_v_pt:
                vmag_pt = _safe_norm(mesh.point_data[vec_field])
                v_pt_min, v_pt_mean, v_pt_max = map(float, (vmag_pt.min(), vmag_pt.mean(), vmag_pt.max()))

            # ---- v sampled at cell centers (better for contact/open splits)
            cent = mesh.cell_centers()
            sampled = cent.sample(mesh)
            has_v_cell = vec_field in sampled.point_data  # sample result stores in point_data
            v_cell_min = v_cell_mean = v_cell_max = np.nan
            v_contact_min = v_contact_mean = v_contact_max = np.nan
            v_open_min = v_open_mean = v_open_max = np.nan

            if has_v_cell:
                v_cell = np.asarray(sampled.point_data[vec_field])
                vmag_cell = _safe_norm(v_cell)
                v_cell_min, v_cell_mean, v_cell_max = map(float, (vmag_cell.min(), vmag_cell.mean(), vmag_cell.max()))

            # ---- contact + aperture (cell data)
            has_contact = contact_field in mesh.cell_data
            contact_frac = np.nan
            contact = None
            if has_contact:
                contact = np.asarray(mesh.cell_data[contact_field]).astype(bool)
                contact_frac = float(contact.mean())

            ap_field = _pick_first_existing_field(mesh, aperture_candidates, where="cell_data")
            has_ap = ap_field is not None

            w_min = w_mean = w_max = np.nan
            w_zero = w_neg = np.nan
            w_contact_min = w_contact_max = np.nan
            w_open_min = w_open_max = np.nan


            if has_ap:
                w = np.asarray(mesh.cell_data[ap_field], float)
            
                # ---- derive contact from aperture (instead of reading contact_field)
                eps = 1e-15  # meters
                contact = (w <= eps)
                has_contact = True
                contact_frac = float(contact.mean())
            
                w_min, w_mean, w_max = map(float, (w.min(), w.mean(), w.max()))
                w_zero = int(np.count_nonzero(w == 0.0))
                w_neg  = int(np.count_nonzero(w < 0.0))
            
                if np.any(contact):
                    w_contact_min = float(w[contact].min())
                    w_contact_max = float(w[contact].max())
                if np.any(~contact):
                    w_open_min = float(w[~contact].min())
                    w_open_max = float(w[~contact].max())
            

            # ---- v(contact/open) using cell-center vmag and contact mask (must align with cells)
            mismatch = None
            if has_contact and has_v_cell and (contact is not None):
                # vmag_cell is per-cell-center (same count as n_cells in typical cases)
                if contact.size == vmag_cell.size:
                    if np.any(contact):
                        v_contact_min = float(vmag_cell[contact].min())
                        v_contact_mean = float(vmag_cell[contact].mean())
                        v_contact_max = float(vmag_cell[contact].max())
                    if np.any(~contact):
                        v_open_min = float(vmag_cell[~contact].min())
                        v_open_mean = float(vmag_cell[~contact].mean())
                        v_open_max = float(vmag_cell[~contact].max())
                else:
                    mismatch = f"contact({contact.size}) != v_cell({vmag_cell.size})"


            row = {
                "pvd": str(pvd),
                "jrc": jrc,
                "sigma_mpa_str": sigma_str,
                "case_title": case_title,

                "has_v_pt": has_v_pt,
                "v_pt_min": v_pt_min, "v_pt_mean": v_pt_mean, "v_pt_max": v_pt_max,

                "has_v_cell": has_v_cell,
                "v_cell_min": v_cell_min, "v_cell_mean": v_cell_mean, "v_cell_max": v_cell_max,

                "has_contact": has_contact,
                "contact_frac": contact_frac,
                "contact_vs_vcell_mismatch": mismatch,

                "aperture_field": ap_field,
                "has_aperture": has_ap,
                "w_min": w_min, "w_mean": w_mean, "w_max": w_max,
                "w_zero": w_zero, "w_neg": w_neg,
                "w_contact_min": w_contact_min, "w_contact_max": w_contact_max,
                "w_open_min": w_open_min, "w_open_max": w_open_max,

                "v_contact_min": v_contact_min, "v_contact_mean": v_contact_mean, "v_contact_max": v_contact_max,
                "v_open_min": v_open_min, "v_open_mean": v_open_mean, "v_open_max": v_open_max,
            }
            rows.append(row)

            # ---- pretty console output
            if printed < max_print:
                printed += 1
                print("\n" + "=" * 80)
                print(case_title)
                print("-" * 80)

                print(f"|v| points : "
                      f"min/mean/max = {_fmt3(v_pt_min)} / {_fmt3(v_pt_mean)} / {_fmt3(v_pt_max)}"
                      f"   ({'OK' if has_v_pt else 'MISSING'})")

                print(f"|v| cells  : "
                      f"min/mean/max = {_fmt3(v_cell_min)} / {_fmt3(v_cell_mean)} / {_fmt3(v_cell_max)}"
                      f"   ({'OK' if has_v_cell else 'MISSING'})")

                if has_contact:
                    msg = f"contact  : fraction = {_fmtf(contact_frac,3)}"
                    if mismatch:
                        msg += f"   ⚠ {mismatch}"
                    print(msg)
                else:
                    print("contact  : MISSING")

                if has_ap:
                    print(f"aperture  : field = '{ap_field}'")
                    print(f"           min/mean/max = {_fmt3(w_min)} / {_fmt3(w_mean)} / {_fmt3(w_max)}")
                    print(f"           zeros={w_zero}  negatives={w_neg}")
                    if has_contact and contact is not None:
                        if not (isinstance(w_contact_min, float) and np.isnan(w_contact_min)):
                            print(f"           w(contact) min/max = {_fmt3(w_contact_min)} / {_fmt3(w_contact_max)}")
                        if not (isinstance(w_open_min, float) and np.isnan(w_open_min)):
                            print(f"           w(open)    min/max = {_fmt3(w_open_min)} / {_fmt3(w_open_max)}")
                else:
                    print(f"aperture  : MISSING (tried {list(aperture_candidates)})")

                if has_contact and has_v_cell and contact is not None and mismatch is None:
                    if not (isinstance(v_contact_min, float) and np.isnan(v_contact_min)):
                        print(f"|v| contact: min/mean/max = {_fmt3(v_contact_min)} / {_fmt3(v_contact_mean)} / {_fmt3(v_contact_max)}")
                    if not (isinstance(v_open_min, float) and np.isnan(v_open_min)):
                        print(f"|v| open   : min/mean/max = {_fmt3(v_open_min)} / {_fmt3(v_open_mean)} / {_fmt3(v_open_max)}")

        except Exception as e:
            print("\n" + "=" * 80)
            print(case_title)
            print("ERROR:", e)
            rows.append({
                "pvd": str(pvd),
                "jrc": jrc,
                "sigma_mpa_str": sigma_str,
                "case_title": case_title,
                "error": str(e),
            })

    if not return_dataframe:
        return rows

    if pd is None:
        print("pandas not available; returning list of dict rows instead of DataFrame.")
        return rows

    df = pd.DataFrame(rows)

    # nice default ordering if present
    sort_cols = [c for c in ["jrc", "sigma_mpa_str", "case_title"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    # quick summary
    ok = df["error"].isna() if "error" in df.columns else np.ones(len(df), dtype=bool)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"  total cases        : {len(df)}")
    print(f"  ok                 : {int(ok.sum())}")
    print(f"  errors             : {int((~ok).sum())}")
    if "aperture_field" in df.columns:
        print("  aperture fields    :")
        print(df.loc[ok, "aperture_field"].value_counts(dropna=False).to_string())

    return df


# -----------------------------
# Optional verify helper (manual use)
# -----------------------------
def run_verify():
    return verify_all_cases(
        base_run_dir="_out_OGS/runs",
        aperture_candidates=("aperture_closed", "aperture_mid", "aperture_raw"),
        contact_field="contact_closed",
        max_print=9999,
        return_dataframe=True,
    )


# %% cell 12
# In Jupyter / VS Code interactive mode, call one of:
run_all_simulations()
run_all_plots()
run_verify()
