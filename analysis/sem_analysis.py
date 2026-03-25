"""
Structural Equation Model (SEM) Analysis
=========================================

Implements the full SEM based on the conceptual model:

    IV  (Firm Active Engagement)
        → M  (Structural Capital, Relational Capital, Cognitive Capital, Trust)
            → DV (Community Engagement)

    with moderators: Social vs Functional Needs, Need for Self-Esteem Enhancement

Uses ``semopy`` for maximum-likelihood estimation and provides:
- Measurement model (CFA) specification for latent constructs
- Structural paths between latent variables
- Standardised and unstandardised path coefficients
- Model fit indices (CFI, TLI, RMSEA, SRMR, etc.)
- Moderation via multi-group SEM at different moderator levels
"""

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import semopy

from config.constructs import (
    ALL_CONSTRUCTS,
    DEPENDENT_VARIABLES,
    INDEPENDENT_VARIABLES,
    MEDIATOR_VARIABLES,
    MODERATOR_VARIABLES,
    get_construct_label,
)

OUTPUT_DIR = pathlib.Path("output")


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _section(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Model specification helpers
# ---------------------------------------------------------------------------

def _build_measurement_model():
    """Build the measurement (CFA) portion of the model specification.

    Returns the lavaan-style syntax lines that map each latent construct
    to its observed indicator items.
    """
    lines = []
    for key, meta in ALL_CONSTRUCTS.items():
        indicators = " + ".join(meta["items"])
        lines.append(f"{key} =~ {indicators}")
    return lines


def _build_structural_model():
    """Build the structural (regression) portion of the model specification.

    Paths based on the conceptual model:
        Firm Active Engagement  → each mediator  (a paths)
        Each mediator           → Community Engagement  (b paths)
        Firm Active Engagement  → Community Engagement  (c' direct path)
    """
    iv_keys = list(INDEPENDENT_VARIABLES.keys())
    med_keys = list(MEDIATOR_VARIABLES.keys())
    dv_keys = list(DEPENDENT_VARIABLES.keys())

    lines = []
    for iv in iv_keys:
        # a paths: IV → each Mediator
        for m in med_keys:
            lines.append(f"{m} ~ {iv}")

    for dv in dv_keys:
        # b paths: each Mediator → DV, plus c' direct path: IV → DV
        predictors = " + ".join(med_keys + iv_keys)
        lines.append(f"{dv} ~ {predictors}")

    return lines


def build_sem_specification():
    """Return the complete SEM specification in lavaan-style syntax.

    Combines the measurement model (latent =~ indicators) and the
    structural model (regressions between latent variables).
    """
    measurement = _build_measurement_model()
    structural = _build_structural_model()
    return "\n".join(measurement + structural)


# ---------------------------------------------------------------------------
# Core SEM estimation
# ---------------------------------------------------------------------------

def fit_sem(df, model_spec=None):
    """Fit the structural equation model to the data.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing all observed indicator columns.
    model_spec : str, optional
        Custom model specification.  When *None* the default conceptual
        model is used.

    Returns
    -------
    semopy.Model
        The fitted model object.
    """
    if model_spec is None:
        model_spec = build_sem_specification()

    model = semopy.Model(model_spec)
    model.fit(df)
    return model


def extract_path_coefficients(model):
    """Extract and label path coefficients from a fitted SEM.

    Returns
    -------
    pandas.DataFrame
        Table of all estimated parameters with labels.
    """
    estimates = model.inspect()

    # Add human-readable labels where available
    labels = {}
    for key in ALL_CONSTRUCTS:
        labels[key] = get_construct_label(key)

    estimates = estimates.copy()
    estimates["lval_label"] = estimates["lval"].map(labels).fillna(estimates["lval"])
    estimates["rval_label"] = estimates["rval"].map(labels).fillna(estimates["rval"])

    return estimates


def compute_fit_indices(model):
    """Compute standard SEM fit indices.

    Returns
    -------
    pandas.DataFrame
        Single-row DataFrame with fit statistics.
    """
    stats_df = semopy.calc_stats(model)
    return stats_df


# ---------------------------------------------------------------------------
# Moderated SEM (multi-group approach)
# ---------------------------------------------------------------------------

def moderated_sem(df, moderator_key, model_spec=None):
    """Estimate the SEM at low / mean / high levels of a moderator.

    Splits the sample at ±1 SD of the moderator composite score and
    fits separate SEMs for each sub-group, allowing path coefficients
    to vary by moderator level.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing all observed indicator columns **and** the
        moderator composite score column.
    moderator_key : str
        Construct key of the moderator variable.
    model_spec : str, optional
        Custom model specification.

    Returns
    -------
    dict
        ``{level_label: (fitted_model, estimates_df, fit_df)}``
    """
    if model_spec is None:
        model_spec = build_sem_specification()

    mod_col = moderator_key
    if mod_col not in df.columns:
        warnings.warn(
            f"Moderator column '{mod_col}' not found in data. "
            "Skipping moderated SEM."
        )
        return {}

    mod_mean = df[mod_col].mean()
    mod_sd = df[mod_col].std()

    groups = {
        "Low (below -1 SD)": df[df[mod_col] <= mod_mean - mod_sd],
        "High (above +1 SD)": df[df[mod_col] >= mod_mean + mod_sd],
    }

    results = {}
    for label, sub_df in groups.items():
        if len(sub_df) < 30:
            warnings.warn(
                f"Moderator group '{label}' has only {len(sub_df)} "
                "observations – SEM results may be unreliable."
            )
        if len(sub_df) < 10:
            print(f"  ⚠  Skipping '{label}' – too few observations ({len(sub_df)}).")
            continue

        model = semopy.Model(model_spec)
        model.fit(sub_df)
        estimates = extract_path_coefficients(model)
        fit = compute_fit_indices(model)
        results[label] = (model, estimates, fit)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _format_structural_paths(estimates):
    """Filter estimates to only structural (regression) paths between latent constructs."""
    construct_keys = set(ALL_CONSTRUCTS.keys())
    mask = (
        (estimates["op"] == "~")
        & estimates["lval"].isin(construct_keys)
        & estimates["rval"].isin(construct_keys)
    )
    return estimates[mask].copy()


def _format_measurement_paths(estimates):
    """Filter estimates to only measurement (loading) paths from latent to observed."""
    construct_keys = set(ALL_CONSTRUCTS.keys())
    mask = (
        (estimates["op"] == "~")
        & ~estimates["lval"].isin(construct_keys)
        & estimates["rval"].isin(construct_keys)
    )
    return estimates[mask].copy()


def run_sem_analysis(df):
    """Execute the full SEM pipeline and save results.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing all observed indicator columns.

    Returns
    -------
    dict
        Dictionary with keys: ``model``, ``estimates``, ``fit``,
        ``structural_paths``, ``measurement_paths``,
        ``moderated_results``.
    """
    _ensure_output_dir()
    _section("Structural Equation Model (SEM)")

    # ----- Build & display specification -----
    spec = build_sem_specification()
    print("\nModel specification:")
    print(spec)
    spec_path = OUTPUT_DIR / "sem_specification.txt"
    spec_path.write_text(spec, encoding="utf-8")
    print(f"\n  → Specification saved to {spec_path}")

    # ----- Fit the model -----
    print("\nFitting SEM (maximum-likelihood estimation) ...")
    model = fit_sem(df, spec)
    print("  → Model fitted successfully.")

    # ----- Path coefficients -----
    estimates = extract_path_coefficients(model)

    structural = _format_structural_paths(estimates)
    measurement = _format_measurement_paths(estimates)

    print("\n--- Structural (regression) paths ---")
    display_cols = ["lval_label", "op", "rval_label", "Estimate",
                    "Std. Err", "z-value", "p-value"]
    avail_cols = [c for c in display_cols if c in structural.columns]
    print(structural[avail_cols].to_string(index=False))

    print("\n--- Measurement (factor loading) paths ---")
    avail_cols_m = [c for c in display_cols if c in measurement.columns]
    print(measurement[avail_cols_m].to_string(index=False))

    # Save full estimates
    estimates.to_csv(OUTPUT_DIR / "sem_estimates.csv", index=False)
    print(f"\n  → All estimates saved to {OUTPUT_DIR / 'sem_estimates.csv'}")

    # ----- Fit indices -----
    fit = compute_fit_indices(model)
    print("\n--- Model fit indices ---")
    print(fit.T.to_string())
    fit.to_csv(OUTPUT_DIR / "sem_fit_indices.csv", index=False)
    print(f"\n  → Fit indices saved to {OUTPUT_DIR / 'sem_fit_indices.csv'}")

    # ----- Moderated SEM -----
    moderator_keys = list(MODERATOR_VARIABLES.keys())
    moderated_results = {}
    all_mod_rows = []

    for mod_key in moderator_keys:
        label_mod = get_construct_label(mod_key)
        _section(f"Moderated SEM – {label_mod}")

        mod_results = moderated_sem(df, mod_key, spec)
        moderated_results[mod_key] = mod_results

        for level_label, (m, est, ft) in mod_results.items():
            print(f"\n--- {level_label} ---")
            struct = _format_structural_paths(est)
            avail = [c for c in display_cols if c in struct.columns]
            print(struct[avail].to_string(index=False))

            # Collect rows for CSV
            for _, row in struct.iterrows():
                all_mod_rows.append({
                    "Moderator": label_mod,
                    "Group": level_label,
                    "DV": row.get("lval_label", row["lval"]),
                    "Predictor": row.get("rval_label", row["rval"]),
                    "Estimate": row["Estimate"],
                    "Std. Err": row.get("Std. Err", np.nan),
                    "z-value": row.get("z-value", np.nan),
                    "p-value": row.get("p-value", np.nan),
                })

    if all_mod_rows:
        mod_df = pd.DataFrame(all_mod_rows)
        mod_df.to_csv(
            OUTPUT_DIR / "sem_moderated_paths.csv", index=False
        )
        print(
            f"\n  → Moderated paths saved to "
            f"{OUTPUT_DIR / 'sem_moderated_paths.csv'}"
        )

    # ----- Path diagram (text-based summary) -----
    _save_path_diagram_summary(structural)

    return {
        "model": model,
        "estimates": estimates,
        "fit": fit,
        "structural_paths": structural,
        "measurement_paths": measurement,
        "moderated_results": moderated_results,
    }


def _save_path_diagram_summary(structural):
    """Write a plain-text path diagram summary to the output directory."""
    _ensure_output_dir()
    lines = [
        "Structural Equation Model – Path Diagram Summary",
        "=" * 50,
        "",
        "Conceptual Model:",
        "",
        "  Firm Active Engagement (IV)",
        "      │",
        "      ├──→ Structural Capital ──┐",
        "      ├──→ Relational Capital ──┤",
        "      ├──→ Cognitive Capital  ──┼──→ Community Engagement (DV)",
        "      ├──→ Trust              ──┘",
        "      │                              ",
        "      └────────────────────────────→ (direct path)",
        "",
        "  Moderated by:",
        "      • Social vs Functional Needs",
        "      • Need for Self-Esteem Enhancement",
        "",
        "Estimated Structural Paths:",
        "-" * 50,
    ]

    for _, row in structural.iterrows():
        lbl = f"  {row.get('rval_label', row['rval']):>30s}  →  " \
              f"{row.get('lval_label', row['lval']):<30s}"
        est = f"  β = {row['Estimate']:.4f}"
        pval = row.get("p-value", np.nan)
        sig = ""
        try:
            pval_f = float(pval)
        except (TypeError, ValueError):
            pval_f = np.nan
        if pd.notna(pval_f):
            if pval_f < 0.001:
                sig = " ***"
            elif pval_f < 0.01:
                sig = " **"
            elif pval_f < 0.05:
                sig = " *"
            est += f"  (p = {pval_f:.4f}){sig}"
        lines.append(lbl + est)

    lines.append("")
    lines.append("Significance: * p<.05  ** p<.01  *** p<.001")

    path = OUTPUT_DIR / "sem_path_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  → Path summary saved to {path}")
