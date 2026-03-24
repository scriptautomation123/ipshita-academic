"""
Mediation–Moderation Analysis
==============================

Implements the full statistical analysis pipeline for the conceptual model:

    IV  (Firm Active Engagement)
        → M  (Structural Capital, Relational Capital, Cognitive Capital, Trust)
            → DV (Community Engagement)

    with moderators: Social vs Functional Needs, Need for Self-Esteem Enhancement

Analyses included
-----------------
1. Descriptive statistics
2. Reliability analysis (Cronbach's alpha)
3. Correlation matrix (with significance)
4. Simple mediation (Baron & Kenny + Sobel test) for each mediator
5. Moderation analysis (interaction effects)
6. Moderated mediation (conditional indirect effects)
7. Visualisations (correlation heat-map, path diagram sketch)
"""

import os
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from analysis.data_loader import (
    compute_construct_scores,
    get_dv_keys,
    get_iv_keys,
    get_mediator_keys,
    get_moderator_keys,
)
from config.constructs import ALL_CONSTRUCTS, get_construct_label

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path("output")


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _section(title):
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# 1. Descriptive Statistics
# ---------------------------------------------------------------------------
def descriptive_statistics(df, construct_keys):
    """Print and return descriptive statistics for each construct."""
    _section("Descriptive Statistics")
    rows = []
    for key in construct_keys:
        if key not in df.columns:
            continue
        s = df[key].dropna()
        rows.append({
            "Construct": get_construct_label(key),
            "N": len(s),
            "Mean": round(s.mean(), 3),
            "Std Dev": round(s.std(), 3),
            "Min": s.min(),
            "Max": s.max(),
            "Skewness": round(s.skew(), 3),
            "Kurtosis": round(s.kurtosis(), 3),
        })
    desc = pd.DataFrame(rows)
    print(desc.to_string(index=False))
    return desc


# ---------------------------------------------------------------------------
# 2. Reliability Analysis (Cronbach's Alpha)
# ---------------------------------------------------------------------------
def reliability_analysis(df):
    """Compute Cronbach's alpha for every construct."""
    _section("Reliability Analysis (Cronbach's Alpha)")
    rows = []
    for key, meta in ALL_CONSTRUCTS.items():
        items = [c for c in meta["items"] if c in df.columns]
        if len(items) < 2:
            continue
        alpha = pg.cronbach_alpha(df[items].dropna())[0]
        rows.append({
            "Construct": meta["label"],
            "Items": len(items),
            "Cronbach α": round(alpha, 3),
        })
    rel = pd.DataFrame(rows)
    print(rel.to_string(index=False))
    return rel


# ---------------------------------------------------------------------------
# 3. Correlation Matrix
# ---------------------------------------------------------------------------
def correlation_matrix(df, construct_keys):
    """Pearson correlations with significance stars."""
    _section("Correlation Matrix")
    sub = df[construct_keys].dropna()
    labels = [get_construct_label(k) for k in construct_keys]
    corr = sub.corr()
    corr.index = labels
    corr.columns = labels

    # Significance matrix
    n = len(sub)
    p_values = pd.DataFrame(np.zeros_like(corr), index=labels, columns=labels)
    for i, ci in enumerate(construct_keys):
        for j, cj in enumerate(construct_keys):
            if i == j:
                p_values.iloc[i, j] = 0.0
            else:
                _, p = stats.pearsonr(sub[ci], sub[cj])
                p_values.iloc[i, j] = p

    print(corr.round(3).to_string())
    print("\np-values:")
    print(p_values.round(4).to_string())

    # Heatmap
    _ensure_output_dir()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Correlation Matrix – All Constructs")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\n  → Heatmap saved to {OUTPUT_DIR / 'correlation_matrix.png'}")

    return corr, p_values


# ---------------------------------------------------------------------------
# 4. Mediation Analysis (Baron & Kenny + Sobel)
# ---------------------------------------------------------------------------
def _ols_summary(y, X, df):
    """Fit OLS and return the results object."""
    X_data = sm.add_constant(df[X].dropna())
    y_data = df[y].loc[X_data.index]
    model = sm.OLS(y_data, X_data).fit()
    return model


def mediation_analysis(df, iv, mediators, dv):
    """Run Baron & Kenny mediation for each mediator.

    Steps per mediator M:
        Path c  : IV → DV  (total effect)
        Path a  : IV → M
        Path b  : M  → DV  (controlling for IV)
        Path c' : IV → DV  (controlling for M – direct effect)
        Indirect = a × b   (Sobel test for significance)
    """
    _section("Mediation Analysis (Baron & Kenny + Sobel Test)")
    results = []
    for m in mediators:
        if m not in df.columns or iv not in df.columns or dv not in df.columns:
            continue
        label_m = get_construct_label(m)
        print(f"\n--- Mediator: {label_m} ---")

        sub = df[[iv, m, dv]].dropna()
        X_iv = sm.add_constant(sub[iv])

        # Path c: IV → DV
        model_c = sm.OLS(sub[dv], X_iv).fit()
        c = model_c.params[iv]
        c_p = model_c.pvalues[iv]

        # Path a: IV → M
        model_a = sm.OLS(sub[m], X_iv).fit()
        a = model_a.params[iv]
        a_se = model_a.bse[iv]
        a_p = model_a.pvalues[iv]

        # Path b & c': M + IV → DV
        X_both = sm.add_constant(sub[[iv, m]])
        model_bc = sm.OLS(sub[dv], X_both).fit()
        b = model_bc.params[m]
        b_se = model_bc.bse[m]
        b_p = model_bc.pvalues[m]
        c_prime = model_bc.params[iv]
        c_prime_p = model_bc.pvalues[iv]

        # Indirect effect & Sobel test
        indirect = a * b
        sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
        sobel_z = indirect / sobel_se
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        info = {
            "Mediator": label_m,
            "Path a (IV→M)": round(a, 4),
            "a p-value": round(a_p, 4),
            "Path b (M→DV)": round(b, 4),
            "b p-value": round(b_p, 4),
            "Total c (IV→DV)": round(c, 4),
            "c p-value": round(c_p, 4),
            "Direct c' (IV→DV|M)": round(c_prime, 4),
            "c' p-value": round(c_prime_p, 4),
            "Indirect (a×b)": round(indirect, 4),
            "Sobel Z": round(sobel_z, 4),
            "Sobel p": round(sobel_p, 4),
        }
        results.append(info)

        for k, v in info.items():
            print(f"  {k:25s} : {v}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. Moderation Analysis
# ---------------------------------------------------------------------------
def moderation_analysis(df, iv, dv, moderators):
    """Test each moderator on the IV → DV relationship.

    Fits  DV = b0 + b1·IV + b2·Mod + b3·(IV×Mod) + ε
    and reports the interaction coefficient b3.
    """
    _section("Moderation Analysis")
    results = []
    for mod in moderators:
        if mod not in df.columns:
            continue
        label_mod = get_construct_label(mod)
        print(f"\n--- Moderator: {label_mod} ---")

        sub = df[[iv, dv, mod]].dropna()
        # Mean-centre for interaction
        iv_c = sub[iv] - sub[iv].mean()
        mod_c = sub[mod] - sub[mod].mean()
        interaction = iv_c * mod_c

        X = pd.DataFrame({
            "const": 1,
            iv: iv_c,
            mod: mod_c,
            f"{iv}_x_{mod}": interaction,
        })
        model = sm.OLS(sub[dv], X).fit()
        print(model.summary2().tables[1].to_string())

        info = {
            "Moderator": label_mod,
            "Interaction coeff": round(model.params[f"{iv}_x_{mod}"], 4),
            "Interaction p": round(model.pvalues[f"{iv}_x_{mod}"], 4),
            "R-squared": round(model.rsquared, 4),
            "Adj R-squared": round(model.rsquared_adj, 4),
        }
        results.append(info)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 6. Moderated Mediation
# ---------------------------------------------------------------------------
def moderated_mediation(df, iv, mediators, dv, moderators):
    """Test if the indirect effect via each mediator varies by moderator.

    For each (mediator, moderator) pair:
        Stage 1  :  M = a0 + a1·IV + a2·Mod + a3·(IV×Mod) + ε1
        Stage 2  :  DV = b0 + b1·M  + b2·IV + ε2
        Conditional indirect at Mod = mean ± 1 SD
    """
    _section("Moderated Mediation (Conditional Indirect Effects)")
    results = []
    for m in mediators:
        for mod in moderators:
            if not all(k in df.columns for k in [iv, m, dv, mod]):
                continue
            label_m = get_construct_label(m)
            label_mod = get_construct_label(mod)
            print(f"\n--- Mediator: {label_m}  |  Moderator: {label_mod} ---")

            sub = df[[iv, m, dv, mod]].dropna()
            mod_mean = sub[mod].mean()
            mod_sd = sub[mod].std()

            iv_c = sub[iv] - sub[iv].mean()
            mod_c = sub[mod] - mod_mean
            interaction = iv_c * mod_c

            # Stage 1: IV (+ Mod + interaction) → M
            X1 = sm.add_constant(pd.DataFrame({
                iv: iv_c,
                mod: mod_c,
                "interaction": interaction,
            }))
            m1 = sm.OLS(sub[m], X1).fit()

            # Stage 2: M + IV → DV
            X2 = sm.add_constant(sub[[m, iv]])
            m2 = sm.OLS(sub[dv], X2).fit()
            b_m = m2.params[m]

            # Conditional indirect effects
            for level_label, mod_val in [
                ("Low (−1 SD)", -mod_sd),
                ("Mean", 0),
                ("High (+1 SD)", mod_sd),
            ]:
                a_cond = m1.params[iv] + m1.params["interaction"] * mod_val
                indirect = a_cond * b_m
                info = {
                    "Mediator": label_m,
                    "Moderator": label_mod,
                    "Mod Level": level_label,
                    "a (conditional)": round(a_cond, 4),
                    "b": round(b_m, 4),
                    "Indirect": round(indirect, 4),
                }
                results.append(info)
                print(f"  {level_label:15s}  a={a_cond:.4f}  b={b_m:.4f}  indirect={indirect:.4f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def run_full_analysis(df):
    """Execute the complete analysis pipeline and save results."""
    _ensure_output_dir()

    # Compute composite scores
    df = compute_construct_scores(df)

    iv_keys = get_iv_keys()
    med_keys = get_mediator_keys()
    dv_keys = get_dv_keys()
    mod_keys = get_moderator_keys()
    all_keys = iv_keys + med_keys + dv_keys + mod_keys

    # 1. Descriptive statistics
    desc = descriptive_statistics(df, all_keys)
    desc.to_csv(OUTPUT_DIR / "descriptive_statistics.csv", index=False)

    # 2. Reliability
    rel = reliability_analysis(df)
    rel.to_csv(OUTPUT_DIR / "reliability.csv", index=False)

    # 3. Correlation matrix
    corr, pvals = correlation_matrix(df, all_keys)
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # 4. Mediation (for each IV → each DV, through all mediators)
    all_med = []
    for iv in iv_keys:
        for dv in dv_keys:
            med_df = mediation_analysis(df, iv, med_keys, dv)
            all_med.append(med_df)
    if all_med:
        med_results = pd.concat(all_med, ignore_index=True)
        med_results.to_csv(OUTPUT_DIR / "mediation_results.csv", index=False)

    # 5. Moderation
    all_mod = []
    for iv in iv_keys:
        for dv in dv_keys:
            mod_df = moderation_analysis(df, iv, dv, mod_keys)
            all_mod.append(mod_df)
    if all_mod:
        mod_results = pd.concat(all_mod, ignore_index=True)
        mod_results.to_csv(OUTPUT_DIR / "moderation_results.csv", index=False)

    # 6. Moderated mediation
    mm_results = moderated_mediation(df, iv_keys[0], med_keys, dv_keys[0], mod_keys)
    mm_results.to_csv(OUTPUT_DIR / "moderated_mediation_results.csv", index=False)

    _section("Analysis Complete")
    print(f"All results saved to {OUTPUT_DIR.resolve()}/")
    return {
        "descriptive": desc,
        "reliability": rel,
        "correlation": corr,
        "mediation": med_results if all_med else pd.DataFrame(),
        "moderation": mod_results if all_mod else pd.DataFrame(),
        "moderated_mediation": mm_results,
    }
