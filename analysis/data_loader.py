"""
Data Loader
Utilities for loading and preparing the survey data for analysis.
"""

import pathlib

import numpy as np
import pandas as pd

from config.constructs import (
    ALL_CONSTRUCTS,
    DEPENDENT_VARIABLES,
    INDEPENDENT_VARIABLES,
    MEDIATOR_VARIABLES,
    MODERATOR_VARIABLES,
)


# Default path – users may override via function arguments.
_DEFAULT_DATA_PATH = pathlib.Path("data")


def load_survey_data(filepath=None):
    """Load the cleaned survey data from an Excel file.

    Parameters
    ----------
    filepath : str or pathlib.Path, optional
        Path to the Excel workbook.  When *None* the loader looks for the
        first ``.xlsx`` file inside the ``data/`` directory.

    Returns
    -------
    pandas.DataFrame
    """
    if filepath is None:
        data_dir = _DEFAULT_DATA_PATH
        xlsx_files = sorted(data_dir.glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(
                f"No .xlsx files found in {data_dir.resolve()}. "
                "Place the survey data file in the data/ directory or "
                "pass an explicit filepath."
            )
        filepath = xlsx_files[0]

    filepath = pathlib.Path(filepath)
    print(f"Loading data from {filepath} ...")
    df = pd.read_excel(filepath, engine="openpyxl")
    print(f"  → {len(df)} rows, {len(df.columns)} columns loaded.")
    return df


def compute_construct_scores(df):
    """Compute composite (mean) scores for every construct.

    For each construct defined in ``config.constructs``, the function
    calculates the row-wise mean of the constituent survey items and
    stores the result in a new column named after the construct key.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw survey data containing the individual item columns.

    Returns
    -------
    pandas.DataFrame
        A **copy** of *df* with the new composite-score columns appended.
    """
    df = df.copy()
    for key, meta in ALL_CONSTRUCTS.items():
        items = meta["items"]
        present = [c for c in items if c in df.columns]
        if not present:
            print(
                f"  ⚠  No matching columns for construct '{key}' "
                f"(expected: {items}). Skipping."
            )
            continue
        if len(present) < len(items):
            missing = set(items) - set(present)
            print(
                f"  ⚠  Construct '{key}': columns {missing} not found. "
                f"Computing mean from {len(present)} of {len(items)} items."
            )
        df[key] = df[present].mean(axis=1)
    return df


def get_variable_keys(variable_group):
    """Return the construct keys for a variable group dictionary."""
    return list(variable_group.keys())


def get_iv_keys():
    """Return independent variable construct keys."""
    return get_variable_keys(INDEPENDENT_VARIABLES)


def get_mediator_keys():
    """Return mediator variable construct keys."""
    return get_variable_keys(MEDIATOR_VARIABLES)


def get_dv_keys():
    """Return dependent variable construct keys."""
    return get_variable_keys(DEPENDENT_VARIABLES)


def get_moderator_keys():
    """Return moderator variable construct keys."""
    return get_variable_keys(MODERATOR_VARIABLES)


def generate_sample_data(n=300, seed=42):
    """Generate synthetic sample data for testing purposes.

    Creates a DataFrame with columns matching every survey item defined in
    ``config.constructs``, filled with random Likert-scale responses
    (1–7).  Correlations are loosely injected so that mediation and
    moderation effects are detectable in the sample.

    Parameters
    ----------
    n : int
        Number of synthetic respondents.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
    """
    rng = np.random.default_rng(seed)

    data = {}
    # Base latent factors to induce correlations
    latent_iv = rng.normal(4, 1, n)
    latent_med = 0.5 * latent_iv + rng.normal(0, 0.8, n)
    latent_dv = 0.4 * latent_med + 0.3 * latent_iv + rng.normal(0, 0.7, n)
    latent_mod = rng.normal(4, 1, n)

    def _likert(latent):
        return np.clip(np.round(latent + rng.normal(0, 0.6, n)), 1, 7).astype(int)

    for key, meta in INDEPENDENT_VARIABLES.items():
        for item in meta["items"]:
            data[item] = _likert(latent_iv)

    for key, meta in MEDIATOR_VARIABLES.items():
        for item in meta["items"]:
            data[item] = _likert(latent_med)

    for key, meta in DEPENDENT_VARIABLES.items():
        for item in meta["items"]:
            data[item] = _likert(latent_dv)

    for key, meta in MODERATOR_VARIABLES.items():
        for item in meta["items"]:
            data[item] = _likert(latent_mod)

    return pd.DataFrame(data)
