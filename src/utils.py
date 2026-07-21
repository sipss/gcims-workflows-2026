"""Utility functions for GC-IMS data.

This module provides reusable helpers for loading tabular data, computing
relative standard deviation (RSD), and building one-hot encoded design
matrices for categorical labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def _read_table(path: str) -> pd.DataFrame:
	"""Read a tabular file into a DataFrame.

	Parameters
	----------
	path : str
		Path to an input table. Supported formats: CSV, TSV, TXT, Excel,
		and Parquet.

	Returns
	-------
	pd.DataFrame
		Loaded table.

	Raises
	------
	ValueError
		If the file extension is not supported.
	"""
	file_path = Path(path)
	suffix = file_path.suffix.lower()

	if suffix in {".csv"}:
		return pd.read_csv(file_path)
	if suffix in {".tsv", ".txt"}:
		return pd.read_csv(file_path, sep="\t")
	if suffix in {".xlsx", ".xls"}:
		return pd.read_excel(file_path)
	if suffix in {".parquet"}:
		return pd.read_parquet(file_path)

	raise ValueError(
		f"Unsupported file extension '{suffix}' for '{path}'. "
		"Supported: .csv, .tsv, .txt, .xlsx, .xls, .parquet"
	)


def load_data(X_path: str, meta_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Load feature and metadata tables with flexible input handling.

	This function supports two common scenarios:
	1. Features and metadata are in separate files.
	2. Features and metadata are in the same file.

	When a single file is provided, columns with numeric dtype are interpreted
	as candidate feature columns and non-numeric columns as metadata columns.
	If columns starting with ``Cluster`` exist, they are prioritized as feature
	columns because this naming pattern is common in GC-IMS peak tables.

	Parameters
	----------
	X_path : str
		Path to the main input table.
	meta_path : str, optional
		Optional path to a separate metadata table.

	Returns
	-------
	tuple[pd.DataFrame, pd.DataFrame]
		``(X_df, meta_df)`` where:
		- ``X_df`` contains only numeric feature columns.
		- ``meta_df`` contains metadata columns.

	Raises
	------
	ValueError
		If no feature columns can be inferred or if row counts do not match.
	"""
	x_table = _read_table(X_path)

	if meta_path is not None:
		meta_df = _read_table(meta_path)
		if len(x_table) != len(meta_df):
			raise ValueError(
				"Feature and metadata tables have different numbers of rows: "
				f"{len(x_table)} vs {len(meta_df)}."
			)

		X_df = x_table.copy()
	else:
		cluster_cols = [col for col in x_table.columns if str(col).startswith("Cluster")]
		if cluster_cols:
			feature_cols = cluster_cols
			meta_cols = [col for col in x_table.columns if col not in feature_cols]
		else:
			feature_cols = [
				col for col in x_table.columns if pd.api.types.is_numeric_dtype(x_table[col])
			]
			meta_cols = [col for col in x_table.columns if col not in feature_cols]

		if not feature_cols:
			raise ValueError(
				"Could not infer feature columns from input table. "
				"Provide a feature-only file or include numeric feature columns."
			)

		X_df = x_table.loc[:, feature_cols].copy()
		meta_df = x_table.loc[:, meta_cols].copy()

	X_df = X_df.apply(pd.to_numeric, errors="coerce")
	X_df = X_df.dropna(axis=1, how="all")

	if X_df.empty:
		raise ValueError("No valid numeric feature columns found after cleaning.")

	X_df = X_df.reset_index(drop=True)
	meta_df = meta_df.reset_index(drop=True)
	return X_df, meta_df


def compute_rsd(X: np.ndarray) -> np.ndarray:
	"""Compute Relative Standard Deviation (RSD, %) for each feature.

	The sample standard deviation is used (``ddof=1``).

	Parameters
	----------
	X : np.ndarray
		Feature matrix with shape ``(n_samples, n_features)``.

	Returns
	-------
	np.ndarray
		Vector of shape ``(n_features,)`` containing RSD values in percent.
		Features with near-zero mean return ``np.nan``.
	"""
	X = np.asarray(X, dtype=float)
	if X.ndim != 2:
		raise ValueError(f"X must be a 2D array. Received shape: {X.shape}")

	means = np.mean(X, axis=0)
	stds = np.std(X, axis=0, ddof=1)
	abs_means = np.abs(means)

	with np.errstate(divide="ignore", invalid="ignore"):
		rsd = (stds / abs_means) * 100.0

	rsd[np.isclose(abs_means, 0.0)] = np.nan
	return rsd


def build_design_matrix(labels: np.ndarray) -> np.ndarray:
	"""Build a one-hot encoded design matrix from categorical labels.

	Parameters
	----------
	labels : np.ndarray
		A one-dimensional array with one label per sample.

	Returns
	-------
	np.ndarray
		One-hot encoded matrix with shape ``(n_samples, n_levels)``.

	Raises
	------
	ValueError
		If labels are not one-dimensional.
	"""
	labels = np.asarray(labels)
	if labels.ndim != 1:
		raise ValueError(f"labels must be 1D. Received shape: {labels.shape}")

	cat = pd.Categorical(labels, categories=pd.unique(labels), ordered=True)
	design = pd.get_dummies(cat, drop_first=False, dtype=float)
	return design.to_numpy()

