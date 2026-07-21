"""Model definitions"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh, solve
from scipy.stats import f_oneway, linregress, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from typing import List, Tuple, Union

from itertools import combinations

# LOESS
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d

# SERRF
from sklearn.ensemble import RandomForestRegressor


from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
import numpy as np
import pandas as pd
from typing import List, Tuple, Union



class EPOCorrector(BaseEstimator, TransformerMixin):
    """
    External Parameter Orthogonalization (EPO) corrector for multivariate data.

    This transformer removes technical variance associated with external
    acquisition factors (e.g., batch effects, temporal drift) using the 
    orthogonal projection workflow described by Duran-Fort et al.

    The correction process follows these steps:
    1.  **Reconstruction:** Each technical component is isolated by projecting the 
        centered data onto the column space of a factor-specific design matrix (A_f):
        X_f = A_f (A_f^T A_f)^-1 A_f^T X
    2.  **Score Extraction:** Principal Component Analysis (PCA) is applied to each 
        reconstructed matrix to extract the dominant technical directions (scores).
    3.  **Orthogonal Projection:** A sample-space projection matrix is built using 
        the extracted scores to remove the technical subspace from the original data.

    Parameters
    ----------
    n_components : int, default=2
        Total number of technical score directions to remove. Kept primarily for 
        backward compatibility. If used alone (while batch/acq specific components 
        are None), the components are distributed evenly among the external factors.
    n_components_batch : int, default=None
        Specific number of principal components to extract and remove from the 
        batch design space (e.g., modeling discrete shifts between batches).
    n_components_acq : int, default=None
        Specific number of principal components to extract and remove from the 
        acquisition order space (e.g., modeling continuous intra-batch drift).
    """

    def __init__(
        self, 
        n_components: int = 2, 
        n_components_batch: Union[int, None] = None, 
        n_components_acq: Union[int, None] = None
    ):
        self.n_components = n_components
        self.n_components_batch = n_components_batch
        self.n_components_acq = n_components_acq

    def _build_design_matrix(self, labels: np.ndarray) -> np.ndarray:
        """
        Build a one-hot encoded design matrix from categorical labels.

        Parameters
        ----------
        labels : np.ndarray
            A 1D array containing class labels for each sample.

        Returns
        -------
        np.ndarray
            A binary matrix of shape (n_samples, n_unique_labels).
        """
        if labels.ndim != 1:
            raise ValueError(f"Labels must be 1D. Received shape: {labels.shape}")
        
        cat = pd.Categorical(labels, categories=pd.unique(labels), ordered=True)
        design = pd.get_dummies(cat, drop_first=False, dtype=float)
        return design.to_numpy()

    def _build_external_designs(self, y_external: np.ndarray) -> List[np.ndarray]:
        """
        Create a list of design matrices for one or multiple external factors.

        Parameters
        ----------
        y_external : np.ndarray
            External technical labels (1D for a single factor, 2D for multiple).

        Returns
        -------
        List[np.ndarray]
            A list containing the one-hot encoded design matrix for each factor.
        """
        y_external = np.asarray(y_external)
        
        if y_external.ndim == 1:
            return [self._build_design_matrix(y_external)]
        if y_external.ndim == 2:
            return [self._build_design_matrix(y_external[:, i]) for i in range(y_external.shape[1])]
        
        raise ValueError("y_external must be a 1D or 2D array-like structure.")

    @staticmethod
    def _hat_project(design: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Project matrix X onto the column space defined by a design matrix.

        This effectively isolates the variance in X that is linearly explained 
        by the external factors encoded in the design matrix.

        Parameters
        ----------
        design : np.ndarray
            The design matrix (A).
        X : np.ndarray
            The feature matrix to project.

        Returns
        -------
        np.ndarray
            The reconstructed matrix: A * (A^T * A)^-1 * A^T * X
        """
        # Using pseudo-inverse (pinv) ensures stability if the design matrix is not full rank
        return design @ np.linalg.pinv(design.T @ design) @ design.T @ X

    @staticmethod
    def _extract_scores(X_recon: np.ndarray, n_comps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the dominant principal component scores from a reconstructed matrix.

        Parameters
        ----------
        X_recon : np.ndarray
            The matrix reconstructed from the design projection.
        n_comps : int
            The number of principal components to extract.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - scores: Array of shape (n_samples, actual_comps) containing the PC scores.
            - ev_ratio: Array of shape (actual_comps,) with the explained variance ratio (%).
        """
        # Autoscale before PCA to give equal weight to all reconstructed features
        std = X_recon.std(axis=0, ddof=1)
        std[std < 1e-12] = 1.0  # Prevent division by zero for constant features
        X_scaled = (X_recon - X_recon.mean(axis=0)) / std

        # Safely determine the maximum number of components that can be extracted
        max_possible_comps = min(X_scaled.shape[0], X_scaled.shape[1])
        actual_comps = min(n_comps, max_possible_comps)

        if actual_comps < 1:
            return np.empty((X_scaled.shape[0], 0)), np.array([])

        pca = PCA(n_components=actual_comps)
        scores = pca.fit_transform(X_scaled)
        ev_ratio = pca.explained_variance_ratio_ * 100.0
        
        return scores, ev_ratio

    def fit(self, X: np.ndarray, y_external: Union[np.ndarray, pd.DataFrame]):
        """
        Fit the EPO correction model by identifying the technical subspaces.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) representing the 
            Quality Control (QC) samples.
        y_external : Union[np.ndarray, pd.DataFrame]
            External technical labels corresponding to X. 
            If a DataFrame is provided, it assumes columns ['batch', 'acq_order'] exist.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Handle DataFrame input naturally
        if hasattr(y_external, 'columns'):
            y_external = y_external[["batch", "acq_order"]].to_numpy()

        X = check_array(X, ensure_2d=True, dtype=float)
        designs = self._build_external_designs(y_external)
        
        # --- Component Distribution Logic ---
        # Determine how many components to extract per factor
        if self.n_components_batch is None or self.n_components_acq is None:
            # Fallback for backward compatibility: distribute n_components evenly
            self.n_batch_ = self.n_components // 2 + (self.n_components % 2) 
            self.n_acq_ = self.n_components // 2
        else:
            self.n_batch_ = self.n_components_batch
            self.n_acq_ = self.n_components_acq
            self.n_components = self.n_batch_ + self.n_acq_ # Sync total

        # Map requested components to the input factors (Factor 0 = batch, Factor 1 = acq_order)
        comp_mapping = [self.n_batch_]
        if len(designs) > 1:
            comp_mapping.append(self.n_acq_)

        # Save input dimensions and global mean for centering
        self.n_features_in_ = X.shape[1]
        self.n_samples_in_ = X.shape[0]
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        self.X_centered_ = X_centered

        # Step 1: Reconstruct the technical data from the external factors
        X_reconstructed_factors = [self._hat_project(design, X_centered) for design in designs]
        self.design_matrices_ = designs
        self.X_reconstructed_factors_ = X_reconstructed_factors

        factor_scores = []
        explained_variance = []
        
        # Step 2: Extract principal components (scores) from each reconstruction
        for i, X_recon in enumerate(X_reconstructed_factors):
            # Use mapped component count, default to 1 if mapping fails
            target_comps = comp_mapping[i] if i < len(comp_mapping) else 1
            
            scores, ev = self._extract_scores(X_recon, target_comps)
            if scores.shape[1] > 0:
                factor_scores.append(scores)
                explained_variance.extend(ev)

        if len(factor_scores) == 0:
            raise ValueError("No technical score directions could be identified. Check inputs.")
            
        # Combine scores from all factors into a single basis matrix (U)
        self.component_scores_ = np.column_stack(factor_scores)
        self.factor_explained_variance_ratio_ = np.asarray(explained_variance, dtype=float)

        # Step 3: Build the sample-space orthogonal projector
        # P = I - U(U^T U)^-1 U^T
        gram_scores = self.component_scores_.T @ self.component_scores_
        self.sample_projection_matrix_ = (
            np.eye(self.n_samples_in_) - 
            self.component_scores_ @ np.linalg.pinv(gram_scores) @ self.component_scores_.T
        )

        self.n_components_effective_ = self.component_scores_.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the sample-space EPO projection to a feature matrix.

        **Warning:** Because this uses a sample-space projector, `X` must have 
        the exact same number of samples (and in the same order) as the data 
        passed to `fit()`. For new/unseen samples, use the `InductiveEPOCorrector` 
        wrapper instead.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The corrected feature matrix with the original global mean restored.
        """
        check_is_fitted(self, attributes=["mean_", "component_scores_", "sample_projection_matrix_"])
        X = check_array(X, ensure_2d=True, dtype=float)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch. Expected {self.n_features_in_}, "
                f"but received {X.shape[1]}."
            )
            
        if X.shape[0] != self.n_samples_in_:
            raise ValueError(
                f"Sample dimension mismatch. Expected {self.n_samples_in_}, "
                f"but received {X.shape[0]}. "
                "Note: EPOCorrector is a sample-space projector. To transform "
                "new samples, you must wrap this model in InductiveEPOCorrector."
            )

        X_centered = X - self.mean_
        X_corrected_centered = self.sample_projection_matrix_ @ X_centered
        
        return X_corrected_centered + self.mean_



class LOESSCorrector(BaseEstimator, TransformerMixin):
    """
    Robust LOESS Signal Correction (QC-RLSC).
    
    Fits a locally weighted polynomial regression (LOESS) curve to the QC samples 
    along the injection time axis for each biomarker independently.
    
    Reference: Dunn, W. B. et al. (2011). Nature Protocols.
    
    Parameters
    ----------
    frac : float, default=0.25
        The fraction of the data used when estimating each y-value.
        Justification: Since our data has defined local batches (e.g., 9 batches), 
        a frac of 0.25 covers approximately 2.25 batches. This localizes the 
        smoothing window enough to capture intra-batch drift efficiently without 
        over-smoothing into a flat global line (which frac=0.5+ might do).
    it : int, default=3
        The number of residual-based robustifying iterations.
        Justification: Protects the drift curve from being distorted by anomalous 
        QC injections (outliers).
    """
    
    def __init__(self, frac=0.12, it=3):
        self.frac = frac
        self.it = it
        
    def fit(self, X_qc, meta_qc):
        """Fits the LOESS curve for each feature based on QC timestamps."""
        X_qc = check_array(X_qc, dtype=float, ensure_2d=True)
        time_qc = self._extract_global_time(meta_qc)
        
        self.qc_medians_ = np.median(X_qc, axis=0)
        self.interpolators_ = []
        
        for j in range(X_qc.shape[1]):
            # Fit LOESS with robustness iterations
            z = lowess(X_qc[:, j], time_qc, frac=self.frac, it=self.it, return_sorted=True)
            x_sm, y_sm = z[:, 0], z[:, 1]
            
            # Clean duplicate x values (prevents interp1d from crashing)
            _, idx = np.unique(x_sm, return_index=True)
            
            # Create a 1D interpolator. 'extrapolate' is required because patient 
            # samples might occur slightly before the first QC or after the last QC.
            f_int = interp1d(
                x_sm[idx], y_sm[idx], 
                kind='linear', bounds_error=False, fill_value="extrapolate"
            )
            self.interpolators_.append(f_int)
            
        return self
        
    def transform(self, X, meta_pat):
        """Applies LOESS correction by dividing by the interpolated curve value."""
        check_is_fitted(self, ['interpolators_', 'qc_medians_'])
        X = check_array(X, dtype=float, ensure_2d=True)
        time_pat = self._extract_global_time(meta_pat)
        
        X_corr = np.zeros_like(X)
        
        for j in range(X.shape[1]):
            pred_trend = self.interpolators_[j](time_pat)
            # Safeguard against zero division and negative extrapolations
            pred_trend = np.clip(pred_trend, a_min=1e-8, a_max=None)
            X_corr[:, j] = (X[:, j] / pred_trend) * self.qc_medians_[j]
            
        return X_corr

    def _extract_global_time(self, meta):
        """Safely unrolls relative batch times into a continuous global timeline."""
        if isinstance(meta, pd.DataFrame) and 'acq_order' in meta.columns and 'batch' in meta.columns:
            offset_multiplier = 20.0 
            batch_numeric = pd.factorize(meta['batch'])[0]
            global_time = (batch_numeric * offset_multiplier) + meta['acq_order'].values
            return global_time.astype(float)
        return np.array(meta).flatten().astype(float)



    
class SERRFCorrector(BaseEstimator, TransformerMixin):
    """
    Systematic Error Removal using Random Forest (SERRF).
    
    This univariate transformer models the technical drift of each biomarker
    in the temporal domain using a Random Forest regressor trained exclusively 
    on the Quality Control (QC) samples.
    
    Reference: Fan, S. et al. (2019). Analytical Chemistry.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the Random Forest. 
        Justification: With ~3 to 7 QCs per batch across ~9 batches (max ~63 QCs), 
        100 trees provide a highly stable ensemble average without causing 
        unnecessary computational overhead during cross-validation or Optuna sweeps.
    max_depth : int, default=None
    min_samples_split : int, default=2
    random_state : int, default=42
        Seed for reproducibility.
    """

    def __init__(self, n_estimators=50, max_depth=5, min_samples_split=2, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
    def fit(self, X_qc, meta_qc):
        X_qc = check_array(X_qc, dtype=float, ensure_2d=True)
        time_qc = self._extract_global_time(meta_qc).reshape(-1, 1)
        
        self.qc_medians_ = np.median(X_qc, axis=0)
        self.models_ = []
        
        for j in range(X_qc.shape[1]):
            # Using parameters explicitly passed to the class
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=1 
            )
            rf.fit(time_qc, X_qc[:, j])
            self.models_.append(rf)
            
        return self
        
    def transform(self, X, meta_pat):
        check_is_fitted(self, ['models_', 'qc_medians_'])
        X = check_array(X, dtype=float, ensure_2d=True)
        time_pat = self._extract_global_time(meta_pat).reshape(-1, 1)
        
        X_corr = np.zeros_like(X)
        for j in range(X.shape[1]):
            pred_trend = self.models_[j].predict(time_pat)
            pred_trend = np.clip(pred_trend, a_min=1e-8, a_max=None)
            X_corr[:, j] = (X[:, j] / pred_trend) * self.qc_medians_[j]
            
        return X_corr
    
    def _extract_global_time(self, meta):
        if isinstance(meta, pd.DataFrame) and 'acq_order' in meta.columns and 'batch' in meta.columns:
            offset_multiplier = 20.0 
            batch_numeric = pd.factorize(meta['batch'])[0]
            global_time = (batch_numeric * offset_multiplier) + meta['acq_order'].values
            return global_time.astype(float)
        return np.array(meta).flatten().astype(float)

