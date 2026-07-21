"""Synthetic data engine for multivariate batch-effect correction benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

        
        
class SimpleClinicalSyntheticGCIMSEngine:
    """
    Injects synthetic clinical biomarker signals into an empirical GC-IMS baseline
    (e.g., pooled urine samples) using a multiplicative antagonistic model.
    
    Clinical Scenario ("Antagonistic" Strict Mode):
      - Group A receives an upregulation scaling (baseline multiplied by effect_size).
      - Group B receives a downregulation scaling (baseline divided by effect_size).
      - Both groups share the exact same active biomarker features to maintain consistency.
          
    Biological Variance & Background Noise:
      - Patient-to-patient heterogeneity is applied as Gaussian noise directly to 
        the scaling multiplier of the active biomarkers, creating realistic biological spread.
      - A very small, independent Gaussian noise is applied to all non-biomarker features 
        to simulate natural, class-specific background variations.
    """

    # Interleaved deterministic patterns for 15-sample batches
    LAYOUT_PATTERNS = {
        3: ["QC", "A", "B", "A", "B", "A", "B", "QC", "A", "B", "A", "B", "A", "B", "QC"],
        5: ["QC", "A", "B", "A", "QC", "B", "A", "B", "QC", "A", "B", "A", "QC", "B", "QC"],
        7: ["QC", "A", "B", "QC", "A", "QC", "B", "QC", "A", "QC", "B", "A", "QC", "B", "QC"]
    }

    def __init__(
        self,
        X_raw: np.ndarray,
        batch_labels: np.ndarray,
        run_order: np.ndarray,
        qc_count_per_batch: int = 3,
        n_biomarkers: int = 1,
        target_feature_idx: int = None,
        effect_size: float = 1.05,           # Scaling magnitude
        biological_variance: float = 0.20,  # Spread of the effect_size across different patients
        background_noise_std: float = 0.01, # Very small Gaussian noise added to non-biomarker features
        random_state: int = 42
    ):
        self.X_raw = np.asarray(X_raw, dtype=float)
        self.batch_labels = np.asarray(batch_labels, dtype=int)
        self.run_order = np.asarray(run_order, dtype=int)
        
        if qc_count_per_batch not in self.LAYOUT_PATTERNS:
            raise ValueError("qc_count_per_batch must be exactly 3, 5, or 7.")
        if not (1 <= n_biomarkers <= self.X_raw.shape[1]):
            raise ValueError(f"n_biomarkers must be between 1 and {self.X_raw.shape[1]}.")
            
        self.qc_count = qc_count_per_batch
        self.n_biomarkers = n_biomarkers
        self.target_feature_idx = target_feature_idx
        self.effect_size = float(effect_size)
        self.bio_var = float(biological_variance)
        self.background_noise_std = float(background_noise_std)
        self.rng = np.random.default_rng(random_state)
        
        self.N, self.P = self.X_raw.shape

    def _allocate_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Maps the layout pattern to the empirical data strictly by acquisition order."""
        sample_type = np.full(self.N, "Unknown", dtype=object)
        group_col = np.full(self.N, "QC", dtype=object)
        
        pattern = self.LAYOUT_PATTERNS[self.qc_count]
        unique_batches = np.unique(self.batch_labels)
        
        for b in unique_batches:
            idx_in_batch = np.where(self.batch_labels == b)[0]
            sorted_idx = idx_in_batch[np.argsort(self.run_order[idx_in_batch])]
            
            if len(sorted_idx) != 15:
                raise ValueError(f"Batch {b} does not contain exactly 15 samples.")
                
            for i, assignment in enumerate(pattern):
                global_idx = sorted_idx[i]
                if assignment == "QC":
                    sample_type[global_idx] = "QC"
                    group_col[global_idx] = "QC"
                else:
                    sample_type[global_idx] = "Patient"
                    group_col[global_idx] = assignment

        qc_indices = np.where(sample_type == "QC")[0]
        patient_indices = np.where(sample_type == "Patient")[0]
        
        return sample_type, group_col, qc_indices, patient_indices

    def generate(self) -> dict:
        sample_type, group_col, qc_indices, patient_indices = self._allocate_samples()
        
        # 1. Select Active Biomarkers and create spatial masks
        if self.target_feature_idx is not None:
            active_features = np.array([self.target_feature_idx])
        else:
            active_features = self.rng.choice(self.P, size=self.n_biomarkers, replace=False)
        
        active_mask = np.zeros(self.P)
        active_mask[active_features] = 1.0
        
        inactive_mask = 1.0 - active_mask # Mask for all non-biomarker features
        
        # 2. Build the Clinical Direction Vector
        # We generate values only for the active features.
        v_clin = self.rng.normal(1.0, 0.2, size=self.P) * active_mask
        
        # Normalize v_clin so its maximum absolute magnitude is exactly 1.0.
        # This guarantees the target features hit the exact peak defined by effect_size.
        max_val = np.max(np.abs(v_clin))
        if max_val > 0:
            v_clin = v_clin / max_val

        # 3. Initialize Matrices
        X_spiked = self.X_raw.copy()
        X_clin_pure = np.zeros((self.N, self.P)) # Stores the absolute clinical variation added
        
        # 4. Multiplicative Signal Injection
        for i in range(self.N):
            if group_col[i] in ["A", "B"]:
                # Generate specific noise distributions for this patient
                patient_bio_noise = self.rng.normal(0.0, self.bio_var, size=self.P) * active_mask
                patient_bg_noise = self.rng.normal(0.0, self.background_noise_std, size=self.P) * inactive_mask
                
                if group_col[i] == "A":
                    # Upregulation: Baseline multiplied by effect_size
                    multiplier = 1.0 + (self.effect_size - 1.0) * v_clin
                else: # group_col[i] == "B"
                    # Downregulation: Baseline divided by effect_size
                    multiplier = 1.0 + ((1.0 / self.effect_size) - 1.0) * v_clin
                
                # Combine the core effect with the biological variance (active features) 
                # and the small background noise (inactive features)
                final_scaling_factor = multiplier + patient_bio_noise + patient_bg_noise
                
                # Prevent negative intensities caused by excessive down-scaling or noise
                final_scaling_factor = np.clip(final_scaling_factor, 0.0, None)
                
                # Apply the transformation to the empirical data
                X_spiked[i, :] = self.X_raw[i, :] * final_scaling_factor
                X_clin_pure[i, :] = X_spiked[i, :] - self.X_raw[i, :]

        patient_groups = group_col[patient_indices]
        group_labels = np.where(patient_groups == "A", 0, 1)

        return {
            "X": X_spiked,
            "X_raw_baseline": self.X_raw.copy(),
            "X_clin_pure": X_clin_pure,
            "X_qc": X_spiked[qc_indices].copy(),
            "X_patients": X_spiked[patient_indices].copy(),
            "sample_type": sample_type,
            "group_col": group_col,
            "qc_indices": qc_indices,
            "patient_indices": patient_indices,
            "batch_labels_qc": self.batch_labels[qc_indices].copy(),
            "run_order_qc": self.run_order[qc_indices].copy(),
            "batch_labels_patients": self.batch_labels[patient_indices].copy(),
            "run_order_patients": self.run_order[patient_indices].copy(),
            "group_labels": group_labels,
            "spike_metadata": {
                "clinical_scenario": "antagonistic",
                "qc_count_per_batch": self.qc_count,
                "n_biomarkers": self.n_biomarkers,
                "effect_size": self.effect_size,
                "biological_variance": self.bio_var,
                "background_noise_std": self.background_noise_std,
                "active_features": active_features.tolist()
            }
        }