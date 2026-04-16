"""Generate SHAP explanations for the EvidenceChain ensemble."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from src.models.ensemble import StackingEnsemble


def explain_ensemble(dataset=None):
    """Create SHAP summary artifacts for a trained ensemble."""

    dataset = config._normalize_dataset_name(dataset)
    feature_path = os.path.join(
        config.TABLES_DIR, f"ensemble_features_{dataset}.csv"
    )
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Missing feature table: {feature_path}. "
            "Train the ensemble before running SHAP."
        )

    ensemble = StackingEnsemble()
    ensemble.load(config.get_ensemble_path(dataset))

    features_df = pd.read_csv(feature_path)
    X = features_df[ensemble.feature_names].to_numpy(dtype=float)
    X_scaled = ensemble.scaler.transform(X)

    explainer = shap.LinearExplainer(ensemble.classifier, X_scaled)
    shap_values = explainer(X_scaled)

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    summary_plot_path = os.path.join(
        config.PLOTS_DIR, f"shap_summary_{dataset}.png"
    )
    importance_path = os.path.join(
        config.TABLES_DIR, f"shap_importance_{dataset}.csv"
    )

    plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": ensemble.feature_names,
            "mean_abs_shap": importance,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(importance_path, index=False)

    print(f"Saved SHAP summary plot to {summary_plot_path}")
    print(f"Saved SHAP feature ranking to {importance_path}")
    return importance_df


if __name__ == "__main__":
    explain_ensemble()
