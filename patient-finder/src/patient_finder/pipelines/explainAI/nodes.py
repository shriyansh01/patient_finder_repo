"""
This is a boilerplate pipeline 'explainAI'
generated using Kedro 1.0.0
"""
import shap
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import os


def explain_models(models, features: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Generate SHAP explanations for trained models with multiple plot types.
    
    Args:
        models (dict): Trained models from train_classical_models.
        features (pd.DataFrame): Feature set (without target/patient_id).
    
    Returns:
        Dict[str, List[str]]: File paths of SHAP plots for each model.
    """
    explanations = {}
    data = features.copy()
    if "target" in data.columns:
        data.drop(columns=["target"], inplace=True)
    if "patient_id" in data.columns:
        data.drop(columns=["patient_id"], inplace=True)
    data.fillna(0, inplace=True)

    # use small background set for speed
    background = shap.sample(data, 200)

    # ensure folders exist
    os.makedirs("data/02_intermediate/shap_plots", exist_ok=True)
    os.makedirs("data/08_reporting/shap_plots", exist_ok=True)

    for name, model in models.items():
        plot_paths = []
        try:
            explainer = shap.Explainer(model, background)
            shap_values = explainer(data)

            # --- Summary plot (dot) ---
            plt.figure()
            shap.summary_plot(shap_values, data, show=False)
            out_path = f"data/08_reporting/shap_plots/shap_summary_{name}.png"
            inter_path = f"data/02_intermediate/shap_plots/shap_summary_{name}.png"
            plt.savefig(inter_path, bbox_inches="tight")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            plot_paths.append(out_path)

            # --- Summary plot (bar) ---
            plt.figure()
            shap.summary_plot(shap_values, data, plot_type="bar", show=False)
            out_path = f"data/08_reporting/shap_plots/shap_bar_{name}.png"
            inter_path = f"data/02_intermediate/shap_plots/shap_bar_{name}.png"
            plt.savefig(inter_path, bbox_inches="tight")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            plot_paths.append(out_path)

            # --- Dependence plot (top feature) ---
            top_feature = data.columns[abs(shap_values.values).mean(0).argmax()]
            plt.figure()
            shap.dependence_plot(top_feature, shap_values.values, data, show=False)
            out_path = f"data/08_reporting/shap_plots/shap_dependence_{name}.png"
            inter_path = f"data/02_intermediate/shap_plots/shap_dependence_{name}.png"
            plt.savefig(inter_path, bbox_inches="tight")
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            plot_paths.append(out_path)

            # --- Force plot (single prediction example) ---
            force_out_path = f"data/08_reporting/shap_plots/shap_force_{name}.html"
            force_inter_path = f"data/02_intermediate/shap_plots/shap_force_{name}.html"
            # just pick first instance for demo
            force_plot = shap.plots.force(explainer.expected_value, 
                                          shap_values.values[0,:], 
                                          data.iloc[0,:], 
                                          matplotlib=False)
            shap.save_html(force_inter_path, force_plot)
            shap.save_html(force_out_path, force_plot)
            plot_paths.append(force_out_path)

            explanations[name] = plot_paths

        except Exception as e:
            explanations[name] = [f"Failed: {str(e)}"]

    return explanations
