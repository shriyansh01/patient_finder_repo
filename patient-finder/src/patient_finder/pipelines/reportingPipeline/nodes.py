"""
This is a boilerplate pipeline 'reportingPipeline'
generated using Kedro 1.0.0
"""
#import pandas as pd
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from datetime import datetime
import warnings
import os
from matplotlib import image as mpimg

warnings.filterwarnings('ignore')

def generate_report(code_var_table, baseline_metrics, finetuned_metrics, 
                   important_features_path="data/02_intermediate/important_features.txt",
                   shap_plots_dir="data/02_intermediate/shap_plots",
                   output_path="docs/business_report.pdf",
                   company_name="Chryselys Services Private Limited",
                   report_title="Patient Finder Model Performance Report"):

    # Set style for professional look
    plt.style.use('default')
    sns.set_palette("husl")
    
    # --- Load Data ---
    code_var_table = code_var_table.sort_values(by="variance", ascending=False)
    
    with open(important_features_path, "r") as f:
        important_feats = [line.strip() for line in f.readlines()]
    
    
    # Create PDF
    with PdfPages(output_path) as pdf:
        
        # === PAGE 1: TITLE PAGE ===
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        gradient = np.linspace(0, 1, 256).reshape(256, -1)
        ax.imshow(gradient, extent=[0, 10, 0, 8], aspect='auto', cmap='Blues_r', alpha=0.3)
        
        ax.text(5, 6.5, report_title, fontsize=28, weight='bold', 
                ha='center', va='center', color='#1f4e79')
        ax.text(5, 5.2, f"Generated on {datetime.now().strftime('%B %d, %Y')}", 
                fontsize=14, ha='center', va='center', color='#666666')
        ax.text(5, 1, "CONFIDENTIAL - INTERNAL USE ONLY", 
                fontsize=10, ha='center', va='center', 
                style='italic', color='#999999')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # === PAGE 2: FEATURE IMPORTANCE ANALYSIS ===
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Top features bar chart
        ax1 = fig.add_subplot(gs[0, :])
        top_10_features = code_var_table.head(10)
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_10_features)))
        bars = ax1.barh(range(len(top_10_features)), top_10_features['variance'], color=colors)
        ax1.set_yticks(range(len(top_10_features)))
        ax1.set_yticklabels(top_10_features['code'], fontsize=10)
        ax1.set_xlabel('Variance Score', fontsize=12, weight='bold')
        ax1.set_title('Top Features by Variance', fontsize=10, weight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3)
        
        for i, (bar, value) in enumerate(zip(bars, top_10_features['variance'])):
            ax1.text(bar.get_width() + max(top_10_features['variance']) * 0.01, 
                    bar.get_y() + bar.get_height()/2, f'{value:.4f}', 
                    va='center', fontsize=9, weight='bold')
        
        # Feature overlap analysis
        ax2 = fig.add_subplot(gs[1, 0])
        intersect_features = [f for f in important_feats if f in code_var_table["code"].values]
        overlap_data = [len(intersect_features), len(important_feats) - len(intersect_features)]
        labels = ['In Analysis', 'Not in Analysis']
        colors_pie = ['#2E8B57', '#CD5C5C']
        
        ax2.pie(overlap_data, labels=labels, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90)
        ax2.set_title('Business-Important Features\nOverlap Analysis', 
                     fontsize=12, weight='bold')
        
        # Feature statistics
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        stats_text = f"""
FEATURE STATISTICS

Total Features Analyzed: {len(code_var_table)}
Business-Important Features: {len(important_feats)}
Overlap Count: {len(intersect_features)}
Overlap Percentage: {len(intersect_features)/len(important_feats)*100:.1f}%

Top 5 Business Features in Analysis:
"""
        for i, feat in enumerate(intersect_features[:5]):
            stats_text += f"{i+1}. {feat}\n"
        
        ax3.text(0.1, 0.9, stats_text, fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        plt.suptitle('FEATURE IMPORTANCE ANALYSIS', fontsize=15, weight='bold', y=0.95)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # === PAGE 3: MODEL PERFORMANCE COMPARISON ===
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
        
        models = list(baseline_metrics.keys())
        metrics_types = ['accuracy', 'f1', 'roc_auc']
        
        baseline_data = []
        finetuned_data = []
        
        for model in models:
            baseline_data.append([baseline_metrics[model][metric] for metric in metrics_types])
            finetuned_data.append([finetuned_metrics.get(model, {}).get(metric, 0) 
                                 for metric in metrics_types])
        
        baseline_df = pd.DataFrame(baseline_data, columns=metrics_types, index=models)
        finetuned_df = pd.DataFrame(finetuned_data, columns=metrics_types, index=models)
        
        # Bar chart comparison
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(metrics_types):
            ax1.bar(x + i*width, baseline_df[metric], width, 
                   label=f'Baseline {metric.upper()}', alpha=0.8)
            ax1.bar(x + i*width, finetuned_df[metric] - baseline_df[metric], width,
                   bottom=baseline_df[metric], label=f'Fine-tuned {metric.upper()}', alpha=0.8)
        
        ax1.set_xlabel('Models', fontsize=12, weight='bold')
        ax1.set_ylabel('Score', fontsize=12, weight='bold')
        ax1.set_title('Model Performance: Baseline vs Fine-tuned', fontsize=14, weight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([m.capitalize() for m in models])
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(axis='y', alpha=0.3)
        
        # Heatmap
        ax2 = fig.add_subplot(gs[1, 0])
        improvement_matrix = finetuned_df - baseline_df
        sns.heatmap(improvement_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0, ax=ax2, cbar_kws={'label': 'Improvement Score'})
        ax2.set_title('Performance Improvement Heatmap', fontsize=12, weight='bold')
        
        # Best model summary
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')
        
        best_model = max(finetuned_metrics.items(), key=lambda x: x[1]["accuracy"])[0]
        best_metrics = finetuned_metrics[best_model]
        
        summary_text = f"""
BEST PERFORMING MODEL

Model: {best_model.upper()}

Performance Metrics:
• Accuracy: {best_metrics['accuracy']:.3f}
• F1 Score: {best_metrics['f1']:.3f}  
• AUC Score: {best_metrics['roc_auc']:.3f}

Improvements over Baseline:
• Accuracy: +{best_metrics['accuracy'] - baseline_metrics[best_model]['accuracy']:.3f}
• F1 Score: +{best_metrics['f1'] - baseline_metrics[best_model]['f1']:.3f}
• AUC Score: +{best_metrics['roc_auc'] - baseline_metrics[best_model]['roc_auc']:.3f}
        """
        
        ax3.text(0.1, 0.9, summary_text, fontsize=11, va='top', ha='left',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.7))
        
        plt.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=20, weight='bold', y=0.95)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # === PAGE 4: SHAP FEATURE EXPLANATIONS ===
        if shap_plots_dir and os.path.exists(shap_plots_dir):
            shap_pngs = [
                "shap_bar_decision_tree.png",
                "shap_bar_lightgbm.png",
                "shap_bar_logistic_regression.png",
                "shap_bar_random_forest.png",
                "shap_bar_xgboost.png",
                "shap_summary_decision_tree.png",
                "shap_summary_lightgbm.png",
                "shap_summary_logistic_regression.png",
                "shap_summary_random_forest.png",
                "shap_summary_xgboost.png",
                "shap_dependence_lightgbm.png",
                "shap_dependence_logistic_regression.png",
                "shap_dependence_xgboost.png"
            ]

            for shap_file in shap_pngs:
                shap_path = os.path.join(shap_plots_dir, shap_file)
                if os.path.exists(shap_path):
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    img = mpimg.imread(shap_path)
                    ax.imshow(img)
                    ax.axis('off')
                    ax.set_title(f"SHAP Explanation: {shap_file.replace('_',' ').replace('.png','').title()}",
                                 fontsize=14, weight='bold')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        
        # === FINAL PAGE: RECOMMENDATIONS & CONCLUSIONS ===
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        
        avg_improvement = np.mean([finetuned_metrics[m]['accuracy'] - baseline_metrics[m]['accuracy'] 
                                 for m in models])
        best_feature = code_var_table.iloc[0]['code']
        
        ax.text(0.05, 0.9, "RECOMMENDATIONS & CONCLUSIONS", fontsize=16, weight='bold', color='#1f4e79')
        ax.text(0.05, 0.75, f"• On average, fine-tuning improved accuracy by {avg_improvement:.3f}.", fontsize=12)
        ax.text(0.05, 0.65, f"• Feature '{best_feature}' showed the highest variance importance.", fontsize=12)
        ax.text(0.05, 0.55, "• SHAP analysis highlights key drivers of model predictions,\n"
                             "   supporting interpretability and business decision-making.", fontsize=12)
        ax.text(0.05, 0.45, "• Recommended next steps: validate top features with domain experts\n"
                             "   and assess fairness across patient subgroups.", fontsize=12)
        
        ax.text(0.5, 0.02, f"Report generated by {company_name} Data Science Team | {datetime.now().strftime('%Y-%m-%d')}",
                fontsize=9, ha='center', va='bottom', style='italic', color='#666666')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF report generated: {output_path}")
    output = {"path": output_path, "report_generated": True}
    
    return output
