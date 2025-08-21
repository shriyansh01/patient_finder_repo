# """
# This is a boilerplate pipeline 'edaPipeline'
# generated using Kedro 1.0.0
# """
# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import warnings
# from matplotlib.backends.backend_pdf import PdfPages
# from matplotlib.gridspec import GridSpec
# import io
# import base64

# warnings.filterwarnings('ignore')

# def create_eda_pdf_report(positive_cohort, negative_cohort, output_filename='docs/claims_eda_report.pdf'):
#     claims_df = pd.concat([positive_cohort, negative_cohort], ignore_index=True)
#     """
#     Generate comprehensive EDA report for claims table and save as PDF
    
#     Parameters:
#     claims_df (pd.DataFrame): Claims data with columns: 
#                              patient_id, age, gender, service_date, code, code_type, target
#     output_filename (str): Name of the output PDF file
    
#     Returns:
#     dict: Dictionary containing all EDA results
#     """
    
#     # Make a copy to avoid modifying original data
#     df = claims_df.copy()
#     #target astype int
#     df['target'] = df['target'].astype(int)
#     # Convert service_date to datetime if it's not already
#     if not pd.api.types.is_datetime64_any_dtype(df['service_date']):
#         df['service_date'] = pd.to_datetime(df['service_date'])
    
#     # Initialize results dictionary
#     eda_results = {}
    
#     # Set up the PDF
#     with PdfPages(output_filename) as pdf:
        
#         # PAGE 1: TITLE PAGE AND OVERVIEW
#         fig = plt.figure(figsize=(8.5, 11))
#         fig.suptitle('CLAIMS TABLE\nEXPLORATORY DATA ANALYSIS REPORT', 
#                      fontsize=20, fontweight='bold', y=0.85)
        
#         # Add report metadata
#         plt.text(0.5, 0.7, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
#                  ha='center', fontsize=12, transform=fig.transFigure)
        
#         # Basic dataset information
#         basic_info_text = f"""
# DATASET OVERVIEW
# {'='*50}

# Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns
# Date Range: {df['service_date'].min().date()} to {df['service_date'].max().date()}
# Unique Patients: {df['patient_id'].nunique():,}
# Total Claims: {len(df):,}
# Analysis Period: {(df['service_date'].max() - df['service_date'].min()).days} days

# DATA QUALITY SUMMARY
# {'='*50}
# """
        
#         # Missing values summary
#         missing_data = df.isnull().sum()
#         if missing_data.sum() > 0:
#             basic_info_text += "Missing Values Detected:\n"
#             for col, missing in missing_data.items():
#                 if missing > 0:
#                     basic_info_text += f"  • {col}: {missing} ({missing/len(df)*100:.1f}%)\n"
#         else:
#             basic_info_text += " No missing values detected\n"
        
#         # Data consistency check
#         age_consistency = df.groupby('patient_id')['age'].nunique().max()
#         if age_consistency > 1:
#             basic_info_text += "⚠️  Some patients have multiple ages recorded\n"
#         else:
#             basic_info_text += " Patient age data appears consistent\n"
        
#         plt.text(0.1, 0.6, basic_info_text, ha='left', va='top', fontsize=10, 
#                  transform=fig.transFigure, fontfamily='monospace')
        
#         plt.axis('off')
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
        
#         # Store basic info
#         eda_results['basic_info'] = {
#             'shape': df.shape,
#             'date_range': (df['service_date'].min(), df['service_date'].max()),
#             'unique_patients': df['patient_id'].nunique(),
#             'total_claims': len(df),
#             'missing_values': missing_data
#         }
        
#         # PAGE 2: TARGET VARIABLE AND DEMOGRAPHICS
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
#         fig.suptitle('Target Variable and Demographics Analysis', fontsize=16, fontweight='bold')
        
#         # Target distribution
#         target_dist = df['target'].value_counts()
#         target_pct = df['target'].value_counts(normalize=True) * 100
#         colors = ['lightcoral' if x else 'lightblue' for x in target_dist.index]
#         bars = ax1.bar([str(x) for x in target_dist.index], target_dist.values, color=colors)
#         ax1.set_title('Target Distribution', fontweight='bold')
#         ax1.set_xlabel('Target')
#         ax1.set_ylabel('Count')
        
#         # Add percentage labels on bars
#         for bar, pct in zip(bars, target_pct.values):
#             height = bar.get_height()
#             ax1.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{pct:.1f}%', ha='center', va='bottom')
        
#         # Age distribution
#         ax2.hist(df['age'], bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
#         ax2.set_title('Age Distribution', fontweight='bold')
#         ax2.set_xlabel('Age')
#         ax2.set_ylabel('Frequency')
#         ax2.axvline(df['age'].mean(), color='red', linestyle='--', 
#                    label=f'Mean: {df["age"].mean():.1f}')
#         ax2.legend()
        
#         # Gender distribution
#         gender_dist = df['gender'].value_counts()
#         colors_gender = ['lightblue', 'pink', 'lightgray'][:len(gender_dist)]
#         wedges, texts, autotexts = ax3.pie(gender_dist.values, labels=gender_dist.index, 
#                                           autopct='%1.1f%%', colors=colors_gender)
#         ax3.set_title('Gender Distribution', fontweight='bold')
        
#         # Age groups analysis
#         df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 65, 100], labels=['<30', '30-50', '50-65', '65+'])
#         age_group_dist = df['age_group'].value_counts().sort_index()
#         ax4.bar(range(len(age_group_dist)), age_group_dist.values, color='gold', alpha=0.7)
#         ax4.set_title('Age Group Distribution', fontweight='bold')
#         ax4.set_xlabel('Age Group')
#         ax4.set_ylabel('Count')
#         ax4.set_xticks(range(len(age_group_dist)))
#         ax4.set_xticklabels(age_group_dist.index)
        
#         plt.tight_layout()
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
        
#         # Store demographics data
#         eda_results['demographics'] = {
#             'age_stats': df['age'].describe(),
#             'gender_distribution': gender_dist,
#             'target_distribution': target_dist
#         }
        
#         # PAGE 3: CODE ANALYSIS
#         fig = plt.figure(figsize=(11, 8.5))
#         gs = GridSpec(3, 2, figure=fig)
#         fig.suptitle('Code Type and Temporal Analysis', fontsize=16, fontweight='bold')
        
#         # Code type distribution
#         ax1 = fig.add_subplot(gs[0, 0])
#         code_type_dist = df['code_type'].value_counts()
#         bars = ax1.bar(code_type_dist.index, code_type_dist.values, color='orange', alpha=0.7)
#         ax1.set_title('Code Type Distribution', fontweight='bold')
#         ax1.set_xlabel('Code Type')
#         ax1.set_ylabel('Count')
        
#         # Add count labels on bars
#         for bar in bars:
#             height = bar.get_height()
#             ax1.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{int(height)}', ha='center', va='bottom')
        
#         # Top codes overall
#         ax2 = fig.add_subplot(gs[0, 1])
#         top_codes = df['code'].value_counts().head(10)
#         ax2.barh(range(len(top_codes)), top_codes.values, color='teal', alpha=0.7)
#         ax2.set_title('Top 10 Most Common Codes', fontweight='bold')
#         ax2.set_xlabel('Count')
#         ax2.set_yticks(range(len(top_codes)))
#         ax2.set_yticklabels(top_codes.index)
        
#         # Temporal analysis - yearly
#         ax3 = fig.add_subplot(gs[1, :])
#         df['year'] = df['service_date'].dt.year
#         yearly_claims = df['year'].value_counts().sort_index()
#         ax3.plot(yearly_claims.index, yearly_claims.values, marker='o', linewidth=2, 
#                 markersize=8, color='purple')
#         ax3.set_title('Claims Trend by Year', fontweight='bold')
#         ax3.set_xlabel('Year')
#         ax3.set_ylabel('Number of Claims')
#         ax3.grid(True, alpha=0.3)
        
#         # Monthly pattern
#         ax4 = fig.add_subplot(gs[2, :])
#         df['month'] = df['service_date'].dt.month
#         monthly_claims = df['month'].value_counts().sort_index()
#         month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#         bars = ax4.bar([month_names[i-1] for i in monthly_claims.index], 
#                       monthly_claims.values, color='darkgreen', alpha=0.7)
#         ax4.set_title('Claims Distribution by Month', fontweight='bold')
#         ax4.set_xlabel('Month')
#         ax4.set_ylabel('Number of Claims')
        
#         plt.tight_layout()
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
        
#         # Store code analysis
#         eda_results['code_analysis'] = {
#             'code_type_distribution': code_type_dist,
#             'top_codes': top_codes,
#             'temporal_patterns': {'yearly': yearly_claims, 'monthly': monthly_claims}
#         }
        
#         # PAGE 4: TARGET RELATIONSHIPS
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
#         fig.suptitle('Target Variable Relationships', fontsize=16, fontweight='bold')
        
#         # Target by gender
#         target_by_gender = df.groupby('gender')['target'].agg(['count', 'sum', 'mean'])
#         ax1.bar(target_by_gender.index, target_by_gender['mean'], color='coral', alpha=0.7)
#         ax1.set_title('Target Rate by Gender', fontweight='bold')
#         ax1.set_xlabel('Gender')
#         ax1.set_ylabel('Target Rate')
#         ax1.set_ylim(0, max(target_by_gender['mean']) * 1.2)
        
#         # Add percentage labels
#         for i, (gender, rate) in enumerate(target_by_gender['mean'].items()):
#             ax1.text(i, rate + max(target_by_gender['mean']) * 0.02,
#                     f'{rate*100:.2f}%', ha='center', va='bottom')
        
#         # Target by code type
#         target_by_code_type = df.groupby('code_type')['target'].agg(['count', 'sum', 'mean'])
#         bars = ax2.bar(target_by_code_type.index, target_by_code_type['mean'], 
#                       color='lightseagreen', alpha=0.7)
#         ax2.set_title('Target Rate by Code Type', fontweight='bold')
#         ax2.set_xlabel('Code Type')
#         ax2.set_ylabel('Target Rate')
        
#         # Add percentage labels
#         for bar, rate in zip(bars, target_by_code_type['mean']):
#             height = bar.get_height()
#             ax2.text(bar.get_x() + bar.get_width()/2., height,
#                     f'{rate*100:.2f}%', ha='center', va='bottom')
        
#         # Target by age group
#         target_by_age = df.groupby('age_group')['target'].agg(['count', 'sum', 'mean'])
#         ax3.bar(range(len(target_by_age)), target_by_age['mean'], color='gold', alpha=0.7)
#         ax3.set_title('Target Rate by Age Group', fontweight='bold')
#         ax3.set_xlabel('Age Group')
#         ax3.set_ylabel('Target Rate')
#         ax3.set_xticks(range(len(target_by_age)))
#         ax3.set_xticklabels(target_by_age.index)
        
#         # Claims per patient distribution
#         claims_per_patient = df.groupby('patient_id').size()
#         ax4.hist(claims_per_patient.values, bins=20, color='mediumpurple', alpha=0.7, edgecolor='black')
#         ax4.set_title('Claims per Patient Distribution', fontweight='bold')
#         ax4.set_xlabel('Number of Claims per Patient')
#         ax4.set_ylabel('Number of Patients')
#         ax4.axvline(claims_per_patient.mean(), color='red', linestyle='--',
#                    label=f'Mean: {claims_per_patient.mean():.1f}')
#         ax4.legend()
        
#         plt.tight_layout()
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
        
#         # Store relationship analysis
#         eda_results['target_relationships'] = {
#             'by_gender': target_by_gender,
#             'by_code_type': target_by_code_type,
#             'by_age_group': target_by_age,
#             'claims_per_patient': claims_per_patient.describe()
#         }
        
#         # PAGE 5: DETAILED STATISTICS TABLES
#         fig = plt.figure(figsize=(8.5, 11))
#         fig.suptitle('Detailed Statistical Summary', fontsize=16, fontweight='bold')
        
#         # Create text summary
#         summary_text = f"""
# STATISTICAL SUMMARY
# {'='*80}

# BASIC STATISTICS
# {'-'*40}
# Total Records: {len(df):,}
# Unique Patients: {df['patient_id'].nunique():,}
# Date Range: {df['service_date'].min().date()} to {df['service_date'].max().date()}
# Analysis Period: {(df['service_date'].max() - df['service_date'].min()).days} days

# TARGET VARIABLE
# {'-'*40}
# Positive Target Rate: {df['target'].mean()*100:.2f}%
# Total Positive Cases: {df['target'].sum():,}
# Total Negative Cases: {(~df['target']).sum():,}

# AGE STATISTICS
# {'-'*40}
# Mean Age: {df['age'].mean():.1f} years
# Median Age: {df['age'].median():.1f} years
# Min Age: {df['age'].min()} years
# Max Age: {df['age'].max()} years
# Standard Deviation: {df['age'].std():.1f} years

# GENDER DISTRIBUTION
# {'-'*40}"""
        
#         for gender, count in df['gender'].value_counts().items():
#             pct = count / len(df) * 100
#             summary_text += f"\n{gender}: {count:,} ({pct:.1f}%)"
        
#         summary_text += f"""

# CODE TYPE DISTRIBUTION
# {'-'*40}"""
        
#         for code_type, count in df['code_type'].value_counts().items():
#             pct = count / len(df) * 100
#             summary_text += f"\n{code_type}: {count:,} ({pct:.1f}%)"
        
#         summary_text += f"""

# PATIENT-LEVEL STATISTICS
# {'-'*40}
# Average Claims per Patient: {claims_per_patient.mean():.1f}
# Median Claims per Patient: {claims_per_patient.median():.1f}
# Max Claims per Patient: {claims_per_patient.max()}
# Patients with Positive Target: {df[df['target'] == True]['patient_id'].nunique():,}
# Patient Target Rate: {df[df['target'] == True]['patient_id'].nunique()/df['patient_id'].nunique()*100:.1f}%

# TOP 10 MOST COMMON CODES
# {'-'*40}"""
        
#         for i, (code, count) in enumerate(df['code'].value_counts().head(10).items(), 1):
#             summary_text += f"\n{i:2}. {code}: {count:,} claims"
        
#         summary_text += f"""

# DATA QUALITY OBSERVATIONS
# {'-'*40}"""
        
#         if missing_data.sum() > 0:
#             summary_text += "\n⚠️  Missing values detected:"
#             for col, missing in missing_data.items():
#                 if missing > 0:
#                     summary_text += f"\n   • {col}: {missing} ({missing/len(df)*100:.1f}%)"
#         else:
#             summary_text += "\n No missing values detected"
        
#         if age_consistency > 1:
#             summary_text += "\n⚠️  Some patients have multiple ages recorded"
#         else:
#             summary_text += "\n Patient age data appears consistent"
        
#         # Add duplicate check
#         duplicates = df.duplicated().sum()
#         if duplicates > 0:
#             summary_text += f"\n⚠️  {duplicates} duplicate rows detected"
#         else:
#             summary_text += "\n No duplicate rows detected"
        
#         plt.text(0.05, 0.95, summary_text, ha='left', va='top', fontsize=9, 
#                  transform=fig.transFigure, fontfamily='monospace')
#         plt.axis('off')
#         pdf.savefig(fig, bbox_inches='tight')
#         plt.close()
    
#     print(f" PDF EDA report generated successfully: {output_filename}")
#     print(f"📊 Report includes {pdf.get_pagecount() if 'pdf' in locals() else '5'} pages of analysis")
#     output_json = {
#         "Report_name" : "EDA report created",
#         "Report_path": output_filename,
#     }
#     return output_json


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

def create_eda_pdf_report(positive_cohort, negative_cohort, output_filename='docs/claims_eda_report.pdf'):
    # Combine datasets
    df = pd.concat([positive_cohort, negative_cohort], ignore_index=True).copy()
    df['target'] = df['target'].astype(int)
    df['service_date'] = pd.to_datetime(df['service_date'])
    
    # Basic info
    missing_data = df.isnull().sum()
    duplicates = df.duplicated().sum()
    unique_patients = df['patient_id'].nunique()
    
    with PdfPages(output_filename) as pdf:
        # PAGE 1: Overview
        fig, ax = plt.subplots(figsize=(8.5, 11))
        overview_text = f"""
CLAIMS TABLE EDA REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset Shape: {df.shape}
Unique Patients: {unique_patients}
Date Range: {df['service_date'].min().date()} - {df['service_date'].max().date()}

Missing Values:\n{missing_data[missing_data > 0] if missing_data.sum() > 0 else 'None'}
Duplicate Rows: {duplicates}
"""
        ax.text(0.05, 0.95, overview_text, fontsize=12, va='top', fontfamily='monospace')
        ax.axis('off')
        pdf.savefig(fig)
        plt.close()

        # PAGE 2: Target & Demographics
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Target and Demographics', fontsize=16, fontweight='bold')

        # Target distribution
        target_counts = df['target'].value_counts()
        axes[0,0].bar(target_counts.index, target_counts.values, color=['lightblue','salmon'])
        axes[0,0].set_title('Target Distribution')

        # Age distribution
        axes[0,1].hist(df['age'], bins=20, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Age Distribution')

        # Gender distribution
        gender_counts = df['gender'].value_counts()
        axes[1,0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue','pink','gray'])
        axes[1,0].set_title('Gender Distribution')

        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0,30,50,65,100], labels=['<30','30-50','50-65','65+'])
        age_group_counts = df['age_group'].value_counts().sort_index()
        axes[1,1].bar(age_group_counts.index, age_group_counts.values, color='gold')
        axes[1,1].set_title('Age Group Distribution')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

        # PAGE 3: Code analysis
        fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Code Analysis', fontsize=16, fontweight='bold')

        # Code type
        code_type_counts = df['code_type'].value_counts()
        axes[0].bar(code_type_counts.index, code_type_counts.values, color='orange')
        axes[0].set_title('Code Type Distribution')

        # Top 10 codes
        top_codes = df['code'].value_counts().head(10)
        axes[1].barh(top_codes.index[::-1], top_codes.values[::-1], color='teal')
        axes[1].set_title('Top 10 Codes')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()

    print(f"PDF EDA report generated: {output_filename}")
    return {"Report_name": "EDA report created", "Report_path": output_filename}
