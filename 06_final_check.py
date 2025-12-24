# -*- coding: utf-8 -*-
"""
Final Validation Check: Predictions vs Known Ground Truth
=========================================================

GOAL:
This script performs a final sanity check by comparing the unsupervised
'mixture_score' against the known ground truth labels (specific samples
known to be pools in this study: SRR17037618-23).

OUTPUTS:
- A validation plot (Bar chart with precision score) saved in:
  <results_dir>/gauss_euler/figures/main/Validation_Score_vs_Truth.png

USAGE EXAMPLE:
  python 06_final_check.py --results_dir "./results"
"""
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ground Truth for this specific study ---
# Samples known to be pools (SRR17037618 ~ SRR17037623)
TRUE_POOLS = [
    'SRR17037618', 'SRR17037619', 'SRR17037620', 
    'SRR17037621', 'SRR17037622', 'SRR17037623'
]

def run_validation(results_dir):
    # Path to S2 Excel file
    excel_path = os.path.join(results_dir, "gauss_euler", "Supplementary_Data_S2_GaussEuler.xlsx")
    
    print(f"[Loading] Reading data from: {excel_path}")
    
    if not os.path.exists(excel_path):
        print(f"[Error] File not found: {excel_path}")
        print("Please run '02_run_geometry_analysis.py' first.")
        return

    # Read 'mixture_ranking' sheet
    try:
        df = pd.read_excel(excel_path, sheet_name='mixture_ranking')
    except Exception as e:
        print(f"[Error] Could not read Excel sheet: {e}")
        return

    # Mark Ground Truth
    df['is_true_pool'] = df['sample'].apply(lambda x: 'Pool (Ground Truth)' if x in TRUE_POOLS else 'Individual')
    
    # Plotting: Top 10 Ranking
    plt.figure(figsize=(12, 6))
    
    plot_data = df.head(10).copy()
    
    sns.barplot(
        data=plot_data, 
        x='sample', 
        y='mixture_score', 
        hue='is_true_pool', 
        dodge=False, 
        palette={'Pool (Ground Truth)': 'red', 'Individual': 'grey'}
    )
    
    # Title & Labels (Removed specific Fig number for publication safety)
    plt.title("Validation: Mathematical Curvature Score vs. Biological Ground Truth", fontsize=14)
    plt.ylabel("Mixture / Bridge Score (Ricci Curvature)")
    plt.xlabel("Sample ID (Ranked by Score)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add Precision Text (Top 3)
    top3 = df.head(3)['sample'].values
    hits = sum([1 for s in top3 if s in TRUE_POOLS])
    
    plt.text(0.5, 0.8, f"Top 3 Precision: {hits}/3 ({hits/3*100:.0f}%)", 
             transform=plt.gca().transAxes, fontsize=15, color='red', fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))

    # Save Figure
    fig_dir = os.path.join(results_dir, "gauss_euler", "figures", "main")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    save_path_png = os.path.join(fig_dir, "Validation_Score_vs_Truth.png")
    save_path_pdf = os.path.join(fig_dir, "Validation_Score_vs_Truth.pdf")
    
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=300)
    plt.savefig(save_path_pdf)
    
    print(f"\n[Validation] SUCCESS! Figure Saved:")
    print(f" -> {save_path_png}")
    print("Check if the RED bars (Pools) are ranked at the top!")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="./results", help="Base results directory (default: ./results)")
    args = ap.parse_args()

    run_validation(args.results_dir)

if __name__ == "__main__":
    main()