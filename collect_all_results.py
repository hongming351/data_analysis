import os
import shutil
import glob

# Define the base directory
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "all_results")

# Remove existing output directory if present
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# ===== KNN Results =====
knn_src = os.path.join(base_dir, "KNN", "output")
knn_dst = os.path.join(output_dir, "KNN")
if os.path.exists(knn_src):
    shutil.copytree(knn_src, knn_dst)
    print(f"Copied KNN results -> {knn_dst}")

# ===== MLP Results =====
mlp_src = os.path.join(base_dir, "MLP", "results")
mlp_dst = os.path.join(output_dir, "MLP")
if os.path.exists(mlp_src):
    shutil.copytree(mlp_src, mlp_dst)
    print(f"Copied MLP results -> {mlp_dst}")
# Also copy comparison_report.md
mlp_report_src = os.path.join(base_dir, "MLP", "comparison_report.md")
mlp_report_dst = os.path.join(output_dir, "MLP", "comparison_report.md")
if os.path.exists(mlp_report_src):
    os.makedirs(os.path.dirname(mlp_report_dst), exist_ok=True)
    shutil.copy2(mlp_report_src, mlp_report_dst)
    print(f"Copied MLP comparison_report.md -> {mlp_report_dst}")

# ===== RANDOMFOREST Results =====
rf_dst = os.path.join(output_dir, "RANDOMFOREST")
os.makedirs(rf_dst, exist_ok=True)

# Main figures
rf_figures_src = os.path.join(base_dir, "RANDOMFOREST", "figures")
rf_figures_dst = os.path.join(rf_dst, "figures")
if os.path.exists(rf_figures_src):
    shutil.copytree(rf_figures_src, rf_figures_dst)
    print(f"Copied RANDOMFOREST figures -> {rf_figures_dst}")

# results.json
rf_json_src = os.path.join(base_dir, "RANDOMFOREST", "results.json")
rf_json_dst = os.path.join(rf_dst, "results.json")
if os.path.exists(rf_json_src):
    shutil.copy2(rf_json_src, rf_json_dst)
    print(f"Copied RANDOMFOREST results.json -> {rf_json_dst}")

# Ensemble results
rf_ensemble_json_src = os.path.join(base_dir, "RANDOMFOREST", "ensemble", "results_ensemble.json")
rf_ensemble_dst = os.path.join(rf_dst, "results_ensemble.json")
if os.path.exists(rf_ensemble_json_src):
    shutil.copy2(rf_ensemble_json_src, rf_ensemble_dst)
    print(f"Copied ensemble results -> {rf_ensemble_dst}")

rf_et_json_src = os.path.join(base_dir, "RANDOMFOREST", "ensemble", "results_et_search.json")
rf_et_dst = os.path.join(rf_dst, "results_et_search.json")
if os.path.exists(rf_et_json_src):
    shutil.copy2(rf_et_json_src, rf_et_dst)
    print(f"Copied ET search results -> {rf_et_dst}")

# Feature engineering results
rf_fe_json_src = os.path.join(base_dir, "RANDOMFOREST", "feature_engineering", "results_feature_engineering.json")
rf_fe_dst = os.path.join(rf_dst, "results_feature_engineering.json")
if os.path.exists(rf_fe_json_src):
    shutil.copy2(rf_fe_json_src, rf_fe_dst)
    print(f"Copied feature engineering results -> {rf_fe_dst}")

# Optimization results
rf_opt_dir = os.path.join(base_dir, "RANDOMFOREST", "optimization")
opt_files = ["results_baseline.json", "results_grid_search_v2.json", "results_variants.json"]
for f in opt_files:
    src = os.path.join(rf_opt_dir, f)
    dst = os.path.join(rf_dst, f)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {f} -> {dst}")

# Optimization figures
rf_opt_fig_src = os.path.join(rf_opt_dir, "figures")
rf_opt_fig_dst = os.path.join(rf_dst, "optimization_figures")
if os.path.exists(rf_opt_fig_src):
    shutil.copytree(rf_opt_fig_src, rf_opt_fig_dst)
    print(f"Copied optimization figures -> {rf_opt_fig_dst}")

# PCA experiment results
rf_pca_json_src = os.path.join(base_dir, "RANDOMFOREST", "pca_experiment", "results_pca_rf.json")
rf_pca_dst = os.path.join(rf_dst, "results_pca_rf.json")
if os.path.exists(rf_pca_json_src):
    shutil.copy2(rf_pca_json_src, rf_pca_dst)
    print(f"Copied PCA experiment results -> {rf_pca_dst}")

# ===== SVM Results =====
svm_dst = os.path.join(output_dir, "SVM")
os.makedirs(svm_dst, exist_ok=True)

# LinearSVM
svm_linear_src = os.path.join(base_dir, "SVM", "LinearSVM", "results_summary.md")
svm_linear_dst = os.path.join(svm_dst, "LinearSVM_results_summary.md")
if os.path.exists(svm_linear_src):
    shutil.copy2(svm_linear_src, svm_linear_dst)
    print(f"Copied LinearSVM results -> {svm_linear_dst}")

# Check for any results in polySVM and RBFSVM
for subfolder in ["polySVM", "RBFSVM"]:
    sub_src = os.path.join(base_dir, "SVM", subfolder)
    sub_dst = os.path.join(svm_dst, subfolder)
    os.makedirs(sub_dst, exist_ok=True)
    for item in os.listdir(sub_src):
        if item.endswith(('.md', '.txt', '.csv', '.json', '.png', '.jpg')):
            shutil.copy2(os.path.join(sub_src, item), os.path.join(sub_dst, item))
            print(f"Copied SVM/{subfolder}/{item} -> {sub_dst}")

# Copy test.md
svm_test_src = os.path.join(base_dir, "SVM", "test.md")
svm_test_dst = os.path.join(svm_dst, "test.md")
if os.path.exists(svm_test_src):
    shutil.copy2(svm_test_src, svm_test_dst)
    print(f"Copied SVM/test.md -> {svm_test_dst}")

# ===== XGBoost Results =====
xgb_src = os.path.join(base_dir, "XGBoost", "output")
xgb_dst = os.path.join(output_dir, "XGBoost")
if os.path.exists(xgb_src):
    shutil.copytree(xgb_src, xgb_dst)
    print(f"Copied XGBoost results -> {xgb_dst}")

# ===== Summary =====
print("\n" + "="*50)
print("All results collected successfully!")
print(f"Output directory: {output_dir}")
print("="*50)

# List the structure
for root, dirs, files in os.walk(output_dir):
    level = root.replace(output_dir, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = '  ' * (level + 1)
    for file in files:
        print(f"{subindent}{file}")