import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier

# Preprocessing & Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    auc,
    silhouette_score,
    f1_score
)

# Hierarchical clustering dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

# Association rule mining (requires: pip install mlxtend)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# =========================================================================
# STEP 1: USER DATA SPLIT CONFIGURATION
# =========================================================================
print("="*60)
print("     UNIFIED FLIGHT DELAY & ROOT CAUSE ANALYTICS SYSTEM     ")
print("="*60)

while True:
    try:
        train_pct = float(input("\nEnter percentage of data to use for TRAINING (e.g. 80 for 80% train / 20% test): "))
        if 10 <= train_pct <= 90:
            test_ratio = (100.0 - train_pct) / 100.0
            break
        else:
            print("Please enter a value between 10 and 90.")
    except ValueError:
        print("Invalid input. Please enter a numerical percentage (e.g. 80).")

print(f"\n[Config] Train/Test Split configured: {100 - test_ratio*100:.0f}% Training / {test_ratio*100:.0f}% Hold-Out Test set.")
if train_pct < 50:
    print("  [Warning] Training share is below 50%. Models (especially Neural Net / SVM) may")
    print("            under-fit or collapse to the majority class, and evaluation will be slow")
    print("            because the hold-out test set is very large. Prefer 70–80% training.")

# =========================================================================
# STEP 2: LOAD & PREPROCESS DATA (TRAINS ON ALL DATA)
# =========================================================================
print("\n--- Loading and Preprocessing 'DelayedFlights.csv' ---")
df = pd.read_csv('DelayedFlights.csv')

# --- CODE 1 DATA PREPARATION ---
df['Is_Severe_Delay'] = (df['ArrDelay'] >= 45).astype(int)
df['DepHour'] = df['CRSDepTime'] // 100
df = df.sort_values(['TailNum', 'CRSDepTime']).reset_index(drop=True)
df['Inbound_Leg_ArrDelay'] = df.groupby('TailNum')['ArrDelay'].shift(1).fillna(0)

features_c1 = ['Distance', 'DepHour', 'Inbound_Leg_ArrDelay']
df_clean_c1 = df[features_c1 + ['ArrDelay', 'Is_Severe_Delay']].dropna()

X_c1 = df_clean_c1[features_c1]
y_class_c1 = df_clean_c1['Is_Severe_Delay']
y_reg_c1 = df_clean_c1['ArrDelay']

X_train_c1, X_test_c1, y_train_class_c1, y_test_class_c1, y_train_reg_c1, y_test_reg_c1 = train_test_split(
    X_c1, y_class_c1, y_reg_c1, test_size=test_ratio, random_state=42
)

scaler_c1 = StandardScaler()
X_train_scaled_c1 = scaler_c1.fit_transform(X_train_c1)
X_test_scaled_c1 = scaler_c1.transform(X_test_c1)

# --- CODE 2 DATA PREPARATION ---
df_delayed = df[df['ArrDelay'] > 0].copy()
features_c2 = ['TaxiOut', 'AirTime', 'Distance']
df_clean_c2 = df_delayed[features_c2 + ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'ArrDelay']].dropna()

def assign_root_cause(row):
    delays = {
        'Carrier': row['CarrierDelay'],
        'Weather': row['WeatherDelay'],
        'NAS/Hub': row['NASDelay']
    }
    max_delay = max(delays.values())
    if max_delay == 0:
        return 'Carrier'
    return max(key for key, val in delays.items() if val == max_delay)

df_clean_c2['RootCause'] = df_clean_c2.apply(assign_root_cause, axis=1)
df_sample_c2 = df_clean_c2

X_c2 = df_sample_c2[features_c2]
y_class_c2 = df_sample_c2['RootCause']
y_reg_c2 = df_sample_c2['CarrierDelay']

X_train_c2, X_test_c2, y_train_class_c2, y_test_class_c2, y_train_reg_c2, y_test_reg_c2 = train_test_split(
    X_c2, y_class_c2, y_reg_c2, test_size=test_ratio, random_state=42
)

scaler_c2 = StandardScaler()
X_train_scaled_c2 = scaler_c2.fit_transform(X_train_c2)
X_test_scaled_c2 = scaler_c2.transform(X_test_c2)

# =========================================================================
# STEP 3: MODEL TRAINING
# =========================================================================
print("\n--- Training Code 1 Models (Severe Delay Occurrence - Full Dataset) ---")
lin_reg_c1 = LinearRegression().fit(X_train_scaled_c1, y_train_reg_c1)
log_reg_c1 = LogisticRegression(class_weight='balanced').fit(X_train_scaled_c1, y_train_class_c1)
dt_cls_c1  = DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight='balanced', random_state=42).fit(X_train_scaled_c1, y_train_class_c1)
rf_cls_c1  = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_scaled_c1, y_train_class_c1)
kmeans_c1  = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_train_scaled_c1)
nn_cls_c1  = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=300, random_state=42).fit(X_train_scaled_c1, y_train_class_c1)

print("--- Training Code 2 Models (Root Cause Diagnosis - Full Dataset) ---")
lin_reg_c2 = LinearRegression().fit(X_train_scaled_c2, y_train_reg_c2)
log_reg_c2 = LogisticRegression(class_weight='balanced', max_iter=500).fit(X_train_scaled_c2, y_train_class_c2)
dt_cls_c2  = DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight='balanced', random_state=42).fit(X_train_scaled_c2, y_train_class_c2)
rf_cls_c2  = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_scaled_c2, y_train_class_c2)
kmeans_c2  = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_train_scaled_c2)
nn_cls_c2  = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=100, random_state=42).fit(X_train_scaled_c2, y_train_class_c2)

print("\nAll models successfully trained!")

# =========================================================================
# STEP 3B: ADDITIONAL MODELS - PCA, SVM, HIERARCHICAL CLUSTERING
# =========================================================================
print("\n--- Training Additional Models: PCA, SVM, Hierarchical Clustering ---")

# ---- PCA (2 components) + downstream Logistic Regression for comparable metrics ----
pca_c1 = PCA(n_components=2, random_state=42).fit(X_train_scaled_c1)
X_train_pca_c1 = pca_c1.transform(X_train_scaled_c1)
X_test_pca_c1  = pca_c1.transform(X_test_scaled_c1)
log_reg_pca_c1 = LogisticRegression(class_weight='balanced').fit(X_train_pca_c1, y_train_class_c1)

pca_c2 = PCA(n_components=2, random_state=42).fit(X_train_scaled_c2)
X_train_pca_c2 = pca_c2.transform(X_train_scaled_c2)
X_test_pca_c2  = pca_c2.transform(X_test_scaled_c2)
log_reg_pca_c2 = LogisticRegression(class_weight='balanced', max_iter=500).fit(X_train_pca_c2, y_train_class_c2)

# ---- SVM (RBF kernel) ----
# NOTE: RBF-kernel SVM training is roughly O(n^2)-O(n^3) and does not scale to the
# ~1M-row full training set, so it is trained on a random stratified-by-chance subsample.
SVM_SAMPLE_SIZE = 5000
svm_idx_c1 = np.random.RandomState(42).choice(len(X_train_scaled_c1), size=min(SVM_SAMPLE_SIZE, len(X_train_scaled_c1)), replace=False)
svm_cls_c1 = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42).fit(
    X_train_scaled_c1[svm_idx_c1], y_train_class_c1.iloc[svm_idx_c1]
)

svm_idx_c2 = np.random.RandomState(42).choice(len(X_train_scaled_c2), size=min(SVM_SAMPLE_SIZE, len(X_train_scaled_c2)), replace=False)
svm_cls_c2 = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42).fit(
    X_train_scaled_c2[svm_idx_c2], y_train_class_c2.iloc[svm_idx_c2]
)

# ---- Hierarchical (Agglomerative) Clustering ----
# NOTE: Agglomerative clustering needs the full pairwise distance matrix (O(n^2) memory),
# so it is trained on a small random subsample rather than the full dataset.
HIER_SAMPLE_SIZE = 2000
hier_idx_c1 = np.random.RandomState(42).choice(len(X_train_scaled_c1), size=min(HIER_SAMPLE_SIZE, len(X_train_scaled_c1)), replace=False)
X_hier_sample_c1 = X_train_scaled_c1[hier_idx_c1]
hier_cls_c1 = AgglomerativeClustering(n_clusters=2).fit(X_hier_sample_c1)
hier_centroids_c1 = np.array([X_hier_sample_c1[hier_cls_c1.labels_ == i].mean(axis=0) for i in range(2)])

hier_idx_c2 = np.random.RandomState(42).choice(len(X_train_scaled_c2), size=min(HIER_SAMPLE_SIZE, len(X_train_scaled_c2)), replace=False)
X_hier_sample_c2 = X_train_scaled_c2[hier_idx_c2]
hier_cls_c2 = AgglomerativeClustering(n_clusters=3).fit(X_hier_sample_c2)
hier_centroids_c2 = np.array([X_hier_sample_c2[hier_cls_c2.labels_ == i].mean(axis=0) for i in range(3)])

def hier_predict(new_scaled_point, centroids):
    """AgglomerativeClustering has no native predict() for unseen points, so new points
    are assigned to the nearest fitted-cluster centroid (nearest-centroid approximation)."""
    dists = np.linalg.norm(centroids - new_scaled_point, axis=1)
    return int(np.argmin(dists))

print("PCA, SVM, and Hierarchical Clustering models trained.")

# =========================================================================
# STEP 3C: EXTRA MODELS — NB, IsolationForest, LabelProp, SelfTrain,
#           Ridge/Lasso, LightGBM, DBSCAN
# (Q-Learning & Transformers omitted: not a natural fit for static 3-feature
#  tabular classification/regression without a full RL environment or
#  sequence model redesign.)
# =========================================================================
print("\n--- Training Extra Models: Naive Bayes, Isolation Forest, Label Propagation,")
print("    Self-Training, Ridge/Lasso, LightGBM, DBSCAN ---")

# Naive Bayes
nb_cls_c1 = GaussianNB().fit(X_train_scaled_c1, y_train_class_c1)
nb_cls_c2 = GaussianNB().fit(X_train_scaled_c2, y_train_class_c2)

# Isolation Forest (anomaly = severe-delay risk proxy for Code 1)
iso_c1 = IsolationForest(n_estimators=100, contamination=0.3, random_state=42, n_jobs=-1).fit(X_train_scaled_c1)

# Label Propagation (semi-supervised) — subsample + mask some labels
LP_SAMPLE = 5000
rng_extra = np.random.RandomState(42)
lp_idx_c1 = rng_extra.choice(len(X_train_scaled_c1), size=min(LP_SAMPLE, len(X_train_scaled_c1)), replace=False)
X_lp_c1 = X_train_scaled_c1[lp_idx_c1]
y_lp_c1 = y_train_class_c1.iloc[lp_idx_c1].values.copy()
# mask 40% of labels as unlabeled (-1)
mask_n = int(0.4 * len(y_lp_c1))
y_lp_c1[rng_extra.choice(len(y_lp_c1), size=mask_n, replace=False)] = -1
lp_cls_c1 = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=400).fit(X_lp_c1, y_lp_c1)

lp_idx_c2 = rng_extra.choice(len(X_train_scaled_c2), size=min(LP_SAMPLE, len(X_train_scaled_c2)), replace=False)
X_lp_c2 = X_train_scaled_c2[lp_idx_c2]
# map root-cause strings to ints for LabelPropagation
_rc_classes = sorted(y_train_class_c2.unique())
_rc_to_i = {c: i for i, c in enumerate(_rc_classes)}
_i_to_rc = {i: c for c, i in _rc_to_i.items()}
y_lp_c2 = y_train_class_c2.iloc[lp_idx_c2].map(_rc_to_i).values.copy().astype(float)
mask_n2 = int(0.4 * len(y_lp_c2))
y_lp_c2[rng_extra.choice(len(y_lp_c2), size=mask_n2, replace=False)] = -1
lp_cls_c2 = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=400).fit(X_lp_c2, y_lp_c2)

# Self-Training (wrap Logistic Regression)
st_base_c1 = LogisticRegression(class_weight='balanced', max_iter=500)
st_cls_c1 = SelfTrainingClassifier(st_base_c1, threshold=0.75, max_iter=10).fit(X_lp_c1, y_lp_c1)
st_base_c2 = LogisticRegression(class_weight='balanced', max_iter=500)
st_cls_c2 = SelfTrainingClassifier(st_base_c2, threshold=0.75, max_iter=10).fit(X_lp_c2, y_lp_c2)

# Ridge & Lasso regression
ridge_c1 = Ridge(alpha=1.0).fit(X_train_scaled_c1, y_train_reg_c1)
lasso_c1 = Lasso(alpha=0.1, max_iter=2000).fit(X_train_scaled_c1, y_train_reg_c1)
ridge_c2 = Ridge(alpha=1.0).fit(X_train_scaled_c2, y_train_reg_c2)
lasso_c2 = Lasso(alpha=0.1, max_iter=2000).fit(X_train_scaled_c2, y_train_reg_c2)

# LightGBM (optional)
if LIGHTGBM_AVAILABLE:
    lgb_cls_c1 = lgb.LGBMClassifier(n_estimators=80, max_depth=6, class_weight='balanced',
                                    random_state=42, verbosity=-1).fit(X_train_scaled_c1, y_train_class_c1)
    lgb_reg_c1 = lgb.LGBMRegressor(n_estimators=80, max_depth=6, random_state=42, verbosity=-1).fit(
        X_train_scaled_c1, y_train_reg_c1)
    lgb_cls_c2 = lgb.LGBMClassifier(n_estimators=80, max_depth=6, class_weight='balanced',
                                    random_state=42, verbosity=-1).fit(X_train_scaled_c2, y_train_class_c2)
    lgb_reg_c2 = lgb.LGBMRegressor(n_estimators=80, max_depth=6, random_state=42, verbosity=-1).fit(
        X_train_scaled_c2, y_train_reg_c2)
else:
    lgb_cls_c1 = lgb_reg_c1 = lgb_cls_c2 = lgb_reg_c2 = None

# DBSCAN (subsample; density-based clusters)
DBSCAN_SAMPLE = 3000
db_idx_c1 = rng_extra.choice(len(X_train_scaled_c1), size=min(DBSCAN_SAMPLE, len(X_train_scaled_c1)), replace=False)
X_db_c1 = X_train_scaled_c1[db_idx_c1]
dbscan_c1 = DBSCAN(eps=0.8, min_samples=15).fit(X_db_c1)
db_idx_c2 = rng_extra.choice(len(X_train_scaled_c2), size=min(DBSCAN_SAMPLE, len(X_train_scaled_c2)), replace=False)
X_db_c2 = X_train_scaled_c2[db_idx_c2]
dbscan_c2 = DBSCAN(eps=0.8, min_samples=15).fit(X_db_c2)

print("Extra models trained (Naive Bayes, Isolation Forest, Label Propagation,")
print(" Self-Training, Ridge/Lasso, LightGBM, DBSCAN).")
print("Note: Q-Learning needs an interactive reward environment; Transformers need")
print(" sequence/token data — neither fits this 3-feature tabular pipeline cleanly.")


# =========================================================================
# STEP 4: MODEL EVALUATION & GRAPH PLOTTING
# =========================================================================
# When the hold-out test set is huge (e.g. 80% of ~1.9M rows), full-set SVM
# prediction and dense scatter plots can take many minutes or appear frozen.
# We therefore cap evaluation / plotting sample sizes for expensive models.
EVAL_MAX_ROWS = 50000          # max rows for metric computation on slow models
PLOT_MAX_ROWS = 8000           # max rows drawn on scatter / ROC plots
SVM_EVAL_MAX  = 20000          # SVM predict on full test is O(n_sv * n); cap it

def _subsample_idx(n, max_n, seed=42):
    if n <= max_n:
        return np.arange(n)
    return np.random.RandomState(seed).choice(n, size=max_n, replace=False)

def evaluate_code_1():
    print("\n" + "="*60)
    print("     CODE 1: SEVERE DELAY PREDICTION EVALUATION METRICS     ")
    print("="*60)

    y_reg_pred = lin_reg_c1.predict(X_test_scaled_c1)
    print(f"[Linear Regression Evaluation]")
    print(f"  • R2 Score: {r2_score(y_test_reg_c1, y_reg_pred):.4f}")
    print(f"  • RMSE:     {np.sqrt(mean_squared_error(y_test_reg_c1, y_reg_pred)):.2f} mins\n")

    # Ridge / Lasso regression metrics
    for reg_name, reg in [("Ridge", ridge_c1), ("Lasso", lasso_c1)]:
        pred = reg.predict(X_test_scaled_c1)
        print(f"[{reg_name} Regression]")
        print(f"  • R2 Score: {r2_score(y_test_reg_c1, pred):.4f}")
        print(f"  • RMSE:     {np.sqrt(mean_squared_error(y_test_reg_c1, pred)):.2f} mins")
    if LIGHTGBM_AVAILABLE and lgb_reg_c1 is not None:
        pred = lgb_reg_c1.predict(X_test_scaled_c1)
        print(f"[LightGBM Regressor]")
        print(f"  • R2 Score: {r2_score(y_test_reg_c1, pred):.4f}")
        print(f"  • RMSE:     {np.sqrt(mean_squared_error(y_test_reg_c1, pred)):.2f} mins")
    print()

    classifiers = {
        "Logistic Regression": log_reg_c1,
        "Decision Tree": dt_cls_c1,
        "Random Forest": rf_cls_c1,
        "Neural Network (MLP)": nn_cls_c1,
        "SVM (RBF)": svm_cls_c1,
        "Naive Bayes": nb_cls_c1,
        "Label Propagation": lp_cls_c1,
        "Self-Training": st_cls_c1,
    }
    if LIGHTGBM_AVAILABLE and lgb_cls_c1 is not None:
        classifiers["LightGBM"] = lgb_cls_c1

    n_test = len(X_test_scaled_c1)
    if n_test > EVAL_MAX_ROWS:
        print(f"  [Note] Test set has {n_test:,} rows. Metrics for heavy models "
              f"(esp. SVM / LabelProp / Self-Training) use a subsample of up to {SVM_EVAL_MAX:,} rows "
              f"so evaluation does not appear frozen.\n")

    for name, clf in classifiers.items():
        print(f"  Evaluating {name} ...", flush=True)
        # SVM (and optionally other heavy models) on a capped subsample
        # SVM / LabelPropagation / Self-Training: O(n_train * n_test) memory — always cap
        heavy = name.startswith("SVM") or name in ("Label Propagation", "Self-Training")
        if heavy and n_test > SVM_EVAL_MAX:
            idx = _subsample_idx(n_test, SVM_EVAL_MAX)
            X_eval = X_test_scaled_c1[idx]
            y_eval = y_test_class_c1.iloc[idx]
            print(f"    ({name} metrics on {len(idx):,} subsampled test rows)")
        else:
            X_eval = X_test_scaled_c1
            y_eval = y_test_class_c1

        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_eval)[:, 1]
            roc_auc = roc_auc_score(y_eval, proba)
        else:
            roc_auc = float('nan')
        f1 = f1_score(y_eval, y_pred, average='binary', zero_division=0)

        print(f"[{name}]")
        print(f"  • Accuracy:  {acc:.4f}")
        print(f"  • ROC-AUC:   {roc_auc:.4f}")
        print(f"  • F1 Score:  {f1:.4f}")
        print("  • Classification Report:")
        print(classification_report(y_eval, y_pred, target_names=['On-Time', 'Delayed'], digits=4, zero_division=0))
        print("-" * 60)


    # Isolation Forest (anomaly detection → map -1 anomaly to Delayed)
    print("  Evaluating Isolation Forest ...", flush=True)
    iso_pred_raw = iso_c1.predict(X_test_scaled_c1 if len(X_test_scaled_c1) <= SVM_EVAL_MAX
                                  else X_test_scaled_c1[_subsample_idx(len(X_test_scaled_c1), SVM_EVAL_MAX)])
    y_iso = y_test_class_c1 if len(X_test_scaled_c1) <= SVM_EVAL_MAX else y_test_class_c1.iloc[_subsample_idx(len(X_test_scaled_c1), SVM_EVAL_MAX)]
    iso_pred = (iso_pred_raw == -1).astype(int)  # anomaly = delayed proxy
    print("[Isolation Forest] (anomaly=-1 → Delayed)")
    print(f"  • Accuracy:  {accuracy_score(y_iso, iso_pred):.4f}")
    print(f"  • F1 Score:  {f1_score(y_iso, iso_pred, average='binary', zero_division=0):.4f}")
    print(classification_report(y_iso, iso_pred, target_names=['On-Time', 'Delayed'], digits=4, zero_division=0))
    print("-" * 60)

    # DBSCAN summary
    n_clusters_db = len(set(dbscan_c1.labels_)) - (1 if -1 in dbscan_c1.labels_ else 0)
    n_noise = list(dbscan_c1.labels_).count(-1)
    print(f"[DBSCAN Code 1] clusters={n_clusters_db}, noise points={n_noise}/{len(dbscan_c1.labels_)}")
    print("-" * 60)

    # ---- Main metrics grid (tree removed from here - see standalone figure below) ----
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('Code 1 - Severe Delay Models Visual Performance', fontsize=16)

    axes[0, 0].scatter(y_test_reg_c1, y_reg_pred, alpha=0.3, color='blue')
    axes[0, 0].plot([y_test_reg_c1.min(), y_test_reg_c1.max()], [y_test_reg_c1.min(), y_test_reg_c1.max()], 'r--', lw=2)
    axes[0, 0].set_title('Multi-Linear Reg: Actual vs Predicted Delay')

    sns.barplot(x=list(log_reg_c1.coef_[0]), y=features_c1, ax=axes[0, 1], palette='viridis', hue=features_c1, legend=False)
    axes[0, 1].set_title('Logistic Regression: Feature Weights')

    # ---- ROC-AUC Curve panel (subsample for speed when test set is huge) ----
    print("  Building Code 1 plots (subsampling large test set if needed) ...", flush=True)
    plot_idx = _subsample_idx(len(X_test_scaled_c1), PLOT_MAX_ROWS)
    X_plot = X_test_scaled_c1[plot_idx]
    y_plot = y_test_class_c1.iloc[plot_idx]
    X_plot_raw = X_test_c1.iloc[plot_idx]
    y_reg_plot = y_test_reg_c1.iloc[plot_idx]
    y_reg_pred_plot = lin_reg_c1.predict(X_plot)

    # overwrite the actual-vs-pred scatter with subsampled points (clearer + faster)
    axes[0, 0].clear()
    axes[0, 0].scatter(y_reg_plot, y_reg_pred_plot, alpha=0.3, color='blue', s=8)
    axes[0, 0].plot([y_reg_plot.min(), y_reg_plot.max()], [y_reg_plot.min(), y_reg_plot.max()], 'r--', lw=2)
    axes[0, 0].set_title('Multi-Linear Reg: Actual vs Predicted Delay')

    roc_ax = axes[0, 2]
    for name, clf in classifiers.items():
        # SVM predict_proba on full test is slow — use plot subsample
        if name.startswith("SVM"):
            X_roc, y_roc = X_plot, y_plot
        else:
            # other models are fast enough; still use plot subsample for consistent overlay size
            X_roc, y_roc = X_plot, y_plot
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_roc)[:, 1]
        else:
            y_score = clf.decision_function(X_roc)
        fpr, tpr, _ = roc_curve(y_roc, y_score)
        roc_ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    roc_ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Chance")
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('ROC Curves: Severe Delay Classifiers')
    roc_ax.legend(loc='lower right', fontsize=8)

    pd.Series(rf_cls_c1.feature_importances_, index=features_c1).plot(kind='barh', color='teal', ax=axes[1, 0])
    axes[1, 0].set_title('Random Forest Feature Importance')

    sns.scatterplot(x=X_plot_raw['Distance'], y=X_plot_raw['Inbound_Leg_ArrDelay'],
                    hue=kmeans_c1.predict(X_plot), palette='Set1', alpha=0.6, ax=axes[1, 1], s=12)
    axes[1, 1].set_title('K-Means Clusters')

    axes[1, 2].plot(nn_cls_c1.loss_curve_, color='purple', lw=2)
    axes[1, 2].set_title('Neural Network Loss Convergence')

    plt.tight_layout(pad=3.0)
    plt.show()

    # ---- FIXED: Decision Tree gets its own full-size figure so leaves render cleanly ----
    fig_tree, ax_tree = plt.subplots(figsize=(20, 12))
    plot_tree(dt_cls_c1, feature_names=features_c1, class_names=['On-Time', 'Delayed'],
              filled=True, ax=ax_tree, fontsize=10, rounded=True, precision=2, proportion=False)
    ax_tree.set_title('Code 1 - Decision Tree Structure (Severe Delay)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_code_2():
    print("\n" + "="*60)
    print("     CODE 2: ROOT CAUSE DIAGNOSIS EVALUATION METRICS     ")
    print("="*60)

    y_reg_pred = lin_reg_c2.predict(X_test_scaled_c2)
    print(f"[Linear Regression Evaluation (Carrier Delay Minutes)]")
    print(f"  • R2 Score: {r2_score(y_test_reg_c2, y_reg_pred):.4f}")
    print(f"  • RMSE:     {np.sqrt(mean_squared_error(y_test_reg_c2, y_reg_pred)):.2f} mins\n")

    for reg_name, reg in [("Ridge", ridge_c2), ("Lasso", lasso_c2)]:
        pred = reg.predict(X_test_scaled_c2)
        print(f"[{reg_name} Regression (Carrier Delay)]")
        print(f"  • R2 Score: {r2_score(y_test_reg_c2, pred):.4f}")
        print(f"  • RMSE:     {np.sqrt(mean_squared_error(y_test_reg_c2, pred)):.2f} mins")
    if LIGHTGBM_AVAILABLE and lgb_reg_c2 is not None:
        pred = lgb_reg_c2.predict(X_test_scaled_c2)
        print(f"[LightGBM Regressor]")
        print(f"  • R2 Score: {r2_score(y_test_reg_c2, pred):.4f}")
        print(f"  • RMSE:     {np.sqrt(mean_squared_error(y_test_reg_c2, pred)):.2f} mins")
    print()

    classifiers = {
        "Logistic Regression": log_reg_c2,
        "Decision Tree": dt_cls_c2,
        "Random Forest": rf_cls_c2,
        "Neural Network (MLP)": nn_cls_c2,
        "SVM (RBF)": svm_cls_c2,
        "Naive Bayes": nb_cls_c2,
    }
    if LIGHTGBM_AVAILABLE and lgb_cls_c2 is not None:
        classifiers["LightGBM"] = lgb_cls_c2

    # Multiclass ROC-AUC needs one-vs-rest binarized labels
    all_classes = sorted(y_class_c2.unique())
    n_test = len(X_test_scaled_c2)
    if n_test > EVAL_MAX_ROWS:
        print(f"  [Note] Test set has {n_test:,} rows. Heavy models (esp. SVM) "
              f"use a subsample of up to {SVM_EVAL_MAX:,} rows for metrics.\n")

    for name, clf in classifiers.items():
        print(f"  Evaluating {name} ...", flush=True)
        heavy = name.startswith("SVM") or name in ("Label Propagation", "Self-Training")
        if heavy and n_test > SVM_EVAL_MAX:
            idx = _subsample_idx(n_test, SVM_EVAL_MAX)
            X_eval = X_test_scaled_c2[idx]
            y_eval = y_test_class_c2.iloc[idx]
            print(f"    ({name} metrics on {len(idx):,} subsampled test rows)")
        else:
            X_eval = X_test_scaled_c2
            y_eval = y_test_class_c2

        y_test_bin_eval = label_binarize(y_eval, classes=all_classes)
        y_pred = clf.predict(X_eval)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average='macro', zero_division=0)
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X_eval)
            proba_aligned = pd.DataFrame(proba, columns=clf.classes_).reindex(columns=all_classes).values
            macro_auc = roc_auc_score(y_test_bin_eval, proba_aligned, average='macro', multi_class='ovr')
        else:
            macro_auc = float('nan')
        print(f"[{name}]")
        print(f"  • Accuracy:       {acc:.4f}")
        print(f"  • Macro ROC-AUC:  {macro_auc:.4f} (one-vs-rest)")
        print(f"  • Macro F1:       {f1:.4f}")
        print("  • Classification Report:")
        print(classification_report(y_eval, y_pred, digits=4, zero_division=0))
        print("-" * 60)


    # Label Propagation & Self-Training (integer-encoded root causes)
    print("  Evaluating Label Propagation (Code 2) ...", flush=True)
    n_eval = min(SVM_EVAL_MAX, len(X_test_scaled_c2))
    idx_e = _subsample_idx(len(X_test_scaled_c2), n_eval)
    X_e = X_test_scaled_c2[idx_e]
    y_e = y_test_class_c2.iloc[idx_e]
    lp_pred_i = lp_cls_c2.predict(X_e).astype(int)
    lp_pred = pd.Series(lp_pred_i).map(_i_to_rc).values
    print("[Label Propagation]")
    print(f"  • Accuracy:  {accuracy_score(y_e, lp_pred):.4f}")
    print(f"  • Macro F1:  {f1_score(y_e, lp_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_e, lp_pred, digits=4, zero_division=0))
    print("-" * 60)

    print("  Evaluating Self-Training (Code 2) ...", flush=True)
    st_pred_i = st_cls_c2.predict(X_e).astype(int)
    st_pred = pd.Series(st_pred_i).map(_i_to_rc).values
    print("[Self-Training]")
    print(f"  • Accuracy:  {accuracy_score(y_e, st_pred):.4f}")
    print(f"  • Macro F1:  {f1_score(y_e, st_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_e, st_pred, digits=4, zero_division=0))
    print("-" * 60)

    n_clusters_db2 = len(set(dbscan_c2.labels_)) - (1 if -1 in dbscan_c2.labels_ else 0)
    n_noise2 = list(dbscan_c2.labels_).count(-1)
    print(f"[DBSCAN Code 2] clusters={n_clusters_db2}, noise points={n_noise2}/{len(dbscan_c2.labels_)}")
    print("-" * 60)

    # ---- Main metrics grid (tree removed from here - see standalone figure below) ----
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('Code 2 - Delay Root Cause Models Visual Performance', fontsize=16)

    axes[0, 0].scatter(y_test_reg_c2, y_reg_pred, alpha=0.3, color='crimson')
    axes[0, 0].plot([y_test_reg_c2.min(), y_test_reg_c2.max()], [y_test_reg_c2.min(), y_test_reg_c2.max()], 'k--', lw=2)
    axes[0, 0].set_title('Multi-Linear Reg: Estimated Carrier Delay')

    coef_df = pd.DataFrame(log_reg_c2.coef_, index=log_reg_c2.classes_, columns=features_c2)
    sns.heatmap(coef_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=axes[0, 1])
    axes[0, 1].set_title('Logistic Reg: Feature Weights Heatmap')

    pd.Series(rf_cls_c2.feature_importances_, index=features_c2).plot(kind='bar', color='darkgreen', rot=0, ax=axes[1, 0])
    axes[1, 0].set_title('Random Forest Feature Importance')

    print("  Building Code 2 plots (subsampling large test set if needed) ...", flush=True)
    plot_idx2 = _subsample_idx(len(X_test_scaled_c2), PLOT_MAX_ROWS)
    X_plot2 = X_test_scaled_c2[plot_idx2]
    X_plot_raw2 = X_test_c2.iloc[plot_idx2]
    y_reg_plot2 = y_test_reg_c2.iloc[plot_idx2]
    y_reg_pred_plot2 = lin_reg_c2.predict(X_plot2)
    axes[0, 0].clear()
    axes[0, 0].scatter(y_reg_plot2, y_reg_pred_plot2, alpha=0.3, color='crimson', s=8)
    axes[0, 0].plot([y_reg_plot2.min(), y_reg_plot2.max()], [y_reg_plot2.min(), y_reg_plot2.max()], 'k--', lw=2)
    axes[0, 0].set_title('Multi-Linear Reg: Estimated Carrier Delay')

    # RF ROC on subsample for speed
    roc_ax = axes[0, 2]
    roc_ax.clear()
    all_classes = sorted(y_class_c2.unique())
    y_plot2 = y_test_class_c2.iloc[plot_idx2]
    y_bin_plot2 = label_binarize(y_plot2, classes=all_classes)
    rf_proba = pd.DataFrame(rf_cls_c2.predict_proba(X_plot2), columns=rf_cls_c2.classes_).reindex(columns=all_classes).values
    for i, cls in enumerate(all_classes):
        fpr, tpr, _ = roc_curve(y_bin_plot2[:, i], rf_proba[:, i])
        roc_ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    roc_ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Chance")
    roc_ax.set_xlabel('False Positive Rate')
    roc_ax.set_ylabel('True Positive Rate')
    roc_ax.set_title('ROC Curves: Random Forest (One-vs-Rest)')
    roc_ax.legend(loc='lower right', fontsize=8)

    sns.scatterplot(x=X_plot_raw2['TaxiOut'], y=X_plot_raw2['AirTime'],
                    hue=kmeans_c2.predict(X_plot2), palette='Set2', alpha=0.6, ax=axes[1, 1], s=12)
    axes[1, 1].set_title('K-Means Clusters')

    axes[1, 2].plot(nn_cls_c2.loss_curve_, color='orange', lw=2)
    axes[1, 2].set_title('Neural Network Loss Convergence')

    plt.tight_layout(pad=3.0)
    plt.show()

    # ---- FIXED: Decision Tree gets its own full-size figure so leaves render cleanly ----
    fig_tree, ax_tree = plt.subplots(figsize=(24, 12))
    plot_tree(dt_cls_c2, feature_names=features_c2, class_names=list(dt_cls_c2.classes_),
              filled=True, ax=ax_tree, fontsize=10, rounded=True, precision=2, proportion=False)
    ax_tree.set_title('Code 2 - Decision Tree Structure (Root Cause)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix():
    print("\n" + "="*60)
    print("     CORRELATION MATRIX - ALL NUMERICAL FEATURES     ")
    print("="*60)

    numeric_df = df.select_dtypes(include=[np.number])
    if 'Unnamed: 0' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Unnamed: 0'])

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        linewidths=0.5
    )
    plt.title('Correlation Matrix of All Numerical Features (DelayedFlights)', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

    if 'ArrDelay' in corr_matrix.columns:
        print("\n--- Correlation with Arrival Delay (ArrDelay) ---")
        print(corr_matrix['ArrDelay'].sort_values(ascending=False))

def evaluate_pca():
    print("\n" + "="*60)
    print("     PCA - DIMENSIONALITY REDUCTION EVALUATION     ")
    print("="*60)

    print(f"[Code 1] Explained variance ratio (PC1, PC2): {pca_c1.explained_variance_ratio_}")
    print(f"[Code 1] Cumulative variance explained: {pca_c1.explained_variance_ratio_.sum()*100:.2f}%\n")

    y_pred_pca_c1 = log_reg_pca_c1.predict(X_test_pca_c1)
    acc_pca_c1 = accuracy_score(y_test_class_c1, y_pred_pca_c1)
    roc_auc_pca_c1 = roc_auc_score(y_test_class_c1, log_reg_pca_c1.predict_proba(X_test_pca_c1)[:, 1])
    print("[Code 1] Logistic Regression trained on 2 PCA components:")
    print(f"  • Accuracy:  {acc_pca_c1:.4f}")
    print(f"  • ROC-AUC:   {roc_auc_pca_c1:.4f}")
    print(classification_report(y_test_class_c1, y_pred_pca_c1, target_names=['On-Time', 'Delayed'], digits=4))

    print(f"\n[Code 2] Explained variance ratio (PC1, PC2): {pca_c2.explained_variance_ratio_}")
    print(f"[Code 2] Cumulative variance explained: {pca_c2.explained_variance_ratio_.sum()*100:.2f}%\n")

    y_pred_pca_c2 = log_reg_pca_c2.predict(X_test_pca_c2)
    acc_pca_c2 = accuracy_score(y_test_class_c2, y_pred_pca_c2)
    print("[Code 2] Logistic Regression trained on 2 PCA components:")
    print(f"  • Accuracy:  {acc_pca_c2:.4f}")
    print(classification_report(y_test_class_c2, y_pred_pca_c2, digits=4))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('PCA - Dimensionality Reduction Analysis', fontsize=16)

    axes[0, 0].bar(['PC1', 'PC2'], pca_c1.explained_variance_ratio_, color='slateblue')
    axes[0, 0].set_title('Code 1: Explained Variance Ratio')
    axes[0, 0].set_ylabel('Variance Ratio')

    sc1 = axes[0, 1].scatter(X_test_pca_c1[:, 0], X_test_pca_c1[:, 1], c=y_test_class_c1, cmap='coolwarm', alpha=0.3, s=8)
    axes[0, 1].set_title('Code 1: Test Set Projected onto PC1 vs PC2')
    axes[0, 1].set_xlabel('PC1'); axes[0, 1].set_ylabel('PC2')
    plt.colorbar(sc1, ax=axes[0, 1], label='Is_Severe_Delay')

    axes[1, 0].bar(['PC1', 'PC2'], pca_c2.explained_variance_ratio_, color='seagreen')
    axes[1, 0].set_title('Code 2: Explained Variance Ratio')
    axes[1, 0].set_ylabel('Variance Ratio')

    for cls in sorted(y_test_class_c2.unique()):
        mask = (y_test_class_c2 == cls).values
        axes[1, 1].scatter(X_test_pca_c2[mask, 0], X_test_pca_c2[mask, 1], alpha=0.3, s=8, label=cls)
    axes[1, 1].set_title('Code 2: Test Set Projected onto PC1 vs PC2')
    axes[1, 1].set_xlabel('PC1'); axes[1, 1].set_ylabel('PC2')
    axes[1, 1].legend()

    plt.tight_layout(pad=3.0)
    plt.show()

def evaluate_svm():
    print("\n" + "="*60)
    print("     SVM (RBF KERNEL) EVALUATION     ")
    print("="*60)
    print(f"Note: trained on a random subsample of {SVM_SAMPLE_SIZE} rows per code "
          f"(full RBF-kernel SVM does not scale to the ~1M-row dataset).")
    print("Accuracy / ROC-AUC / F1 for SVM are included in the Code 1 and Code 2 evaluation menus")
    print("(option 1/2/3), alongside the other classifiers. Below is a decision-boundary visual.\n")

    print("  Building SVM decision-boundary plot (subsampled) ...", flush=True)
    # Visualize decision boundary in 2D PCA space (trained on the same SVM subsample)
    svm_viz_c1 = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42).fit(
        X_train_pca_c1[svm_idx_c1], y_train_class_c1.iloc[svm_idx_c1]
    )

    plot_idx = _subsample_idx(len(X_test_pca_c1), PLOT_MAX_ROWS)
    X_pca_plot = X_test_pca_c1[plot_idx]
    y_plot = y_test_class_c1.iloc[plot_idx]

    xx, yy = np.meshgrid(
        np.linspace(X_pca_plot[:, 0].min(), X_pca_plot[:, 0].max(), 150),
        np.linspace(X_pca_plot[:, 1].min(), X_pca_plot[:, 1].max(), 150)
    )
    Z = svm_viz_c1.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    sc = ax.scatter(X_pca_plot[:, 0], X_pca_plot[:, 1], c=y_plot, cmap='coolwarm', edgecolor='k', s=10, alpha=0.6)
    ax.set_title('Code 1: SVM (RBF) Decision Boundary in PCA Space')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    plt.colorbar(sc, ax=ax, label='Is_Severe_Delay')
    plt.tight_layout()
    plt.show()

def evaluate_hierarchical():
    print("\n" + "="*60)
    print("     HIERARCHICAL (AGGLOMERATIVE) CLUSTERING EVALUATION     ")
    print("="*60)
    print(f"Note: trained on a random subsample of {HIER_SAMPLE_SIZE} rows per code "
          f"(Agglomerative Clustering needs an O(n^2) distance matrix, so it cannot run on the full dataset).\n")

    sil_c1 = silhouette_score(X_hier_sample_c1, hier_cls_c1.labels_)
    print(f"[Code 1] Silhouette Score (2 clusters): {sil_c1:.4f}")

    sil_c2 = silhouette_score(X_hier_sample_c2, hier_cls_c2.labels_)
    print(f"[Code 2] Silhouette Score (3 clusters): {sil_c2:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Hierarchical (Agglomerative) Clustering Analysis', fontsize=16)

    dendro_idx_c1 = np.random.RandomState(1).choice(len(X_hier_sample_c1), size=min(40, len(X_hier_sample_c1)), replace=False)
    Z1 = linkage(X_hier_sample_c1[dendro_idx_c1], method='ward')
    dendrogram(Z1, ax=axes[0, 0])
    axes[0, 0].set_title('Code 1: Dendrogram (40-point sample)')
    axes[0, 0].set_xlabel('Sample Index'); axes[0, 0].set_ylabel('Distance')

    axes[0, 1].scatter(X_hier_sample_c1[:, 0], X_hier_sample_c1[:, 2], c=hier_cls_c1.labels_, cmap='Set1', alpha=0.6)
    axes[0, 1].set_title('Code 1: Hierarchical Clusters (Distance vs Inbound Delay, scaled)')
    axes[0, 1].set_xlabel('Distance (scaled)'); axes[0, 1].set_ylabel('Inbound Leg ArrDelay (scaled)')

    dendro_idx_c2 = np.random.RandomState(1).choice(len(X_hier_sample_c2), size=min(40, len(X_hier_sample_c2)), replace=False)
    Z2 = linkage(X_hier_sample_c2[dendro_idx_c2], method='ward')
    dendrogram(Z2, ax=axes[1, 0])
    axes[1, 0].set_title('Code 2: Dendrogram (40-point sample)')
    axes[1, 0].set_xlabel('Sample Index'); axes[1, 0].set_ylabel('Distance')

    axes[1, 1].scatter(X_hier_sample_c2[:, 0], X_hier_sample_c2[:, 1], c=hier_cls_c2.labels_, cmap='Set2', alpha=0.6)
    axes[1, 1].set_title('Code 2: Hierarchical Clusters (TaxiOut vs AirTime, scaled)')
    axes[1, 1].set_xlabel('TaxiOut (scaled)'); axes[1, 1].set_ylabel('AirTime (scaled)')

    plt.tight_layout(pad=3.0)
    plt.show()

def run_apriori():
    print("\n" + "="*60)
    print("     APRIORI - ASSOCIATION RULE MINING     ")
    print("="*60)

    if not MLXTEND_AVAILABLE:
        print("The 'mlxtend' package is required for this feature but is not installed.")
        print("Install it with:  pip install mlxtend")
        return

    apriori_df = df_clean_c1.copy()

    def bin_distance(d):
        if d < 500:
            return 'Distance_Short'
        elif d < 1500:
            return 'Distance_Medium'
        return 'Distance_Long'

    def bin_hour(h):
        if 5 <= h <= 11:
            return 'Dep_Morning'
        elif 12 <= h <= 17:
            return 'Dep_Afternoon'
        elif 18 <= h <= 21:
            return 'Dep_Evening'
        return 'Dep_Night'

    def bin_inbound_delay(x):
        if x <= 0:
            return 'InboundDelay_None'
        elif x <= 30:
            return 'InboundDelay_Minor'
        return 'InboundDelay_Major'

    apriori_df['DistanceBin'] = apriori_df['Distance'].apply(bin_distance)
    apriori_df['DepHourBin'] = apriori_df['DepHour'].apply(bin_hour)
    apriori_df['InboundDelayBin'] = apriori_df['Inbound_Leg_ArrDelay'].apply(bin_inbound_delay)
    apriori_df['DelayStatus'] = apriori_df['Is_Severe_Delay'].map({0: 'OnTime', 1: 'SevereDelay'})

    basket = pd.get_dummies(apriori_df[['DistanceBin', 'DepHourBin', 'InboundDelayBin', 'DelayStatus']])

    frequent_itemsets = apriori(basket, min_support=0.03, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

    delay_rules = rules[rules['consequents'].astype(str).str.contains('SevereDelay')].sort_values('lift', ascending=False)

    print(f"Found {len(frequent_itemsets)} frequent itemsets and {len(rules)} total association rules.")
    print("\nTop rules predicting SEVERE DELAY (sorted by lift):")
    print(delay_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Apriori - Association Rule Mining (Flight Delay Patterns)', fontsize=16)

    axes[0].scatter(rules['support'], rules['confidence'], s=rules['lift']*40, alpha=0.5, c=rules['lift'], cmap='viridis')
    axes[0].set_xlabel('Support'); axes[0].set_ylabel('Confidence')
    axes[0].set_title('All Rules: Support vs Confidence (bubble size/color = Lift)')

    top10 = delay_rules.head(10).iloc[::-1]
    labels = [f"{list(a)} -> {list(c)}" for a, c in zip(top10['antecedents'], top10['consequents'])]
    axes[1].barh(labels, top10['lift'], color='darkorange')
    axes[1].set_xlabel('Lift')
    axes[1].set_title('Top 10 Rules Predicting Severe Delay (by Lift)')

    plt.tight_layout(pad=3.0)
    plt.show()


def evaluate_extra_models():
    """Visuals for Naive Bayes, Isolation Forest, Label Propagation, Self-Training,
    Ridge/Lasso, LightGBM, and DBSCAN."""
    print("\n" + "="*60)
    print("     EXTRA MODELS — VISUAL ANALYSIS")
    print("="*60)

    plot_idx = _subsample_idx(len(X_test_scaled_c1), PLOT_MAX_ROWS)
    X_plot = X_test_scaled_c1[plot_idx]
    y_plot = y_test_class_c1.iloc[plot_idx]
    X_plot_raw = X_test_c1.iloc[plot_idx]
    y_reg_plot = y_test_reg_c1.iloc[plot_idx]

    # ========== Figure 1: Ridge / Lasso / LightGBM regression ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Extra Models — Regression: Actual vs Predicted Arrival Delay (Code 1)', fontsize=14)

    for ax, model, name, color in [
        (axes[0], ridge_c1, 'Ridge', 'steelblue'),
        (axes[1], lasso_c1, 'Lasso', 'darkorange'),
        (axes[2], lgb_reg_c1 if LIGHTGBM_AVAILABLE else ridge_c1,
         'LightGBM' if LIGHTGBM_AVAILABLE else 'Ridge (LGBM N/A)', 'seagreen'),
    ]:
        if model is None:
            ax.set_title(f'{name}: not available')
            continue
        pred = model.predict(X_plot)
        ax.scatter(y_reg_plot, pred, alpha=0.3, s=10, color=color)
        lo = min(y_reg_plot.min(), pred.min())
        hi = max(y_reg_plot.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5)
        r2 = r2_score(y_reg_plot, pred)
        ax.set_title(f'{name} (R²={r2:.3f})')
        ax.set_xlabel('Actual ArrDelay'); ax.set_ylabel('Predicted')
    plt.tight_layout()
    plt.show()

    # Coefficient comparison Ridge vs Lasso
    fig, ax = plt.subplots(figsize=(8, 4))
    coef_df = pd.DataFrame({
        'Ridge': ridge_c1.coef_,
        'Lasso': lasso_c1.coef_,
    }, index=features_c1)
    coef_df.plot(kind='bar', ax=ax, color=['steelblue', 'darkorange'], rot=0)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title('Ridge vs Lasso Coefficients (Code 1)')
    ax.set_ylabel('Coefficient')
    plt.tight_layout()
    plt.show()

    # ========== Figure 2: Isolation Forest ==========
    print("  Plotting Isolation Forest anomaly scores ...", flush=True)
    iso_scores = -iso_c1.score_samples(X_plot)  # higher = more anomalous
    iso_labels = iso_c1.predict(X_plot)  # -1 anomaly, 1 normal

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Isolation Forest — Anomaly Detection (Code 1)', fontsize=14)

    axes[0].hist(iso_scores[iso_labels == 1], bins=40, alpha=0.7, label='Normal', color='steelblue')
    axes[0].hist(iso_scores[iso_labels == -1], bins=40, alpha=0.7, label='Anomaly', color='crimson')
    axes[0].set_xlabel('Anomaly Score (higher = more anomalous)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Anomaly Score Distribution')
    axes[0].legend()

    sc = axes[1].scatter(X_plot_raw['Distance'], X_plot_raw['Inbound_Leg_ArrDelay'],
                         c=iso_labels, cmap='RdYlGn', alpha=0.5, s=12)
    axes[1].set_xlabel('Distance'); axes[1].set_ylabel('Inbound Leg ArrDelay')
    axes[1].set_title('Anomalies in Feature Space (red=anomaly)')
    plt.colorbar(sc, ax=axes[1], label='Label (-1=anomaly)')
    plt.tight_layout()
    plt.show()

    # ========== Figure 3: Naive Bayes / Label Prop / Self-Train ROC ==========
    print("  Plotting ROC curves for NB / LabelProp / Self-Training ...", flush=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for clf, name in [
        (nb_cls_c1, 'Naive Bayes'),
        (lp_cls_c1, 'Label Propagation'),
        (st_cls_c1, 'Self-Training'),
    ]:
        if hasattr(clf, 'predict_proba'):
            try:
                scores = clf.predict_proba(X_plot)[:, 1]
            except Exception:
                scores = clf.predict(X_plot).astype(float)
        else:
            scores = clf.predict(X_plot).astype(float)
        fpr, tpr, _ = roc_curve(y_plot, scores)
        ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.3f})')
    if LIGHTGBM_AVAILABLE and lgb_cls_c1 is not None:
        scores = lgb_cls_c1.predict_proba(X_plot)[:, 1]
        fpr, tpr, _ = roc_curve(y_plot, scores)
        ax.plot(fpr, tpr, lw=2, label=f'LightGBM (AUC={auc(fpr, tpr):.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Chance')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — Extra Classifiers (Code 1)')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # ========== Figure 4: LightGBM feature importance ==========
    if LIGHTGBM_AVAILABLE and lgb_cls_c1 is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('LightGBM Feature Importance', fontsize=14)
        pd.Series(lgb_cls_c1.feature_importances_, index=features_c1).plot(
            kind='barh', ax=axes[0], color='teal')
        axes[0].set_title('Code 1 Classifier')
        pd.Series(lgb_cls_c2.feature_importances_, index=features_c2).plot(
            kind='barh', ax=axes[1], color='darkgreen')
        axes[1].set_title('Code 2 Classifier')
        plt.tight_layout()
        plt.show()
    else:
        print("  [LightGBM not installed — skipping importance plot. pip install lightgbm]")

    # ========== Figure 5: DBSCAN clusters ==========
    print("  Plotting DBSCAN clusters ...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DBSCAN Density-Based Clustering', fontsize=14)

    # Code 1 — use stored sample
    labels_db1 = dbscan_c1.labels_
    # X_db was not stored — re-derive from a subsample of train for viz
    # Use hier sample size style: predict on plot points via fit labels on X_db
    # We only have labels on the training subsample; scatter those
    # Re-run a small fit for viz consistency on plot points is expensive;
    # show the fitted subsample using distance vs inbound from original scale approx
    axes[0].scatter(range(len(labels_db1)), labels_db1, c=labels_db1, cmap='tab10', s=8, alpha=0.6)
    axes[0].set_title(f'Code 1 DBSCAN labels (sample n={len(labels_db1)})\n'
                      f'clusters={len(set(labels_db1))-(1 if -1 in labels_db1 else 0)}, '
                      f'noise={list(labels_db1).count(-1)}')
    axes[0].set_xlabel('Sample index'); axes[0].set_ylabel('Cluster ID (-1 = noise)')

    labels_db2 = dbscan_c2.labels_
    axes[1].scatter(range(len(labels_db2)), labels_db2, c=labels_db2, cmap='tab10', s=8, alpha=0.6)
    axes[1].set_title(f'Code 2 DBSCAN labels (sample n={len(labels_db2)})\n'
                      f'clusters={len(set(labels_db2))-(1 if -1 in labels_db2 else 0)}, '
                      f'noise={list(labels_db2).count(-1)}')
    axes[1].set_xlabel('Sample index'); axes[1].set_ylabel('Cluster ID (-1 = noise)')
    plt.tight_layout()
    plt.show()

    # Better DBSCAN 2D scatter: fit on plot subsample for display only
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DBSCAN Clusters in Feature Space (viz subsample)', fontsize=14)
    db_viz1 = DBSCAN(eps=0.8, min_samples=12).fit(X_plot)
    axes[0].scatter(X_plot_raw['Distance'], X_plot_raw['Inbound_Leg_ArrDelay'],
                    c=db_viz1.labels_, cmap='tab10', s=12, alpha=0.6)
    axes[0].set_xlabel('Distance'); axes[0].set_ylabel('Inbound Leg ArrDelay')
    axes[0].set_title('Code 1: DBSCAN on Distance vs Inbound Delay')

    plot_idx2 = _subsample_idx(len(X_test_scaled_c2), PLOT_MAX_ROWS)
    X_plot2 = X_test_scaled_c2[plot_idx2]
    X_raw2 = X_test_c2.iloc[plot_idx2]
    db_viz2 = DBSCAN(eps=0.8, min_samples=12).fit(X_plot2)
    axes[1].scatter(X_raw2['TaxiOut'], X_raw2['AirTime'],
                    c=db_viz2.labels_, cmap='tab10', s=12, alpha=0.6)
    axes[1].set_xlabel('TaxiOut'); axes[1].set_ylabel('AirTime')
    axes[1].set_title('Code 2: DBSCAN on TaxiOut vs AirTime')
    plt.tight_layout()
    plt.show()

    # ========== Figure 6: Semi-supervised agreement heatmap-style bar ==========
    print("  Comparing semi-supervised vs supervised predictions ...", flush=True)
    pred_nb = nb_cls_c1.predict(X_plot)
    pred_lp = lp_cls_c1.predict(X_plot)
    pred_st = st_cls_c1.predict(X_plot)
    pred_rf = rf_cls_c1.predict(X_plot)

    agree = pd.DataFrame({
        'NB vs RF': (pred_nb == pred_rf).mean(),
        'LabelProp vs RF': (pred_lp == pred_rf).mean(),
        'SelfTrain vs RF': (pred_st == pred_rf).mean(),
        'NB vs LabelProp': (pred_nb == pred_lp).mean(),
    }, index=['Agreement Rate'])

    fig, ax = plt.subplots(figsize=(8, 3))
    agree.T.plot(kind='barh', ax=ax, legend=False, color='slateblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Fraction of matching predictions')
    ax.set_title('Prediction Agreement with Random Forest (Code 1 subsample)')
    plt.tight_layout()
    plt.show()

    print("\nExtra-model visuals complete.")

# =========================================================================
# MAIN MENU LOOP (Back / navigate freely — never stuck)
# =========================================================================

def _normalize_upload_df(raw):
    dfu = raw.copy()
    dfu.columns = [str(c).strip() for c in dfu.columns]
    lower = {c.lower(): c for c in dfu.columns}
    aliases = {
        "distance": "Distance", "crsdeptime": "CRSDepTime", "deptime": "CRSDepTime",
        "dephour": "DepHour", "arrdelay": "ArrDelay", "taxiout": "TaxiOut",
        "airtime": "AirTime", "tailnum": "TailNum", "carrierdelay": "CarrierDelay",
        "weatherdelay": "WeatherDelay", "nasdelay": "NASDelay",
        "inbound_leg_arrdelay": "Inbound_Leg_ArrDelay", "inbounddelay": "Inbound_Leg_ArrDelay",
    }
    rename = {}
    for low, std in aliases.items():
        if low in lower and std not in dfu.columns:
            rename[lower[low]] = std
    if rename:
        dfu = dfu.rename(columns=rename)
    if "DepHour" not in dfu.columns and "CRSDepTime" in dfu.columns:
        dfu["DepHour"] = (pd.to_numeric(dfu["CRSDepTime"], errors="coerce").fillna(1200) // 100).astype(int)
    if "DepHour" not in dfu.columns:
        dfu["DepHour"] = 12
    for col, default in [
        ("Distance", 500.0), ("Inbound_Leg_ArrDelay", 0.0), ("TaxiOut", 15.0),
        ("AirTime", 100.0), ("CarrierDelay", 0.0), ("WeatherDelay", 0.0),
        ("NASDelay", 0.0), ("ArrDelay", 0.0),
    ]:
        if col not in dfu.columns:
            dfu[col] = default
        else:
            dfu[col] = pd.to_numeric(dfu[col], errors="coerce").fillna(default)
    dfu["DepHour"] = dfu["DepHour"].clip(0, 23).astype(int)
    return dfu


def run_upload_and_score():
    """Score an external CSV/Excel and optionally append to DelayedFlightswupdates.csv."""
    print("\n" + "="*60)
    print("   UPLOAD & SCORE NEW FLIGHTS → DelayedFlightswupdates.csv")
    print("="*60)
    print("Expected columns (subset OK): Distance, CRSDepTime/DepHour, ArrDelay,")
    print("  TaxiOut, AirTime, TailNum, CarrierDelay, WeatherDelay, NASDelay")
    path = input("\nPath to CSV or Excel file (or press Enter to cancel): ").strip().strip('"')
    if not path:
        print("Cancelled.")
        return
    try:
        if path.lower().endswith((".xlsx", ".xls")):
            raw = pd.read_excel(path)
        else:
            raw = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    print(f"Loaded {len(raw):,} rows from {path}")
    dfu = _normalize_upload_df(raw)

    # Code 1 score
    X1s = scaler_c1.transform(dfu[features_c1])
    X1p = pca_c1.transform(X1s)
    dfu["Pred_ArrDelay_mins"] = lin_reg_c1.predict(X1s)
    dfu["Pred_SevereDelay_RF"] = rf_cls_c1.predict(X1s)
    dfu["Pred_SevereDelay_LogReg"] = log_reg_c1.predict(X1s)
    dfu["Pred_SevereDelay_NB"] = nb_cls_c1.predict(X1s)
    dfu["Pred_SevereDelay_SVM"] = svm_cls_c1.predict(X1s)
    dfu["Pred_SevereDelay_PCA"] = log_reg_pca_c1.predict(X1p)
    dfu["KMeans_Cluster_C1"] = kmeans_c1.predict(X1s)
    dfu["Hier_Cluster_C1"] = [hier_predict(row, hier_centroids_c1) for row in X1s]
    dfu["IsolationForest"] = iso_c1.predict(X1s)
    dfu["Is_Anomaly"] = (dfu["IsolationForest"] == -1).astype(int)
    if hasattr(rf_cls_c1, "predict_proba"):
        dfu["Prob_SevereDelay_RF"] = rf_cls_c1.predict_proba(X1s)[:, 1]

    # Code 2 score
    X2s = scaler_c2.transform(dfu[features_c2])
    X2p = pca_c2.transform(X2s)
    dfu["Pred_CarrierDelay_mins"] = lin_reg_c2.predict(X2s)
    dfu["Pred_RootCause_RF"] = rf_cls_c2.predict(X2s)
    dfu["Pred_RootCause_LogReg"] = log_reg_c2.predict(X2s)
    dfu["Pred_RootCause_NB"] = nb_cls_c2.predict(X2s)
    dfu["Pred_RootCause_SVM"] = svm_cls_c2.predict(X2s)
    dfu["Pred_RootCause_PCA"] = log_reg_pca_c2.predict(X2p)
    dfu["KMeans_Cluster_C2"] = kmeans_c2.predict(X2s)
    dfu["Hier_Cluster_C2"] = [hier_predict(row, hier_centroids_c2) for row in X2s]
    if LIGHTGBM_AVAILABLE and lgb_cls_c1 is not None:
        dfu["Pred_SevereDelay_LGBM"] = lgb_cls_c1.predict(X1s)
        dfu["Pred_RootCause_LGBM"] = lgb_cls_c2.predict(X2s)

    # Metrics if labels present
    true_sev = (dfu["ArrDelay"] >= 45).astype(int)
    pred_sev = dfu["Pred_SevereDelay_RF"].astype(int)
    print(f"\nRF accuracy vs ArrDelay≥45 on upload: {(true_sev == pred_sev).mean():.4f}")
    print(f"Predicted severe rate: {pred_sev.mean():.4f} | Actual severe rate: {true_sev.mean():.4f}")
    print("\nSample scored rows:")
    cols_show = [c for c in [
        "Distance", "DepHour", "ArrDelay", "Pred_ArrDelay_mins", "Pred_SevereDelay_RF",
        "Pred_RootCause_RF", "KMeans_Cluster_C1", "Is_Anomaly"
    ] if c in dfu.columns]
    print(dfu[cols_show].head(10).to_string(index=False))

    out_path = "DelayedFlightswupdates.csv"
    ans = input(f"\nAppend these {len(dfu):,} scored rows to {out_path}? [y/N]: ").strip().lower()
    if ans == "y":
        if Path(out_path).exists():
            prev = pd.read_csv(out_path)
            combined = pd.concat([prev, dfu], ignore_index=True)
            print(f"Appending to existing file ({len(prev):,} prior rows).")
        else:
            combined = dfu
        combined.to_csv(out_path, index=False)
        print(f"Saved {len(combined):,} total rows → {out_path}")
    else:
        # still offer a one-off export
        ans2 = input("Save this batch only (overwrite/create DelayedFlightswupdates.csv)? [y/N]: ").strip().lower()
        if ans2 == "y":
            dfu.to_csv(out_path, index=False)
            print(f"Wrote {len(dfu):,} rows → {out_path}")
        else:
            print("Not saved.")


def show_main_menu():
    print("\n" + "="*60)
    print("                    MAIN MENU")
    print("="*60)
    print("1. View Code 1 Metrics & Visuals (Severe Delay Prediction)")
    print("2. View Code 2 Metrics & Visuals (Root Cause Diagnosis)")
    print("3. View Both Code 1 & Code 2")
    print("4. View Correlation Matrix (All Numerical Features)")
    print("5. View PCA Analysis (Dimensionality Reduction)")
    print("6. View SVM Decision Boundary Visualization")
    print("7. View Hierarchical Clustering Analysis")
    print("8. View Apriori Association Rules")
    print("9. Interactive Predictor & Diagnosis")
    print("10. Show Random Hold-Out Test Samples")
    print("11. Extra Models Visuals (NB / IsolationForest / LabelProp / SelfTrain / Ridge-Lasso / LightGBM / DBSCAN)")
    print("12. Upload & Score New Flights → DelayedFlightswupdates.csv")
    print("0. Exit System")
    return input("\nSelect an option (0-12): ").strip()

def show_predictor_menu():
    print("\n" + "="*60)
    print("        FLIGHT PREDICTION & DIAGNOSIS ROUTINE")
    print("="*60)
    print("1. Predict if a flight will be SEVERELY DELAYED (Code 1)")
    print("2. Diagnose the ROOT CAUSE & CLUSTER of a delay (Code 2)")
    print("3. Back to Main Menu")
    print("0. Exit System")
    return input("\nEnter choice (0, 1, 2, or 3): ").strip()

def show_random_samples():
    print("\n" + "="*60)
    print(f"   RANDOM UNSEEN SAMPLES FROM YOUR {test_ratio*100:.0f}% HOLD-OUT TEST DATA")
    print("="*60)
    print("\n--- Code 1 Random Test Samples (Distance, Scheduled Dep Hour, Inbound Delay) ---")
    print(X_test_c1.sample(n=min(5, len(X_test_c1))).to_string())
    print("\n--- Code 2 Random Test Samples (Taxi-Out Time, Air Time, Distance) ---")
    print(X_test_c2.sample(n=min(5, len(X_test_c2))).to_string())

cluster_meanings = {
    0: "Cluster 0: Low Taxi-Out, Short Air Time (Short-haul / Normal Ops)",
    1: "Cluster 1: Long Air Time (Long-haul / Airborne constraints)",
    2: "Cluster 2: High Taxi-Out Time (Ground Congestion / Hub Bottleneck)"
}

def run_interactive_predictor():
    """Inner loop for predictions — option 3 returns to main menu."""
    while True:
        choice = show_predictor_menu()

        if choice == '1':
            print("\n--- [Code 1] Enter Flight Parameters ---")
            try:
                dist = float(input("Enter Route Distance (miles): "))
                hour = float(input("Enter Scheduled Departure Hour (0-23): "))
                lag  = float(input("Enter Previous Leg Delay of Aircraft (mins): "))
            except ValueError:
                print("Invalid number. Returning to predictor menu.")
                continue

            input_df = pd.DataFrame([[dist, hour, lag]], columns=features_c1)
            input_scaled = scaler_c1.transform(input_df)
            input_pca = pca_c1.transform(input_scaled)

            print("\n>>> PREDICTION RESULTS (CODE 1) <<<")
            print(f"• Linear Regression: Expected Delay = {lin_reg_c1.predict(input_scaled)[0]:.1f} mins")
            print(f"• Logistic Regression:  {'DELAYED' if log_reg_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• Decision Tree:        {'DELAYED' if dt_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• Random Forest (BEST): {'DELAYED' if rf_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• Neural Network (MLP): {'DELAYED' if nn_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• SVM (RBF):            {'DELAYED' if svm_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• PCA + Logistic Reg:   {'DELAYED' if log_reg_pca_c1.predict(input_pca)[0] == 1 else 'ON-TIME'}")
            print(f"• K-Means Risk Cluster: Group #{kmeans_c1.predict(input_scaled)[0]}")
            print(f"• Hierarchical Cluster: Group #{hier_predict(input_scaled[0], hier_centroids_c1)}")
            print(f"• Naive Bayes:          {'DELAYED' if nb_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• Isolation Forest:     {'ANOMALY/DELAYED' if iso_c1.predict(input_scaled)[0] == -1 else 'NORMAL'}")
            print(f"• Label Propagation:    {'DELAYED' if lp_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• Self-Training:        {'DELAYED' if st_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
            print(f"• Ridge Reg (mins):     {ridge_c1.predict(input_scaled)[0]:.1f}")
            print(f"• Lasso Reg (mins):     {lasso_c1.predict(input_scaled)[0]:.1f}")
            if LIGHTGBM_AVAILABLE and lgb_cls_c1 is not None:
                print(f"• LightGBM:             {'DELAYED' if lgb_cls_c1.predict(input_scaled)[0] == 1 else 'ON-TIME'}")
                print(f"• LightGBM Reg (mins):  {lgb_reg_c1.predict(input_scaled)[0]:.1f}")

        elif choice == '2':
            print("\n--- [Code 2] Enter Operational Parameters ---")
            try:
                taxi = float(input("Enter Taxi-Out Time (mins): "))
                air  = float(input("Enter Air Time (mins): "))
                dist = float(input("Enter Distance (miles): "))
            except ValueError:
                print("Invalid number. Returning to predictor menu.")
                continue

            input_df = pd.DataFrame([[taxi, air, dist]], columns=features_c2)
            input_scaled = scaler_c2.transform(input_df)
            assigned_cluster = kmeans_c2.predict(input_scaled)[0]
            input_pca = pca_c2.transform(input_scaled)

            print("\n>>> DIAGNOSTIC RESULTS (CODE 2) <<<")
            print(f"• Linear Regression: Expected Carrier Delay = {lin_reg_c2.predict(input_scaled)[0]:.1f} mins")
            print(f"• Logistic Regression Cause: {log_reg_c2.predict(input_scaled)[0]}")
            print(f"• Decision Tree Cause:       {dt_cls_c2.predict(input_scaled)[0]}")
            print(f"• Random Forest Cause:       {rf_cls_c2.predict(input_scaled)[0]}")
            print(f"• Neural Network Cause:      {nn_cls_c2.predict(input_scaled)[0]}")
            print(f"• SVM (RBF) Cause:           {svm_cls_c2.predict(input_scaled)[0]}")
            print(f"• PCA + Logistic Reg Cause:  {log_reg_pca_c2.predict(input_pca)[0]}")
            print(f"• K-Means Profile:           {cluster_meanings.get(assigned_cluster, f'Cluster #{assigned_cluster}')}")
            print(f"• Hierarchical Cluster:      Group #{hier_predict(input_scaled[0], hier_centroids_c2)}")
            print(f"• Naive Bayes Cause:         {nb_cls_c2.predict(input_scaled)[0]}")
            lp_i = int(lp_cls_c2.predict(input_scaled)[0])
            print(f"• Label Propagation Cause:   {_i_to_rc.get(lp_i, lp_i)}")
            st_i = int(st_cls_c2.predict(input_scaled)[0])
            print(f"• Self-Training Cause:       {_i_to_rc.get(st_i, st_i)}")
            print(f"• Ridge Reg (Carrier mins):  {ridge_c2.predict(input_scaled)[0]:.1f}")
            print(f"• Lasso Reg (Carrier mins):  {lasso_c2.predict(input_scaled)[0]:.1f}")
            if LIGHTGBM_AVAILABLE and lgb_cls_c2 is not None:
                print(f"• LightGBM Cause:            {lgb_cls_c2.predict(input_scaled)[0]}")
                print(f"• LightGBM Reg (Carrier):    {lgb_reg_c2.predict(input_scaled)[0]:.1f}")

        elif choice == '3':
            print("\n← Returning to Main Menu...")
            return False  # back to main, do not exit

        elif choice == '0':
            print("\nExiting Flight Analytics System. Have a great day!")
            return True  # signal full exit

        else:
            print("\nInvalid choice. Please enter 0, 1, 2, or 3.")

# ---- Main navigation loop ----
while True:
    eval_choice = show_main_menu()

    if eval_choice == '1':
        evaluate_code_1()
    elif eval_choice == '2':
        evaluate_code_2()
    elif eval_choice == '3':
        evaluate_code_1()
        evaluate_code_2()
    elif eval_choice == '4':
        plot_correlation_matrix()
    elif eval_choice == '5':
        evaluate_pca()
    elif eval_choice == '6':
        evaluate_svm()
    elif eval_choice == '7':
        evaluate_hierarchical()
    elif eval_choice == '8':
        run_apriori()
    elif eval_choice == '9':
        should_exit = run_interactive_predictor()
        if should_exit:
            break
        # else: loop continues → main menu again
    elif eval_choice == '10':
        show_random_samples()
    elif eval_choice == '11':
        evaluate_extra_models()
    elif eval_choice == '12':
        run_upload_and_score()
    elif eval_choice == '0':
        print("\nExiting Flight Analytics System. Have a great day!")
        break
    else:
        print("\nInvalid choice. Please enter a number from 0–12.")

    # After any evaluation (1–8, 10), offer a quick continue prompt
    if eval_choice in {'1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12'}:
        cont = input("\nPress Enter to return to Main Menu (or type 0 then Enter to Exit): ").strip()
        if cont == '0':
            print("\nExiting Flight Analytics System. Have a great day!")
            break