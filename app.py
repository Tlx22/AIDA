import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

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

from scipy.cluster.hierarchy import dendrogram, linkage

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

st.set_page_config(page_title="Flight Delay AI Dashboard", layout="wide")
st.title("✈️ Flight Delay & Root Cause Analytics Dashboard")

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.header("⚙️ Configuration Panel")
train_pct = st.sidebar.slider("Training Data Percentage (%)", min_value=10, max_value=90, value=80, step=5)
test_ratio = (100.0 - train_pct) / 100.0
st.sidebar.markdown(f"**Split Ratio:** {train_pct}% Train / {100-train_pct}% Test")

app_mode = st.sidebar.selectbox(
    "Choose Dashboard View:",
    [
        "Interactive Predictor & Diagnostics",
        "Model Evaluations & Metrics",
        "PCA / SVM / Hierarchical / Apriori",
        "Correlation Matrix"
    ]
)

@st.cache_data
def load_data():
    cols = [
        'ArrDelay', 'CRSDepTime', 'TailNum', 'Distance',
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'TaxiOut', 'AirTime'
    ]
    df = pd.read_csv('DelayedFlights.csv', usecols=cols)
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading DelayedFlights.csv: {e}")
    st.stop()

# =========================================================================
# TRAIN MODELS (incl. PCA, SVM, Hierarchical)
# =========================================================================
@st.cache_resource
def train_models(test_r):
    rng = np.random.RandomState(42)
    SVM_SAMPLE_SIZE = 3000
    HIER_SAMPLE_SIZE = 1500

    # --- CODE 1 ---
    df_c1 = df.copy()
    df_c1['Is_Severe_Delay'] = (df_c1['ArrDelay'] >= 45).astype(int)
    df_c1['DepHour'] = (df_c1['CRSDepTime'] // 100).astype(int)
    df_c1 = df_c1.sort_values(['TailNum', 'CRSDepTime']).reset_index(drop=True)
    df_c1['Inbound_Leg_ArrDelay'] = df_c1.groupby('TailNum')['ArrDelay'].shift(1).fillna(0)

    features_c1 = ['Distance', 'DepHour', 'Inbound_Leg_ArrDelay']
    df_clean_c1 = df_c1[features_c1 + ['ArrDelay', 'Is_Severe_Delay']].dropna()

    X_c1 = df_clean_c1[features_c1]
    y_class_c1 = df_clean_c1['Is_Severe_Delay']
    y_reg_c1 = df_clean_c1['ArrDelay']

    X_train_c1, X_test_c1, y_train_class_c1, y_test_class_c1, y_train_reg_c1, y_test_reg_c1 = train_test_split(
        X_c1, y_class_c1, y_reg_c1, test_size=test_r, random_state=42
    )

    scaler_c1 = StandardScaler()
    X_train_scaled_c1 = scaler_c1.fit_transform(X_train_c1)
    X_test_scaled_c1 = scaler_c1.transform(X_test_c1)

    lin_reg_c1 = LinearRegression().fit(X_train_scaled_c1, y_train_reg_c1)
    log_reg_c1 = LogisticRegression(class_weight='balanced', max_iter=500).fit(X_train_scaled_c1, y_train_class_c1)
    dt_cls_c1  = DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight='balanced', random_state=42).fit(X_train_scaled_c1, y_train_class_c1)
    rf_cls_c1  = RandomForestClassifier(n_estimators=30, max_depth=5, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_scaled_c1, y_train_class_c1)
    kmeans_c1  = KMeans(n_clusters=2, random_state=42, n_init=5).fit(X_train_scaled_c1)
    nn_cls_c1  = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=100, random_state=42).fit(X_train_scaled_c1, y_train_class_c1)

    pca_c1 = PCA(n_components=2, random_state=42).fit(X_train_scaled_c1)
    X_train_pca_c1 = pca_c1.transform(X_train_scaled_c1)
    X_test_pca_c1  = pca_c1.transform(X_test_scaled_c1)
    log_reg_pca_c1 = LogisticRegression(class_weight='balanced', max_iter=500).fit(X_train_pca_c1, y_train_class_c1)

    svm_idx_c1 = rng.choice(len(X_train_scaled_c1), size=min(SVM_SAMPLE_SIZE, len(X_train_scaled_c1)), replace=False)
    svm_cls_c1 = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42).fit(
        X_train_scaled_c1[svm_idx_c1], y_train_class_c1.iloc[svm_idx_c1]
    )

    hier_idx_c1 = rng.choice(len(X_train_scaled_c1), size=min(HIER_SAMPLE_SIZE, len(X_train_scaled_c1)), replace=False)
    X_hier_sample_c1 = X_train_scaled_c1[hier_idx_c1]
    hier_cls_c1 = AgglomerativeClustering(n_clusters=2).fit(X_hier_sample_c1)
    hier_centroids_c1 = np.array([X_hier_sample_c1[hier_cls_c1.labels_ == i].mean(axis=0) for i in range(2)])

    # --- CODE 2 ---
    df_c2 = df.copy()
    delay_cols = ['WeatherDelay', 'NASDelay', 'CarrierDelay']
    df_c2[delay_cols] = df_c2[delay_cols].fillna(0)

    def assign_root_cause(row):
        delays = {'Carrier': row['CarrierDelay'], 'Weather': row['WeatherDelay'], 'NAS/Hub': row['NASDelay']}
        max_delay = max(delays.values())
        if max_delay == 0:
            return 'Carrier'
        return max(key for key, val in delays.items() if val == max_delay)

    df_c2['RootCause'] = df_c2.apply(assign_root_cause, axis=1)
    features_c2 = ['TaxiOut', 'AirTime', 'Distance']
    df_clean_c2 = df_c2[features_c2 + ['RootCause', 'CarrierDelay']].dropna()

    X_c2 = df_clean_c2[features_c2]
    y_class_c2 = df_clean_c2['RootCause']
    y_reg_c2 = df_clean_c2['CarrierDelay']

    X_train_c2, X_test_c2, y_train_class_c2, y_test_class_c2, y_train_reg_c2, y_test_reg_c2 = train_test_split(
        X_c2, y_class_c2, y_reg_c2, test_size=test_r, random_state=42
    )

    scaler_c2 = StandardScaler()
    X_train_scaled_c2 = scaler_c2.fit_transform(X_train_c2)
    X_test_scaled_c2 = scaler_c2.transform(X_test_c2)

    lin_reg_c2 = LinearRegression().fit(X_train_scaled_c2, y_train_reg_c2)
    log_reg_c2 = LogisticRegression(class_weight='balanced', max_iter=500).fit(X_train_scaled_c2, y_train_class_c2)
    dt_cls_c2  = DecisionTreeClassifier(criterion='gini', max_depth=3, class_weight='balanced', random_state=42).fit(X_train_scaled_c2, y_train_class_c2)
    rf_cls_c2  = RandomForestClassifier(n_estimators=30, max_depth=5, class_weight='balanced', n_jobs=-1, random_state=42).fit(X_train_scaled_c2, y_train_class_c2)
    kmeans_c2  = KMeans(n_clusters=3, random_state=42, n_init=5).fit(X_train_scaled_c2)
    nn_cls_c2  = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=100, random_state=42).fit(X_train_scaled_c2, y_train_class_c2)

    pca_c2 = PCA(n_components=2, random_state=42).fit(X_train_scaled_c2)
    X_train_pca_c2 = pca_c2.transform(X_train_scaled_c2)
    X_test_pca_c2  = pca_c2.transform(X_test_scaled_c2)
    log_reg_pca_c2 = LogisticRegression(class_weight='balanced', max_iter=500).fit(X_train_pca_c2, y_train_class_c2)

    svm_idx_c2 = rng.choice(len(X_train_scaled_c2), size=min(SVM_SAMPLE_SIZE, len(X_train_scaled_c2)), replace=False)
    svm_cls_c2 = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42).fit(
        X_train_scaled_c2[svm_idx_c2], y_train_class_c2.iloc[svm_idx_c2]
    )

    hier_idx_c2 = rng.choice(len(X_train_scaled_c2), size=min(HIER_SAMPLE_SIZE, len(X_train_scaled_c2)), replace=False)
    X_hier_sample_c2 = X_train_scaled_c2[hier_idx_c2]
    hier_cls_c2 = AgglomerativeClustering(n_clusters=3).fit(X_hier_sample_c2)
    hier_centroids_c2 = np.array([X_hier_sample_c2[hier_cls_c2.labels_ == i].mean(axis=0) for i in range(3)])

    return (
        (X_test_c1, y_test_class_c1, y_test_reg_c1, X_test_scaled_c1, scaler_c1, features_c1,
         lin_reg_c1, log_reg_c1, dt_cls_c1, rf_cls_c1, kmeans_c1, nn_cls_c1),
        (X_test_c2, y_test_class_c2, y_test_reg_c2, X_test_scaled_c2, scaler_c2, features_c2,
         lin_reg_c2, log_reg_c2, dt_cls_c2, rf_cls_c2, kmeans_c2, nn_cls_c2),
        (pca_c1, X_test_pca_c1, log_reg_pca_c1, svm_cls_c1, hier_cls_c1, X_hier_sample_c1, hier_centroids_c1),
        (pca_c2, X_test_pca_c2, log_reg_pca_c2, svm_cls_c2, hier_cls_c2, X_hier_sample_c2, hier_centroids_c2),
        df_clean_c1
    )

with st.spinner("Training models (incl. PCA, SVM, Hierarchical)... Please wait."):
    c1_bundle, c2_bundle, extra_c1, extra_c2, df_clean_c1 = train_models(test_ratio)

(X_test_c1, y_test_class_c1, y_test_reg_c1, X_test_scaled_c1, scaler_c1, features_c1,
 lin_reg_c1, log_reg_c1, dt_cls_c1, rf_cls_c1, kmeans_c1, nn_cls_c1) = c1_bundle
(X_test_c2, y_test_class_c2, y_test_reg_c2, X_test_scaled_c2, scaler_c2, features_c2,
 lin_reg_c2, log_reg_c2, dt_cls_c2, rf_cls_c2, kmeans_c2, nn_cls_c2) = c2_bundle
(pca_c1, X_test_pca_c1, log_reg_pca_c1, svm_cls_c1, hier_cls_c1, X_hier_sample_c1, hier_centroids_c1) = extra_c1
(pca_c2, X_test_pca_c2, log_reg_pca_c2, svm_cls_c2, hier_cls_c2, X_hier_sample_c2, hier_centroids_c2) = extra_c2

# =========================================================================
# CORRELATION MATRIX
# =========================================================================
if app_mode == "Correlation Matrix":
    st.header("🔗 Correlation Matrix — All Numerical Features")
    st.write("Pearson correlation on the cloud-optimized column subset / capped sample.")
    numeric_df = df.select_dtypes(include=[np.number])
    if 'Unnamed: 0' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['Unnamed: 0'])
    corr_matrix = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of All Numerical Features (DelayedFlights)', fontsize=16)
    st.pyplot(fig); plt.close(fig)
    if 'ArrDelay' in corr_matrix.columns:
        st.subheader("Top Features Correlated with Arrival Delay (ArrDelay)")
        st.dataframe(corr_matrix['ArrDelay'].sort_values(ascending=False).rename("Correlation"))

# =========================================================================
# INTERACTIVE PREDICTOR
# =========================================================================
elif app_mode == "Interactive Predictor & Diagnostics":
    st.header("🎮 Live Flight Predictor & Root Cause Diagnosis")
    analysis_choice = st.radio(
        "Select Analysis Routine:",
        ["Code 1: Predict Severe Delay & Risk Profile", "Code 2: Diagnose Delay Root Cause & Operational Cluster"],
        horizontal=True
    )

    if "Code 1" in analysis_choice:
        st.subheader("Enter Flight Parameters (Code 1)")
        col1, col2, col3 = st.columns(3)
        with col1:
            dist = st.number_input("Route Distance (miles)", min_value=10.0, max_value=5000.0, value=800.0)
        with col2:
            hour = st.slider("Scheduled Departure Hour (0-23)", 0, 23, 14)
        with col3:
            lag = st.number_input("Previous Leg Delay of Aircraft (mins)", min_value=0.0, max_value=1000.0, value=25.0)

        input_df = pd.DataFrame([[dist, hour, lag]], columns=features_c1)
        input_scaled = scaler_c1.transform(input_df)
        input_pca = pca_c1.transform(input_scaled)

        st.markdown("---")
        st.subheader(">>> Prediction Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Linear Regression (Delay)", f"{lin_reg_c1.predict(input_scaled)[0]:.1f} mins")
        res_col1.metric("Logistic Regression", "DELAYED ⚠️" if log_reg_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        res_col2.metric("Decision Tree", "DELAYED ⚠️" if dt_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        res_col2.metric("Random Forest (BEST)", "DELAYED ⚠️" if rf_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        res_col3.metric("Neural Network (MLP)", "DELAYED ⚠️" if nn_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        res_col3.metric("K-Means Risk Cluster", f"Group #{kmeans_c1.predict(input_scaled)[0]}")

        st.markdown("##### Additional Models")
        e1, e2, e3 = st.columns(3)
        e1.metric("PCA + Logistic Reg", "DELAYED ⚠️" if log_reg_pca_c1.predict(input_pca)[0] == 1 else "ON-TIME 🟢")
        e2.metric("SVM (RBF)", "DELAYED ⚠️" if svm_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        hier_pred = int(np.argmin(np.linalg.norm(hier_centroids_c1 - input_scaled, axis=1)))
        e3.metric("Hierarchical Cluster", f"Group #{hier_pred}")

        st.markdown("---")
        st.subheader("📊 Comprehensive Model Visualizations & Insights")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            y_reg_pred_test = lin_reg_c1.predict(X_test_scaled_c1)
            ax.scatter(y_test_reg_c1, y_reg_pred_test, alpha=0.3, color='blue')
            ax.plot([y_test_reg_c1.min(), y_test_reg_c1.max()], [y_test_reg_c1.min(), y_test_reg_c1.max()], 'r--', lw=2)
            ax.set_title('Multi-Linear Reg: Actual vs Predicted Delay')
            ax.set_xlabel('Actual Delay'); ax.set_ylabel('Predicted Delay')
            st.pyplot(fig); plt.close(fig)
        with v_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=list(log_reg_c1.coef_[0]), y=features_c1, ax=ax, palette='viridis', hue=features_c1, legend=False)
            ax.set_title('Logistic Regression: Feature Weights')
            st.pyplot(fig); plt.close(fig)

        v_col3, v_col4 = st.columns(2)
        with v_col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            pd.Series(rf_cls_c1.feature_importances_, index=features_c1).plot(kind='barh', ax=ax, color='teal')
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig); plt.close(fig)
        with v_col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=X_test_c1['Distance'], y=X_test_c1['Inbound_Leg_ArrDelay'],
                            hue=kmeans_c1.predict(X_test_scaled_c1), palette='Set1', alpha=0.6, ax=ax)
            ax.set_title('K-Means Clusters')
            ax.set_xlabel('Distance'); ax.set_ylabel('Inbound Leg ArrDelay')
            st.pyplot(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(nn_cls_c1.loss_curve_, color='purple', lw=2)
        ax.set_title('Neural Network Loss Convergence')
        ax.set_xlabel('Iterations / Epochs'); ax.set_ylabel('Loss')
        st.pyplot(fig); plt.close(fig)

        st.write("#### Decision Tree Diagram")
        fig, ax = plt.subplots(figsize=(18, 9))
        plot_tree(dt_cls_c1, feature_names=features_c1, class_names=['On-Time', 'Delayed'],
                  filled=True, ax=ax, fontsize=10, rounded=True, precision=2)
        st.pyplot(fig); plt.close(fig)

    else:
        st.subheader("Enter Operational Parameters (Code 2)")
        col1, col2, col3 = st.columns(3)
        with col1:
            taxi = st.number_input("Taxi-Out Time (mins)", min_value=0.0, max_value=300.0, value=20.0)
        with col2:
            air = st.number_input("Air Time (mins)", min_value=0.0, max_value=1000.0, value=150.0)
        with col3:
            dist = st.number_input("Distance (miles)", min_value=10.0, max_value=5000.0, value=900.0)

        input_df = pd.DataFrame([[taxi, air, dist]], columns=features_c2)
        input_scaled = scaler_c2.transform(input_df)
        input_pca = pca_c2.transform(input_scaled)
        assigned_cluster = kmeans_c2.predict(input_scaled)[0]
        cluster_meanings = {
            0: "Cluster 0: Low Taxi-Out, Short Air Time (Short-haul / Normal Ops)",
            1: "Cluster 1: Long Air Time (Long-haul / Airborne constraints)",
            2: "Cluster 2: High Taxi-Out Time (Ground Congestion / Hub Bottleneck)"
        }

        st.markdown("---")
        st.subheader(">>> Diagnostic Results")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Linear Regression (Carrier Delay)", f"{lin_reg_c2.predict(input_scaled)[0]:.1f} mins")
        res_col1.metric("Logistic Regression Cause", str(log_reg_c2.predict(input_scaled)[0]))
        res_col1.metric("Decision Tree Cause", str(dt_cls_c2.predict(input_scaled)[0]))
        res_col2.metric("Random Forest Cause", str(rf_cls_c2.predict(input_scaled)[0]))
        res_col2.metric("Neural Network Cause", str(nn_cls_c2.predict(input_scaled)[0]))
        res_col2.metric("K-Means Profile", cluster_meanings.get(assigned_cluster, f"Cluster #{assigned_cluster}"))

        st.markdown("##### Additional Models")
        e1, e2, e3 = st.columns(3)
        e1.metric("PCA + Logistic Reg", str(log_reg_pca_c2.predict(input_pca)[0]))
        e2.metric("SVM (RBF)", str(svm_cls_c2.predict(input_scaled)[0]))
        hier_pred = int(np.argmin(np.linalg.norm(hier_centroids_c2 - input_scaled, axis=1)))
        e3.metric("Hierarchical Cluster", f"Group #{hier_pred}")

        st.markdown("---")
        st.subheader("📊 Comprehensive Diagnostic Visualizations & Insights")
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            y_reg_pred_test_c2 = lin_reg_c2.predict(X_test_scaled_c2)
            ax.scatter(y_test_reg_c2, y_reg_pred_test_c2, alpha=0.3, color='crimson')
            ax.plot([y_test_reg_c2.min(), y_test_reg_c2.max()], [y_test_reg_c2.min(), y_test_reg_c2.max()], 'k--', lw=2)
            ax.set_title('Multi-Linear Reg: Estimated Carrier Delay')
            ax.set_xlabel('Actual Carrier Delay'); ax.set_ylabel('Predicted Carrier Delay')
            st.pyplot(fig); plt.close(fig)
        with v_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            coef_df = pd.DataFrame(log_reg_c2.coef_, index=log_reg_c2.classes_, columns=features_c2)
            sns.heatmap(coef_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Logistic Reg: Feature Weights Heatmap")
            st.pyplot(fig); plt.close(fig)

        v_col3, v_col4 = st.columns(2)
        with v_col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            pd.Series(rf_cls_c2.feature_importances_, index=features_c2).plot(kind='bar', color='darkgreen', ax=ax, rot=0)
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig); plt.close(fig)
        with v_col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=X_test_c2['TaxiOut'], y=X_test_c2['AirTime'],
                            hue=kmeans_c2.predict(X_test_scaled_c2), palette='Set2', alpha=0.6, ax=ax)
            ax.set_title('K-Means Clusters')
            ax.set_xlabel('TaxiOut'); ax.set_ylabel('AirTime')
            st.pyplot(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(nn_cls_c2.loss_curve_, color='orange', lw=2)
        ax.set_title('Neural Network Loss Convergence')
        ax.set_xlabel('Iterations / Epochs'); ax.set_ylabel('Loss')
        st.pyplot(fig); plt.close(fig)

        st.write("#### Decision Tree Breakdown")
        fig, ax = plt.subplots(figsize=(20, 9))
        plot_tree(dt_cls_c2, feature_names=features_c2, class_names=list(dt_cls_c2.classes_),
                  filled=True, ax=ax, fontsize=10, rounded=True, precision=2)
        st.pyplot(fig); plt.close(fig)

# =========================================================================
# MODEL EVALUATIONS (includes SVM)
# =========================================================================
elif app_mode == "Model Evaluations & Metrics":
    st.header("📊 Model Performance & Evaluation Metrics")
    tab1, tab2 = st.tabs(["Code 1: Severe Delay Metrics", "Code 2: Root Cause Metrics"])

    with tab1:
        st.subheader("Severe Delay Classification & Regression Reports")
        y_reg_pred_c1 = lin_reg_c1.predict(X_test_scaled_c1)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Linear Regression R2 Score", f"{r2_score(y_test_reg_c1, y_reg_pred_c1):.4f}")
        col_m2.metric("Linear Regression RMSE", f"{np.sqrt(mean_squared_error(y_test_reg_c1, y_reg_pred_c1)):.2f} mins")
        st.markdown("---")

        classifiers_c1 = {
            "Logistic Regression": log_reg_c1,
            "Decision Tree": dt_cls_c1,
            "Random Forest": rf_cls_c1,
            "Neural Network (MLP)": nn_cls_c1,
            "SVM (RBF)": svm_cls_c1
        }
        for name, clf in classifiers_c1.items():
            st.write(f"#### {name}")
            y_pred = clf.predict(X_test_scaled_c1)
            acc = accuracy_score(y_test_class_c1, y_pred)
            has_proba = hasattr(clf, "predict_proba")
            roc_auc = roc_auc_score(y_test_class_c1, clf.predict_proba(X_test_scaled_c1)[:, 1]) if has_proba else float('nan')
            f1 = f1_score(y_test_class_c1, y_pred, average='binary')
            sub_c1, sub_c2, sub_c3 = st.columns(3)
            sub_c1.metric("Accuracy", f"{acc:.4f}")
            sub_c2.metric("ROC-AUC Score", f"{roc_auc:.4f}")
            sub_c3.metric("F1 Score", f"{f1:.4f}")
            report = classification_report(y_test_class_c1, y_pred, target_names=['On-Time', 'Delayed'], output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
            if has_proba:
                y_score = clf.predict_proba(X_test_scaled_c1)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_class_c1, y_score)
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.3f}")
                ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Chance")
                ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"ROC Curve: {name}")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc); plt.close(fig_roc)
            st.markdown("---")
        with st.expander("🎲 View Random Unseen Test Samples (Code 1)"):
            st.dataframe(X_test_c1.sample(n=min(5, len(X_test_c1)), random_state=None))

    with tab2:
        st.subheader("Root Cause Diagnosis Performance Reports")
        y_reg_pred_c2 = lin_reg_c2.predict(X_test_scaled_c2)
        col_cm1, col_cm2 = st.columns(2)
        col_cm1.metric("Carrier Delay Reg. R2 Score", f"{r2_score(y_test_reg_c2, y_reg_pred_c2):.4f}")
        col_cm2.metric("Carrier Delay Reg. RMSE", f"{np.sqrt(mean_squared_error(y_test_reg_c2, y_reg_pred_c2)):.2f} mins")
        st.markdown("---")

        classifiers_c2 = {
            "Logistic Regression": log_reg_c2,
            "Decision Tree": dt_cls_c2,
            "Random Forest": rf_cls_c2,
            "Neural Network (MLP)": nn_cls_c2,
            "SVM (RBF)": svm_cls_c2
        }
        all_classes_c2 = sorted(y_test_class_c2.unique())
        y_test_bin_c2 = label_binarize(y_test_class_c2, classes=all_classes_c2)

        for name, clf in classifiers_c2.items():
            st.write(f"#### {name}")
            y_pred = clf.predict(X_test_scaled_c2)
            acc = accuracy_score(y_test_class_c2, y_pred)
            has_proba = hasattr(clf, "predict_proba")
            f1 = f1_score(y_test_class_c2, y_pred, average='macro')
            if has_proba:
                proba = pd.DataFrame(clf.predict_proba(X_test_scaled_c2), columns=clf.classes_).reindex(columns=all_classes_c2).values
                macro_auc = roc_auc_score(y_test_bin_c2, proba, average='macro', multi_class='ovr')
            else:
                macro_auc = float('nan')
            sub_c1, sub_c2, sub_c3 = st.columns(3)
            sub_c1.metric("Accuracy", f"{acc:.4f}")
            sub_c2.metric("Macro ROC-AUC (OvR)", f"{macro_auc:.4f}")
            sub_c3.metric("Macro F1 Score", f"{f1:.4f}")
            report = classification_report(y_test_class_c2, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
            if has_proba:
                fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                for i, cls in enumerate(all_classes_c2):
                    fpr, tpr, _ = roc_curve(y_test_bin_c2[:, i], proba[:, i])
                    ax_roc.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
                ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label="Chance")
                ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"ROC Curves (One-vs-Rest): {name}")
                ax_roc.legend(loc="lower right", fontsize=8)
                st.pyplot(fig_roc); plt.close(fig_roc)
            st.markdown("---")
        with st.expander("🎲 View Random Unseen Test Samples (Code 2)"):
            st.dataframe(X_test_c2.sample(n=min(5, len(X_test_c2)), random_state=None))

# =========================================================================
# PCA / SVM / HIERARCHICAL / APRIORI
# =========================================================================
elif app_mode == "PCA / SVM / Hierarchical / Apriori":
    st.header("🔬 Advanced ML Techniques: PCA · SVM · Hierarchical · Apriori")
    tab_pca, tab_svm, tab_hier, tab_apri = st.tabs(["PCA", "SVM (RBF)", "Hierarchical Clustering", "Apriori Rules"])

    all_classes_c2 = sorted(y_test_class_c2.unique())
    y_test_bin_c2 = label_binarize(y_test_class_c2, classes=all_classes_c2)

    with tab_pca:
        st.subheader("PCA — Dimensionality Reduction Evaluation")
        st.write("**Code 1 (Severe Delay)**")
        st.write(f"Explained variance ratio (PC1, PC2): `{pca_c1.explained_variance_ratio_}`")
        st.write(f"Cumulative variance explained: **{pca_c1.explained_variance_ratio_.sum()*100:.2f}%**")
        y_pred_pca_c1 = log_reg_pca_c1.predict(X_test_pca_c1)
        acc_pca_c1 = accuracy_score(y_test_class_c1, y_pred_pca_c1)
        roc_auc_pca_c1 = roc_auc_score(y_test_class_c1, log_reg_pca_c1.predict_proba(X_test_pca_c1)[:, 1])
        f1_pca_c1 = f1_score(y_test_class_c1, y_pred_pca_c1, average='binary')
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc_pca_c1:.4f}")
        m2.metric("ROC-AUC", f"{roc_auc_pca_c1:.4f}")
        m3.metric("F1 Score", f"{f1_pca_c1:.4f}")
        st.dataframe(pd.DataFrame(classification_report(y_test_class_c1, y_pred_pca_c1, target_names=['On-Time', 'Delayed'], output_dict=True, zero_division=0)).transpose())

        st.write("**Code 2 (Root Cause)**")
        st.write(f"Explained variance ratio (PC1, PC2): `{pca_c2.explained_variance_ratio_}`")
        st.write(f"Cumulative variance explained: **{pca_c2.explained_variance_ratio_.sum()*100:.2f}%**")
        y_pred_pca_c2 = log_reg_pca_c2.predict(X_test_pca_c2)
        acc_pca_c2 = accuracy_score(y_test_class_c2, y_pred_pca_c2)
        f1_pca_c2 = f1_score(y_test_class_c2, y_pred_pca_c2, average='macro')
        proba_pca_c2 = pd.DataFrame(log_reg_pca_c2.predict_proba(X_test_pca_c2), columns=log_reg_pca_c2.classes_).reindex(columns=all_classes_c2).values
        macro_auc_pca_c2 = roc_auc_score(y_test_bin_c2, proba_pca_c2, average='macro', multi_class='ovr')
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc_pca_c2:.4f}")
        m2.metric("Macro ROC-AUC", f"{macro_auc_pca_c2:.4f}")
        m3.metric("Macro F1", f"{f1_pca_c2:.4f}")
        st.dataframe(pd.DataFrame(classification_report(y_test_class_c2, y_pred_pca_c2, output_dict=True, zero_division=0)).transpose())

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('PCA — Dimensionality Reduction Analysis', fontsize=14)
        axes[0, 0].bar(['PC1', 'PC2'], pca_c1.explained_variance_ratio_, color='slateblue')
        axes[0, 0].set_title('Code 1: Explained Variance Ratio'); axes[0, 0].set_ylabel('Variance Ratio')
        sc1 = axes[0, 1].scatter(X_test_pca_c1[:, 0], X_test_pca_c1[:, 1], c=y_test_class_c1, cmap='coolwarm', alpha=0.3, s=8)
        axes[0, 1].set_title('Code 1: Test Set on PC1 vs PC2'); axes[0, 1].set_xlabel('PC1'); axes[0, 1].set_ylabel('PC2')
        plt.colorbar(sc1, ax=axes[0, 1], label='Is_Severe_Delay')
        axes[1, 0].bar(['PC1', 'PC2'], pca_c2.explained_variance_ratio_, color='seagreen')
        axes[1, 0].set_title('Code 2: Explained Variance Ratio'); axes[1, 0].set_ylabel('Variance Ratio')
        for cls in sorted(y_test_class_c2.unique()):
            mask = (y_test_class_c2 == cls).values
            axes[1, 1].scatter(X_test_pca_c2[mask, 0], X_test_pca_c2[mask, 1], alpha=0.3, s=8, label=cls)
        axes[1, 1].set_title('Code 2: Test Set on PC1 vs PC2'); axes[1, 1].set_xlabel('PC1'); axes[1, 1].set_ylabel('PC2')
        axes[1, 1].legend()
        plt.tight_layout(pad=2.0)
        st.pyplot(fig); plt.close(fig)

        st.write("#### ROC Curve — PCA + Logistic Regression (Code 1)")
        y_score_pca = log_reg_pca_c1.predict_proba(X_test_pca_c1)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_class_c1, y_score_pca)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc_pca_c1:.3f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC: PCA + Logistic Regression (Code 1)")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc); plt.close(fig_roc)

    with tab_svm:
        st.subheader("SVM (RBF Kernel) Evaluation")
        st.info("Trained on a random subsample of 3,000 rows (cloud RAM limit). Metrics on full hold-out test set.")

        st.write("**Code 1 — Severe Delay**")
        y_pred_svm_c1 = svm_cls_c1.predict(X_test_scaled_c1)
        acc_svm = accuracy_score(y_test_class_c1, y_pred_svm_c1)
        roc_auc_svm = roc_auc_score(y_test_class_c1, svm_cls_c1.predict_proba(X_test_scaled_c1)[:, 1])
        f1_svm = f1_score(y_test_class_c1, y_pred_svm_c1, average='binary')
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc_svm:.4f}")
        m2.metric("ROC-AUC", f"{roc_auc_svm:.4f}")
        m3.metric("F1 Score", f"{f1_svm:.4f}")
        st.dataframe(pd.DataFrame(classification_report(y_test_class_c1, y_pred_svm_c1, target_names=['On-Time', 'Delayed'], output_dict=True, zero_division=0)).transpose())

        y_score = svm_cls_c1.predict_proba(X_test_scaled_c1)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_class_c1, y_score)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc_svm:.3f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve: SVM (RBF) — Code 1")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc); plt.close(fig_roc)

        st.write("#### Decision Boundary (Code 1, projected to PCA space)")
        n_viz = min(2000, len(X_test_pca_c1))
        X_viz = X_test_pca_c1[:n_viz]
        y_viz = y_test_class_c1.iloc[:n_viz]
        svm_viz = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42).fit(X_viz, y_viz)
        xx, yy = np.meshgrid(
            np.linspace(X_test_pca_c1[:, 0].min(), X_test_pca_c1[:, 0].max(), 120),
            np.linspace(X_test_pca_c1[:, 1].min(), X_test_pca_c1[:, 1].max(), 150)
        )
        Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        sc = ax.scatter(X_test_pca_c1[:, 0], X_test_pca_c1[:, 1], c=y_test_class_c1, cmap='coolwarm', edgecolor='k', s=8, alpha=0.5)
        ax.set_title('Code 1: SVM (RBF) Decision Boundary in PCA Space')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        plt.colorbar(sc, ax=ax, label='Is_Severe_Delay')
        st.pyplot(fig); plt.close(fig)

        st.write("**Code 2 — Root Cause**")
        y_pred_svm_c2 = svm_cls_c2.predict(X_test_scaled_c2)
        acc_svm2 = accuracy_score(y_test_class_c2, y_pred_svm_c2)
        f1_svm2 = f1_score(y_test_class_c2, y_pred_svm_c2, average='macro')
        proba_svm2 = pd.DataFrame(svm_cls_c2.predict_proba(X_test_scaled_c2), columns=svm_cls_c2.classes_).reindex(columns=all_classes_c2).values
        macro_auc_svm2 = roc_auc_score(y_test_bin_c2, proba_svm2, average='macro', multi_class='ovr')
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc_svm2:.4f}")
        m2.metric("Macro ROC-AUC", f"{macro_auc_svm2:.4f}")
        m3.metric("Macro F1", f"{f1_svm2:.4f}")
        st.dataframe(pd.DataFrame(classification_report(y_test_class_c2, y_pred_svm_c2, output_dict=True, zero_division=0)).transpose())

        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        for i, cls in enumerate(all_classes_c2):
            fpr, tpr, _ = roc_curve(y_test_bin_c2[:, i], proba_svm2[:, i])
            ax_roc.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6)
        ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves (OvR): SVM — Code 2")
        ax_roc.legend(loc="lower right", fontsize=8)
        st.pyplot(fig_roc); plt.close(fig_roc)

    with tab_hier:
        st.subheader("Hierarchical (Agglomerative) Clustering Evaluation")
        st.info("Trained on a random subsample of 1,500 rows (O(n²) distance matrix / cloud RAM).")
        sil_c1 = silhouette_score(X_hier_sample_c1, hier_cls_c1.labels_)
        sil_c2 = silhouette_score(X_hier_sample_c2, hier_cls_c2.labels_)
        m1, m2 = st.columns(2)
        m1.metric("Code 1 Silhouette (2 clusters)", f"{sil_c1:.4f}")
        m2.metric("Code 2 Silhouette (3 clusters)", f"{sil_c2:.4f}")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hierarchical (Agglomerative) Clustering Analysis', fontsize=14)
        dendro_idx_c1 = np.random.RandomState(1).choice(len(X_hier_sample_c1), size=min(40, len(X_hier_sample_c1)), replace=False)
        Z1 = linkage(X_hier_sample_c1[dendro_idx_c1], method='ward')
        dendrogram(Z1, ax=axes[0, 0])
        axes[0, 0].set_title('Code 1: Dendrogram (40-point sample)')
        axes[0, 0].set_xlabel('Sample Index'); axes[0, 0].set_ylabel('Distance')
        axes[0, 1].scatter(X_hier_sample_c1[:, 0], X_hier_sample_c1[:, 2], c=hier_cls_c1.labels_, cmap='Set1', alpha=0.6)
        axes[0, 1].set_title('Code 1: Clusters (Distance vs Inbound Delay, scaled)')
        axes[0, 1].set_xlabel('Distance (scaled)'); axes[0, 1].set_ylabel('Inbound Leg ArrDelay (scaled)')
        dendro_idx_c2 = np.random.RandomState(1).choice(len(X_hier_sample_c2), size=min(40, len(X_hier_sample_c2)), replace=False)
        Z2 = linkage(X_hier_sample_c2[dendro_idx_c2], method='ward')
        dendrogram(Z2, ax=axes[1, 0])
        axes[1, 0].set_title('Code 2: Dendrogram (40-point sample)')
        axes[1, 0].set_xlabel('Sample Index'); axes[1, 0].set_ylabel('Distance')
        axes[1, 1].scatter(X_hier_sample_c2[:, 0], X_hier_sample_c2[:, 1], c=hier_cls_c2.labels_, cmap='Set2', alpha=0.6)
        axes[1, 1].set_title('Code 2: Clusters (TaxiOut vs AirTime, scaled)')
        axes[1, 1].set_xlabel('TaxiOut (scaled)'); axes[1, 1].set_ylabel('AirTime (scaled)')
        plt.tight_layout(pad=2.0)
        st.pyplot(fig); plt.close(fig)

    with tab_apri:
        st.subheader("Apriori — Association Rule Mining")
        if not MLXTEND_AVAILABLE:
            st.warning("The `mlxtend` package is required for Apriori. Install with: `pip install mlxtend`")
        else:
            apriori_df = df_clean_c1.sample(n=min(20000, len(df_clean_c1)), random_state=42).copy()
            def bin_distance(d):
                if d < 500: return 'Distance_Short'
                elif d < 1500: return 'Distance_Medium'
                return 'Distance_Long'
            def bin_hour(h):
                if 5 <= h <= 11: return 'Dep_Morning'
                elif 12 <= h <= 17: return 'Dep_Afternoon'
                elif 18 <= h <= 21: return 'Dep_Evening'
                return 'Dep_Night'
            def bin_inbound_delay(x):
                if x <= 0: return 'InboundDelay_None'
                elif x <= 30: return 'InboundDelay_Minor'
                return 'InboundDelay_Major'
            apriori_df['DistanceBin'] = apriori_df['Distance'].apply(bin_distance)
            apriori_df['DepHourBin'] = apriori_df['DepHour'].apply(bin_hour)
            apriori_df['InboundDelayBin'] = apriori_df['Inbound_Leg_ArrDelay'].apply(bin_inbound_delay)
            apriori_df['DelayStatus'] = apriori_df['Is_Severe_Delay'].map({0: 'OnTime', 1: 'SevereDelay'})
            basket = pd.get_dummies(apriori_df[['DistanceBin', 'DepHourBin', 'InboundDelayBin', 'DelayStatus']])
            frequent_itemsets = apriori(basket, min_support=0.03, use_colnames=True)
            rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
            delay_rules = rules[rules['consequents'].astype(str).str.contains('SevereDelay')].sort_values('lift', ascending=False)
            st.write(f"Found **{len(frequent_itemsets)}** frequent itemsets and **{len(rules)}** total association rules.")
            st.write("Top rules predicting **SEVERE DELAY** (sorted by lift):")
            if len(delay_rules) > 0:
                display_rules = delay_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).copy()
                display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                st.dataframe(display_rules)
            else:
                st.info("No rules with SevereDelay consequent found at current support/lift thresholds.")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle('Apriori — Association Rule Mining (Flight Delay Patterns)', fontsize=14)
            if len(rules) > 0:
                axes[0].scatter(rules['support'], rules['confidence'], s=rules['lift']*40, alpha=0.5, c=rules['lift'], cmap='viridis')
                axes[0].set_xlabel('Support'); axes[0].set_ylabel('Confidence')
                axes[0].set_title('All Rules: Support vs Confidence (bubble = Lift)')
            if len(delay_rules) > 0:
                top10 = delay_rules.head(10).iloc[::-1]
                labels = [f"{list(a)} → {list(c)}" for a, c in zip(top10['antecedents'], top10['consequents'])]
                axes[1].barh(labels, top10['lift'], color='darkorange')
                axes[1].set_xlabel('Lift')
                axes[1].set_title('Top 10 Rules Predicting Severe Delay (by Lift)')
            plt.tight_layout(pad=2.0)
            st.pyplot(fig); plt.close(fig)