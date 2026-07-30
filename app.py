import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier

# Preprocessing & Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    mean_squared_error, 
    r2_score,
    roc_auc_score
)

# Page Layout Config
st.set_page_config(page_title="Flight Delay AI Dashboard", layout="wide")

st.title("✈️ Flight Delay & Root Cause Analytics Dashboard")

# =========================================================================
# SIDEBAR CONTROLS
# =========================================================================
st.sidebar.header("⚙️ Configuration Panel")

train_pct = st.sidebar.slider("Training Data Percentage (%)", min_value=10, max_value=90, value=80, step=5)
test_ratio = (100.0 - train_pct) / 100.0

st.sidebar.markdown(f"**Split Ratio:** {train_pct}% Train / {100-train_pct}% Test")

app_mode = st.sidebar.selectbox(
    "Choose Dashboard View:",
    ["Interactive Predictor & Diagnostics", "Model Evaluations & Metrics"]
)

# =========================================================================
# MEMORY-OPTIMIZED DATA LOADING (Prevents OOM Crashes)
# =========================================================================
@st.cache_data
def load_data():
    cols = [
        'ArrDelay', 'CRSDepTime', 'TailNum', 'Distance', 
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'TaxiOut', 'AirTime'
    ]
    # Read only required columns
    df = pd.read_csv('DelayedFlights.csv', usecols=cols)
    
    # Cap dataset size for Streamlit Cloud 1GB RAM limit
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42).reset_index(drop=True)
        
    # Downcast float64 to float32 to save RAM
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading 'DelayedFlights.csv'. Make sure it is in the same repository folder. Details: {e}")
    st.stop()

# =========================================================================
# CACHED MODEL TRAINING ROUTINE
# =========================================================================
@st.cache_resource
def train_models(test_r):
    # --- CODE 1 PREP ---
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

    # --- CODE 2 PREP ---
    df_c2 = df.copy()
    delay_cols = ['WeatherDelay', 'NASDelay', 'CarrierDelay']
    df_c2[delay_cols] = df_c2[delay_cols].fillna(0)

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

    return (
        (X_test_c1, y_test_class_c1, y_test_reg_c1, X_test_scaled_c1, scaler_c1, features_c1, 
         lin_reg_c1, log_reg_c1, dt_cls_c1, rf_cls_c1, kmeans_c1, nn_cls_c1),
        (X_test_c2, y_test_class_c2, y_test_reg_c2, X_test_scaled_c2, scaler_c2, features_c2, 
         lin_reg_c2, log_reg_c2, dt_cls_c2, rf_cls_c2, kmeans_c2, nn_cls_c2)
    )

with st.spinner("Initializing models... Please wait a moment."):
    c1_bundle, c2_bundle = train_models(test_ratio)

(X_test_c1, y_test_class_c1, y_test_reg_c1, X_test_scaled_c1, scaler_c1, features_c1, 
 lin_reg_c1, log_reg_c1, dt_cls_c1, rf_cls_c1, kmeans_c1, nn_cls_c1) = c1_bundle

(X_test_c2, y_test_class_c2, y_test_reg_c2, X_test_scaled_c2, scaler_c2, features_c2, 
 lin_reg_c2, log_reg_c2, dt_cls_c2, rf_cls_c2, kmeans_c2, nn_cls_c2) = c2_bundle

# =========================================================================
# VIEW 1: INTERACTIVE PREDICTOR & DIAGNOSTICS
# =========================================================================
if app_mode == "Interactive Predictor & Diagnostics":
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
        
        st.markdown("---")
        st.subheader(">>> Prediction Results")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Linear Regression (Delay)", f"{lin_reg_c1.predict(input_scaled)[0]:.1f} mins")
        res_col1.metric("Logistic Regression", "DELAYED ⚠️" if log_reg_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        
        res_col2.metric("Decision Tree", "DELAYED ⚠️" if dt_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        res_col2.metric("Random Forest (BEST)", "DELAYED ⚠️" if rf_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        
        res_col3.metric("Neural Network (MLP)", "DELAYED ⚠️" if nn_cls_c1.predict(input_scaled)[0] == 1 else "ON-TIME 🟢")
        res_col3.metric("K-Means Risk Cluster", f"Group #{kmeans_c1.predict(input_scaled)[0]}")
        
        st.markdown("---")
        st.subheader("📊 Model Visualizations & Insights")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            y_reg_pred_test = lin_reg_c1.predict(X_test_scaled_c1)
            ax.scatter(y_test_reg_c1, y_reg_pred_test, alpha=0.3, color='blue')
            ax.plot([y_test_reg_c1.min(), y_test_reg_c1.max()], [y_test_reg_c1.min(), y_test_reg_c1.max()], 'r--', lw=2)
            ax.set_title('Multi-Linear Reg: Actual vs Predicted Delay')
            ax.set_xlabel('Actual Delay')
            ax.set_ylabel('Predicted Delay')
            st.pyplot(fig)
            plt.close(fig)
            
        with v_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=list(log_reg_c1.coef_[0]), y=features_c1, ax=ax, palette='viridis', hue=features_c1, legend=False)
            ax.set_title('Logistic Regression: Feature Weights')
            st.pyplot(fig)
            plt.close(fig)

        v_col3, v_col4 = st.columns(2)
        with v_col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            pd.Series(rf_cls_c1.feature_importances_, index=features_c1).plot(kind='barh', ax=ax, color='teal')
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)
            plt.close(fig)
            
        with v_col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=X_test_c1['Distance'], y=X_test_c1['Inbound_Leg_ArrDelay'], hue=kmeans_c1.predict(X_test_scaled_c1), palette='Set1', alpha=0.6, ax=ax)
            ax.set_title('K-Means Clusters')
            ax.set_xlabel('Distance')
            ax.set_ylabel('Inbound Leg ArrDelay')
            st.pyplot(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(nn_cls_c1.loss_curve_, color='purple', lw=2)
        ax.set_title('Neural Network Loss Convergence')
        ax.set_xlabel('Iterations / Epochs')
        ax.set_ylabel('Loss')
        st.pyplot(fig)
        plt.close(fig)
            
        st.write("#### Decision Tree Diagram")
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_tree(dt_cls_c1, feature_names=features_c1, class_names=['On-Time', 'Delayed'], filled=True, ax=ax, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

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
        
        st.markdown("---")
        st.subheader("📊 Diagnostic Visualizations & Insights")
        
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            y_reg_pred_test_c2 = lin_reg_c2.predict(X_test_scaled_c2)
            ax.scatter(y_test_reg_c2, y_reg_pred_test_c2, alpha=0.3, color='crimson')
            ax.plot([y_test_reg_c2.min(), y_test_reg_c2.max()], [y_test_reg_c2.min(), y_test_reg_c2.max()], 'k--', lw=2)
            ax.set_title('Multi-Linear Reg: Estimated Carrier Delay')
            ax.set_xlabel('Actual Carrier Delay')
            ax.set_ylabel('Predicted Carrier Delay')
            st.pyplot(fig)
            plt.close(fig)
            
        with v_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            coef_df = pd.DataFrame(log_reg_c2.coef_, index=log_reg_c2.classes_, columns=features_c2)
            sns.heatmap(coef_df, annot=True, fmt=".2f", cmap='coolwarm', center=0, ax=ax)
            ax.set_title("Logistic Reg: Feature Weights Heatmap")
            st.pyplot(fig)
            plt.close(fig)

        v_col3, v_col4 = st.columns(2)
        with v_col3:
            fig, ax = plt.subplots(figsize=(6, 4))
            pd.Series(rf_cls_c2.feature_importances_, index=features_c2).plot(kind='bar', color='darkgreen', ax=ax, rot=0)
            ax.set_title("Random Forest Feature Importance")
            st.pyplot(fig)
            plt.close(fig)
            
        with v_col4:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=X_test_c2['TaxiOut'], y=X_test_c2['AirTime'], hue=kmeans_c2.predict(X_test_scaled_c2), palette='Set2', alpha=0.6, ax=ax)
            ax.set_title('K-Means Clusters')
            ax.set_xlabel('TaxiOut')
            ax.set_ylabel('AirTime')
            st.pyplot(fig)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(nn_cls_c2.loss_curve_, color='orange', lw=2)
        ax.set_title('Neural Network Loss Convergence')
        ax.set_xlabel('Iterations / Epochs')
        ax.set_ylabel('Loss')
        st.pyplot(fig)
        plt.close(fig)
            
        st.write("#### Decision Tree Breakdown")
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_tree(dt_cls_c2, feature_names=features_c2, class_names=list(dt_cls_c2.classes_), filled=True, ax=ax, fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

# =========================================================================
# VIEW 2: MODEL EVALUATIONS & METRICS
# =========================================================================
else:
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
            "Neural Network (MLP)": nn_cls_c1
        }
        
        for name, clf in classifiers_c1.items():
            st.write(f"#### {name}")
            y_pred = clf.predict(X_test_scaled_c1)
            acc = accuracy_score(y_test_class_c1, y_pred)
            roc_auc = roc_auc_score(y_test_class_c1, clf.predict_proba(X_test_scaled_c1)[:, 1]) if hasattr(clf, "predict_proba") else float('nan')
            
            sub_c1, sub_c2 = st.columns(2)
            sub_c1.metric("Accuracy", f"{acc:.4f}")
            sub_c2.metric("ROC-AUC Score", f"{roc_auc:.4f}")
            
            report = classification_report(y_test_class_c1, y_pred, target_names=['On-Time', 'Delayed'], output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
            st.markdown("---")

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
            "Neural Network (MLP)": nn_cls_c2
        }
        
        for name, clf in classifiers_c2.items():
            st.write(f"#### {name}")
            y_pred = clf.predict(X_test_scaled_c2)
            acc = accuracy_score(y_test_class_c2, y_pred)
            
            st.metric("Accuracy", f"{acc:.4f}")
            report = classification_report(y_test_class_c2, y_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose())
            st.markdown("---")