import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Churn Intelligence", page_icon="🏦", layout="wide"
)

sns.set_theme(style="whitegrid", palette="muted")

PALETTE = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#64748B", "#EC4899"]


def style_fig(fig, ax):
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")


# ============================================================
# LOAD RAW DATA
# ============================================================


@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df_clean = df.copy()

    outlier_cols = ["CreditScore", "Age", "Tenure", "NumOfProducts"]
    for col in outlier_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        df_clean[col] = df_clean[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return df, df_clean


# ============================================================
# LOAD THE REAL ARTIFACTS EXPORTED FROM THE NOTEBOOK
# (No retraining here -> numbers on every page match the notebook exactly)
# ============================================================


@st.cache_resource
def load_artifacts():
    return {
        "rf_model": joblib.load("rf_churn_model.pkl"),
        "rf_scaler": joblib.load("rf_scaler.pkl"),
        "rf_columns": joblib.load("rf_columns.pkl"),
        "classification_results": joblib.load("classification_results.pkl"),
        "confusion_matrix": joblib.load("rf_confusion_matrix.pkl"),
        "roc_auc_results": joblib.load("roc_auc_results.pkl"),
        "feature_importance": joblib.load("rf_feature_importance.pkl"),
        "classification_reports": joblib.load("classification_reports.pkl"),
        "regression_results": joblib.load("regression_results.pkl"),
        "kmeans_model": joblib.load("kmeans_model.pkl"),
        "kmeans_scaler": joblib.load("kmeans_scaler.pkl"),
        "segmentation_features": joblib.load("segmentation_features.pkl"),
        "cluster_profile": joblib.load("cluster_profile.pkl"),
        "cluster_comparison": joblib.load("kmeans_dbscan_comparison.pkl"),
        "silhouette_results": joblib.load("silhouette_results.pkl"),
        "segmentation_labels": joblib.load("segmentation_labels.pkl"),
    }


try:
    df, df_clean = load_data()
except FileNotFoundError:
    st.error("train.csv was not found. Put train.csv in the same folder as app.py.")
    st.stop()

try:
    art = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model file: "
        f"{e}. Run the notebook's 'Export Artifacts for the GUI' cell "
        "first (Run All), then copy every generated .pkl file into this "
        "same folder next to app.py."
    )
    st.stop()

rf_model = art["rf_model"]
rf_scaler = art["rf_scaler"]
rf_columns = art["rf_columns"]
final_results = art["classification_results"]
best_row = final_results.sort_values("F1-Score", ascending=False).iloc[0]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Bank Churn Intelligence")
st.sidebar.markdown("**Churn • Value • Segmentation**\n\nMachine Learning Capstone Project")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎯 Churn Prediction", "📊 Customer Analytics", "📈 Model Performance"],
)

st.sidebar.divider()
st.sidebar.caption(f"Best model so far: **{best_row['Model']}** (F1 = {best_row['F1-Score']:.3f})")
st.sidebar.caption("Bank Customer Churn — 165,034 records")


# ============================================================
# 1) OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.title("🏦 Bank Churn Intelligence Dashboard")
    st.caption("Understand the customer base in a few seconds before diving into predictions.")

    st.divider()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Churn Rate", f"{df['Exited'].mean() * 100:.1f}%")
    c3.metric("Active Members", f"{df['IsActiveMember'].mean() * 100:.1f}%")
    c4.metric("Avg. Balance", f"{df['Balance'].mean():,.0f}")
    c5.metric("Avg. Age", f"{df['Age'].mean():.1f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customers by Geography")
        geo_counts = df["Geography"].value_counts().reset_index()
        geo_counts.columns = ["Geography", "Customers"]
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        sns.barplot(data=geo_counts, x="Geography", y="Customers", ax=ax, palette=PALETTE)
        ax.set_title("Customers by Geography", color="#1E293B", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Churned vs Retained Customers")
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        sns.countplot(data=df, x="Exited", ax=ax, palette=["#3B82F6", "#EF4444"])
        ax.set_xticklabels(["Retained (0)", "Exited (1)"])
        ax.set_xlabel("")
        ax.set_ylabel("Customers")
        ax.set_title("Exited = 0 vs 1", color="#1E293B", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "This dashboard analyzes Bank Customer Churn data across three ML tasks: "
        "predicting Balance (regression), predicting Exited/churn (classification), "
        "and grouping customers into behavioral segments (KMeans/DBSCAN)."
    )


# ============================================================
# 2) CHURN PREDICTION
# ============================================================

elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")
    st.write(
        "Enter a customer's information. The prediction comes from the exact "
        f"tuned **{best_row['Model']}** model trained in the notebook "
        "(same model shown in Model Performance)."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=350, max_value=850, value=650)
        age = st.number_input("Age", min_value=18, max_value=92, value=35)
        tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)

    with col2:
        geography = st.selectbox("Geography", sorted(df["Geography"].unique()))
        gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
        balance = st.number_input("Balance", min_value=0.0, value=float(df["Balance"].median()))

    with col3:
        products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        active = st.selectbox("Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        salary = st.number_input("Estimated Salary", min_value=0.0, value=float(df["EstimatedSalary"].median()))

    st.divider()

    if st.button("Predict Churn", type="primary", use_container_width=True):

        customer = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary,
        }])

        customer = pd.get_dummies(customer, columns=["Geography", "Gender"], drop_first=True)
        customer = customer.reindex(columns=rf_columns, fill_value=0)
        customer_scaled = rf_scaler.transform(customer)

        prediction = rf_model.predict(customer_scaled)[0]
        probability = rf_model.predict_proba(customer_scaled)[0][1]

        st.divider()
        st.subheader("Customer Churn Prediction")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            if probability >= 0.5:
                st.error("⚠️ HIGH RISK OF CHURN")
            else:
                st.success("✅ LOW RISK OF CHURN")

        with result_col2:
            st.metric("Probability of Churn", f"{probability * 100:.0f}%")

        st.divider()
        st.subheader("Main Factors Behind This Model's Predictions")

        top_features = art["feature_importance"].head(5)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        style_fig(fig, ax)
        sns.barplot(
            data=top_features, x="Importance", y="Feature", ax=ax, palette=PALETTE
        )
        ax.set_title("Top 5 Most Influential Features (model-wide)", color="#1E293B", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "These are the model's overall most influential features (global "
            "feature importance), not a per-customer explanation — they show "
            "what generally drives churn predictions across all customers."
        )

        st.info(
            "This is the model's confidence for THIS ONE customer, not the "
            "model's overall accuracy. A model with high overall accuracy can "
            "still give any single customer a probability anywhere from 0% to "
            "100% depending on their own attributes."
        )


# ============================================================
# 3) CUSTOMER ANALYTICS
# ============================================================

elif page == "📊 Customer Analytics":

    st.title("📊 Customer Analytics")
    st.caption("Filter the customer base and explore churn patterns interactively.")

    st.divider()

    f1, f2, f3 = st.columns(3)

    with f1:
        geo_options = ["All"] + sorted(df["Geography"].unique().tolist())
        geo_filter = st.selectbox("Geography", geo_options)

    with f2:
        gender_options = ["All"] + sorted(df["Gender"].unique().tolist())
        gender_filter = st.selectbox("Gender", gender_options)

    with f3:
        age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
        age_range = st.slider("Age", age_min, age_max, (age_min, age_max))

    filtered = df.copy()
    if geo_filter != "All":
        filtered = filtered[filtered["Geography"] == geo_filter]
    if gender_filter != "All":
        filtered = filtered[filtered["Gender"] == gender_filter]
    filtered = filtered[filtered["Age"].between(age_range[0], age_range[1])]

    st.caption(f"Showing **{len(filtered):,}** customers matching the current filters.")

    if len(filtered) == 0:
        st.warning("No customers match these filters.")
        st.stop()

    st.divider()

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("Churn by Age Group")
        age_bins = pd.cut(filtered["Age"], bins=[18, 30, 40, 50, 60, 70, 100])
        churn_by_age = filtered.groupby(age_bins, observed=True)["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_by_age.plot(kind="bar", ax=ax, color=PALETTE[0])
        ax.set_ylabel("Churn Rate (%)")
        ax.set_xlabel("Age Group")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        plt.close(fig)

    with row1_col2:
        st.subheader("Churn by Geography")
        churn_geo = filtered.groupby("Geography")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_geo.plot(kind="bar", ax=ax, color=PALETTE[1])
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("Churn by Gender")
        churn_gender = filtered.groupby("Gender")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_gender.plot(kind="bar", ax=ax, color=PALETTE[2])
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with row2_col2:
        st.subheader("Churn by Active Membership")
        churn_active = filtered.groupby("IsActiveMember")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_active.plot(kind="bar", ax=ax, color=PALETTE[3])
        ax.set_xticklabels(["Inactive", "Active"], rotation=0)
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
        plt.close(fig)

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        st.subheader("Churn by Number of Products")
        churn_products = filtered.groupby("NumOfProducts")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_products.plot(kind="bar", ax=ax, color=PALETTE[4])
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with row3_col2:
        st.subheader("Balance Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        sns.histplot(data=filtered, x="Balance", bins=30, kde=True, ax=ax, color=PALETTE[5])
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Credit Score Distribution")
    fig, ax = plt.subplots(figsize=(12, 4))
    style_fig(fig, ax)
    sns.histplot(data=filtered, x="CreditScore", bins=30, kde=True, ax=ax, color=PALETTE[6])
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# 4) MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.success(
        f"🏆 Best model so far: **{best_row['Model']}** — highest F1-Score "
        f"({best_row['F1-Score']:.3f}) in the tuned comparison below."
    )

    st.subheader("Classification — Model Comparison")

    display_df = final_results.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    display_df = display_df.rename(columns={"Training_Time_Sec": "Training Time (sec)"})
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Training Time by Model")
    st.caption(
        "165,034 rows is enough to feel the real effect of data volume on "
        "training time — notice how much longer SVM takes than the rest."
    )
    time_df = final_results[["Model", "Training_Time_Sec"]].sort_values("Training_Time_Sec")
    fig, ax = plt.subplots(figsize=(10, 4))
    style_fig(fig, ax)
    sns.barplot(data=time_df, x="Training_Time_Sec", y="Model", ax=ax, palette=PALETTE)
    ax.set_xlabel("Training Time (seconds)")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader(f"Confusion Matrix — {best_row['Model']}")
    cm = art["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 4))
    style_fig(fig, ax)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Stay", "Predicted Churn"],
        yticklabels=["Actual Stay", "Actual Churn"], ax=ax,
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("ROC-AUC Comparison")
    st.dataframe(art["roc_auc_results"], use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Classification Report")
    report_model = st.selectbox("Choose a model", list(art["classification_reports"].keys()))
    st.code(art["classification_reports"][report_model], language="text")

    st.divider()

    with st.expander("Regression Results (Predicting Balance)"):
        st.dataframe(art["regression_results"], use_container_width=True, hide_index=True)
        st.caption(
            "The project's main target is Exited (classification), shown above. "
            "Balance regression is included here for completeness per the project brief."
        )
