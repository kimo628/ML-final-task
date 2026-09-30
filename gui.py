import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import joblib

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Bank Churn Intelligence", page_icon="🏦", layout="wide")
sns.set_theme(style="whitegrid", palette="muted")

PALETTE = ["#EF4444", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6", "#64748B", "#EC4899", "#14B8A6"]


def style_fig(fig, ax):
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")


# ============================================================
# LOAD RAW DATA
# ============================================================


@st.cache_data
def load_data():
    return pd.read_csv("train.csv")


@st.cache_resource
def load_artifacts():
    return {
        "model": joblib.load("churn_model.pkl"),
        "needs_scaling": joblib.load("model_needs_scaling.pkl"),
        "scaler": joblib.load("scaler.pkl"),
        "columns": joblib.load("model_columns.pkl"),
        "results": joblib.load("results.pkl"),
        "confusion_matrix": joblib.load("confusion_matrix.pkl"),
        "feature_importance": joblib.load("feature_importance.pkl"),
        "classification_reports": joblib.load("classification_reports.pkl"),
        "all_probas": joblib.load("all_probas.pkl"),
        "y_test": joblib.load("y_test.pkl"),
    }


try:
    df = load_data()
except FileNotFoundError:
    st.error("train.csv was not found. Put train.csv in the same folder as app.py.")
    st.stop()

try:
    art = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "Missing model file: "
        f"{e}. Run the notebook's 'Export Artifacts for the GUI' cell first "
        "(Run All), then copy every generated .pkl file into this same "
        "folder next to app.py."
    )
    st.stop()

model = art["model"]
model_columns = art["columns"]
results_df = art["results"]
GUI_MODEL_NAME = "Gradient Boosting"
best_row = results_df[results_df["Model"] == GUI_MODEL_NAME].iloc[0]

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🏦 Bank Churn Intelligence")
st.sidebar.markdown("**Churn Prediction Dashboard**\n\nMachine Learning Capstone Project")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🎯 Churn Prediction", "📊 Customer Analytics", "📈 Model Performance"],
)

st.sidebar.divider()
st.sidebar.caption(
    f"Deployed model: **{GUI_MODEL_NAME}** "
    f"(Accuracy = {best_row['Accuracy'] * 100:.2f}%, F1 = {best_row['F1']:.3f})"
)
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
        f"This dashboard predicts churn using a tuned **{GUI_MODEL_NAME}** model "
        f"(Accuracy = {best_row['Accuracy'] * 100:.2f}%, ROC-AUC = {best_row['ROC-AUC']:.3f})."
    )


# ============================================================
# 2) CHURN PREDICTION
# ============================================================

elif page == "🎯 Churn Prediction":

    st.title("🎯 Customer Churn Prediction")
    st.write(
        f"Enter a customer's information. The prediction comes from the "
        f"trained **{GUI_MODEL_NAME}** model (Accuracy = "
        f"{best_row['Accuracy'] * 100:.2f}% on the test set)."
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

        # Same preprocessing as the notebook: dummy-encode Geography,
        # map Gender to 0/1, then align to the exact training column order.
        customer = pd.get_dummies(customer, columns=["Geography"], drop_first=True, dtype=int)
        customer["Gender"] = (customer["Gender"] == "Male").astype(int)
        customer = customer.reindex(columns=model_columns, fill_value=0)

        if art["needs_scaling"]:
            customer_input = art["scaler"].transform(customer)
        else:
            customer_input = customer  # Gradient Boosting was trained on raw features

        prediction = model.predict(customer_input)[0]
        probability = model.predict_proba(customer_input)[0][1]

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
        sns.barplot(data=top_features, x="Importance", y="Feature", ax=ax, palette=PALETTE)
        ax.set_title("Top 5 Most Influential Features (model-wide)", color="#1E293B", fontweight="bold")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "These are the model's overall most influential features (global "
            "feature importance), not a per-customer explanation."
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
        geo_filter = st.selectbox("Geography", ["All"] + sorted(df["Geography"].unique().tolist()))
    with f2:
        gender_filter = st.selectbox("Gender", ["All"] + sorted(df["Gender"].unique().tolist()))
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

    r1c1, r1c2 = st.columns(2)
    with r1c1:
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

    with r1c2:
        st.subheader("Churn by Geography")
        churn_geo = filtered.groupby("Geography")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_geo.plot(kind="bar", ax=ax, color=PALETTE[1])
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("Churn by Gender")
        churn_gender = filtered.groupby("Gender")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_gender.plot(kind="bar", ax=ax, color=PALETTE[2])
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with r2c2:
        st.subheader("Churn by Active Membership")
        churn_active = filtered.groupby("IsActiveMember")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_active.plot(kind="bar", ax=ax, color=PALETTE[3])
        ax.set_xticklabels(["Inactive", "Active"], rotation=0)
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
        plt.close(fig)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.subheader("Churn by Number of Products")
        churn_products = filtered.groupby("NumOfProducts")["Exited"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(6, 4))
        style_fig(fig, ax)
        churn_products.plot(kind="bar", ax=ax, color=PALETTE[4])
        ax.set_ylabel("Churn Rate (%)")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        plt.close(fig)

    with r3c2:
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
        f"✅ Deployed model: **{GUI_MODEL_NAME}** — Accuracy = "
        f"{best_row['Accuracy'] * 100:.2f}%, ROC-AUC = {best_row['ROC-AUC']:.3f}"
    )
    st.caption(
        "Random Forest has a slightly higher F1-Score (better at catching "
        "churners), while Gradient Boosting has the highest Accuracy and "
        "ROC-AUC. Both are shown below so the trade-off is transparent."
    )

    st.subheader("Model Comparison (all 8 models)")
    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader(f"Confusion Matrix — {GUI_MODEL_NAME}")
    cm = art["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6, 4))
    style_fig(fig, ax)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Stayed", "Predicted Churned"],
        yticklabels=["Actual Stayed", "Actual Churned"], ax=ax,
    )
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("ROC Curves — All Models")
    from sklearn.metrics import roc_curve

    y_test = art["y_test"]
    fig, ax = plt.subplots(figsize=(9, 6))
    style_fig(fig, ax)
    for i, (name, proba) in enumerate(art["all_probas"].items()):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = results_df.loc[results_df["Model"] == name, "ROC-AUC"].values[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=PALETTE[i % len(PALETTE)])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8)
    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("Classification Report")
    report_model = st.selectbox("Choose a model", list(art["classification_reports"].keys()))
    st.code(art["classification_reports"][report_model], language="text")