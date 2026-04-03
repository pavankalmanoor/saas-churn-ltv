from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RISK_COLORS = {"Low Risk": "#1e8449", "Medium Risk": "#d68910", "High Risk": "#c0392b"}

st.set_page_config(page_title="SaaS Customer Intelligence Platform", layout="wide")


@st.cache_data
def load_users() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH / "saas_users.csv")


@st.cache_data
def load_segment_stats() -> pd.DataFrame:
    path = DATA_PATH / "segment_stats.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_cohort_retention() -> pd.DataFrame:
    path = DATA_PATH / "cohort_retention.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)



def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = df.copy()
    metrics["recency_score"] = 1 - (metrics["days_since_last_login"] / metrics["days_since_last_login"].max())
    metrics["frequency_score"] = metrics["daily_active_days"] / 30
    metrics["monetary_score"] = metrics["total_revenue"] / metrics["total_revenue"].max()
    metrics["rfm_score"] = (
        metrics["recency_score"] * 0.35
        + metrics["frequency_score"] * 0.40
        + metrics["monetary_score"] * 0.25
    )
    metrics["churn_risk"] = (
        0.30 * (1 - metrics["daily_active_days"] / 30)
        + 0.20 * (metrics["days_since_last_login"] / 60)
        + 0.20 * (metrics["payment_failures"] / 3)
        + 0.15 * (1 - metrics["feature_adoption_score"] / 100)
        + 0.15 * (1 - metrics["goals_set"])
    ).clip(0, 1)
    metrics["risk_tier"] = pd.cut(
        metrics["churn_risk"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
        include_lowest=True,
    )
    return metrics



def render_kpis(df: pd.DataFrame) -> None:
    columns = st.columns(5)
    total_users = len(df)
    churned = int(df["churned"].sum())
    churn_rate = df["churned"].mean()
    total_mrr = df["monthly_charge"].sum()
    high_risk_users = int((df["risk_tier"] == "High Risk").sum())

    columns[0].metric("Total Users", f"{total_users:,}")
    columns[1].metric("Churned Users", f"{churned:,}")
    columns[2].metric("Churn Rate", f"{churn_rate:.1%}")
    columns[3].metric("Monthly Recurring Revenue", f"${total_mrr:,.0f}")
    columns[4].metric("High Risk Users", f"{high_risk_users:,}")



def render_churn_analysis(df: pd.DataFrame) -> None:
    st.subheader("Churn analysis")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_plan = df.groupby("plan_tier")["churned"].mean()
        churn_by_plan.plot(kind="bar", ax=ax, color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="black")
        ax.set_title("Churn Rate by Plan Tier")
        ax.set_ylabel("Churn Rate")
        ax.tick_params(axis="x", rotation=0)
        for i, value in enumerate(churn_by_plan):
            ax.text(i, value + 0.005, f"{value:.1%}", ha="center", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        risk_counts = df["risk_tier"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"])
        risk_counts.plot(kind="bar", ax=ax, color=[RISK_COLORS[label] for label in risk_counts.index], edgecolor="black")
        ax.set_title("Users by Risk Tier")
        ax.set_ylabel("Number of Users")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Churn drivers")
    col3, col4 = st.columns(2)

    with col3:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df[df["churned"] == 0]["daily_active_days"], bins=25, alpha=0.7, color="#2ecc71", label="Retained")
        ax.hist(df[df["churned"] == 1]["daily_active_days"], bins=25, alpha=0.7, color="#e74c3c", label="Churned")
        ax.set_title("Daily Active Days Distribution")
        ax.set_xlabel("Active Days in Last 30 Days")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        fig, ax = plt.subplots(figsize=(6, 4))
        payment_churn = df.groupby("payment_failures")["churned"].mean()
        payment_churn.plot(kind="bar", ax=ax, color="#e74c3c", edgecolor="black")
        ax.set_title("Churn Rate by Payment Failures")
        ax.set_xlabel("Payment Failures")
        ax.set_ylabel("Churn Rate")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



def render_segments(segment_stats: pd.DataFrame) -> None:
    st.subheader("Customer segmentation and LTV")
    if segment_stats.empty:
        st.warning("Segment statistics file is not available.")
        return

    col1, col2, col3 = st.columns(3)
    chart_colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        segment_stats.set_index("segment_label")["expected_ltv"].plot(kind="bar", ax=ax, color=chart_colors, edgecolor="black")
        ax.set_title("Expected LTV by Segment")
        ax.set_ylabel("LTV ($)")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        segment_stats.set_index("segment_label")["churn_rate"].plot(kind="bar", ax=ax, color=chart_colors, edgecolor="black")
        ax.set_title("Churn Rate by Segment")
        ax.set_ylabel("Churn Rate")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        segment_stats.set_index("segment_label")["users"].plot(kind="bar", ax=ax, color=chart_colors, edgecolor="black")
        ax.set_title("Users per Segment")
        ax.set_ylabel("Users")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Segment summary")
    st.dataframe(
        segment_stats[["segment_label", "users", "churn_rate", "avg_monthly_rev", "expected_ltv", "total_arr"]].style.format(
            {
                "churn_rate": "{:.1%}",
                "avg_monthly_rev": "${:.2f}",
                "expected_ltv": "${:.0f}",
                "total_arr": "${:,.0f}",
            }
        )
    )



def render_cohorts(cohort_data: pd.DataFrame) -> None:
    st.subheader("Cohort retention")
    if cohort_data.empty:
        st.warning("Cohort retention file is not available.")
        return

    retention_pivot = cohort_data.pivot_table(index="cohort", columns="plan_tier", values="retention_rate")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(retention_pivot, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax, vmin=0.4, vmax=1.0, linewidths=0.5)
    ax.set_title("Retention Rate by Cohort and Plan Tier")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        """
Key observations:
- Premium plans retain the largest share of users across cohorts.
- Monthly plans show the highest attrition and the clearest upgrade opportunity.
- Annual plans improve with tenure, which supports plan-conversion strategies.
"""
    )



def render_user_explorer() -> None:
    st.subheader("Individual user risk explorer")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("Adjust the user profile:")
        active_days = st.slider("Daily Active Days", 0, 30, 10)
        adoption = st.slider("Feature Adoption Score", 0, 100, 40)
        last_login = st.slider("Days Since Last Login", 0, 60, 10)
        payment_failures = st.selectbox("Payment Failures", [0, 1, 2, 3])
        plan = st.selectbox("Plan Tier", ["monthly", "annual", "premium"])
        goals_set = st.checkbox("Goals Set", value=True)

        risk = (
            0.30 * (1 - active_days / 30)
            + 0.20 * (last_login / 60)
            + 0.20 * (payment_failures / 3)
            + 0.15 * (1 - adoption / 100)
            + 0.15 * (1 - int(goals_set))
        )
        risk *= {"monthly": 1.3, "annual": 0.85, "premium": 0.6}[plan]
        risk = min(risk, 1.0)

    with col2:
        if risk > 0.6:
            tier = "High Risk"
        elif risk > 0.3:
            tier = "Medium Risk"
        else:
            tier = "Low Risk"

        st.metric("Churn Risk Score", f"{risk:.1%}")
        st.markdown(
            f"<p style='color:{RISK_COLORS[tier]}; font-size:1.25rem; font-weight:600;'>{tier}</p>",
            unsafe_allow_html=True,
        )

        factors = {
            "Inactivity": 0.30 * (1 - active_days / 30),
            "Login Recency": 0.20 * (last_login / 60),
            "Payment Issues": 0.20 * (payment_failures / 3),
            "Low Adoption": 0.15 * (1 - adoption / 100),
            "No Goals Set": 0.15 * (1 - int(goals_set)),
        }

        fig, ax = plt.subplots(figsize=(6, 3))
        colors = [RISK_COLORS["High Risk"] if value > 0.1 else RISK_COLORS["Medium Risk"] if value > 0.05 else RISK_COLORS["Low Risk"] for value in factors.values()]
        ax.barh(list(factors.keys()), list(factors.values()), color=colors, edgecolor="black")
        ax.set_xlabel("Risk Contribution")
        ax.set_title("Risk Score Components")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)



def main() -> None:
    st.title("SaaS Customer Intelligence Platform")
    st.markdown("Churn prediction, LTV analysis, behavioral segmentation, and retention reporting.")
    st.divider()

    df = build_metrics(load_users())
    render_kpis(df)
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Churn Analysis", "LTV and Segments", "Cohort Retention", "User Explorer"])
    with tab1:
        render_churn_analysis(df)
    with tab2:
        render_segments(load_segment_stats())
    with tab3:
        render_cohorts(load_cohort_retention())
    with tab4:
        render_user_explorer()

    st.divider()
    st.caption("Synthetic SaaS dataset with 10,000 users.")


if __name__ == "__main__":
    main()
