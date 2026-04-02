import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="SaaS Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path('/Users/pavan/saas-churn-ltv')

# ── Load Data ──────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(PROJECT_ROOT / 'data' / 'saas_users.csv')

df = load_data()

# Recompute segments
df['recency_score']   = 1 - (df['days_since_last_login'] / df['days_since_last_login'].max())
df['frequency_score'] = df['daily_active_days'] / 30
df['monetary_score']  = df['total_revenue'] / df['total_revenue'].max()
df['rfm_score']       = (df['recency_score'] * 0.35 + 
                          df['frequency_score'] * 0.40 + 
                          df['monetary_score'] * 0.25)

# Simple risk score
df['churn_risk'] = (
    0.3 * (1 - df['daily_active_days']/30) +
    0.2 * (df['days_since_last_login']/60) +
    0.2 * (df['payment_failures']/3) +
    0.15 * (1 - df['feature_adoption_score']/100) +
    0.15 * (1 - df['goals_set'])
).clip(0,1)

df['risk_tier'] = pd.cut(df['churn_risk'], 
                          bins=[0, 0.3, 0.6, 1.0],
                          labels=['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'])

# ── Header ─────────────────────────────────────────────
st.title("📊 SaaS Customer Intelligence Platform")
st.markdown("**Churn Prediction · LTV Analysis · Behavioral Segmentation · Product Analytics**")
st.divider()

# ── KPI Row ────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_users    = len(df)
churned        = df['churned'].sum()
churn_rate     = df['churned'].mean()
avg_ltv        = (df['monthly_charge'] / df['churned'].replace(0, 0.01)).mean()
total_mrr      = df['monthly_charge'].sum()

col1.metric("Total Users", f"{total_users:,}")
col2.metric("Churned Users", f"{churned:,}")
col3.metric("Churn Rate", f"{churn_rate:.1%}")
col4.metric("Avg MRR", f"${total_mrr:,.0f}")
col5.metric("High Risk Users", f"{(df['risk_tier']=='🔴 High Risk').sum():,}")

st.divider()

# ── Tabs ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Churn Analysis", 
    "💰 LTV & Segments",
    "📈 Cohort Retention",
    "🔍 User Explorer"
])

with tab1:
    st.subheader("Churn Risk Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by plan
        fig, ax = plt.subplots(figsize=(6,4))
        churn_plan = df.groupby('plan_tier')['churned'].mean()
        colors = ['#2ecc71','#f39c12','#e74c3c']
        churn_plan.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        ax.set_title('Churn Rate by Plan Tier')
        ax.set_ylabel('Churn Rate')
        ax.tick_params(axis='x', rotation=0)
        for i, v in enumerate(churn_plan):
            ax.text(i, v+0.005, f'{v:.1%}', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # Risk tier distribution
        fig, ax = plt.subplots(figsize=(6,4))
        risk_counts = df['risk_tier'].value_counts()
        colors_risk = ['#2ecc71','#f39c12','#e74c3c']
        risk_counts.plot(kind='bar', ax=ax, color=colors_risk, edgecolor='black')
        ax.set_title('Users by Risk Tier')
        ax.set_ylabel('Number of Users')
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout()
        st.pyplot(fig)

    # Churn drivers
    st.subheader("Key Churn Drivers")
    col3, col4 = st.columns(2)
    
    with col3:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df[df['churned']==0]['daily_active_days'],
                bins=25, alpha=0.7, color='#2ecc71', label='Retained')
        ax.hist(df[df['churned']==1]['daily_active_days'],
                bins=25, alpha=0.7, color='#e74c3c', label='Churned')
        ax.set_title('Daily Active Days Distribution')
        ax.set_xlabel('Active Days (last 30)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    with col4:
        fig, ax = plt.subplots(figsize=(6,4))
        payment_churn = df.groupby('payment_failures')['churned'].mean()
        payment_churn.plot(kind='bar', ax=ax, color='#e74c3c', edgecolor='black')
        ax.set_title('Churn Rate by Payment Failures')
        ax.set_xlabel('Payment Failures')
        ax.set_ylabel('Churn Rate')
        ax.tick_params(axis='x', rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.subheader("Customer Segmentation & LTV")
    
    # Load segment stats
    try:
        seg_stats = pd.read_csv(PROJECT_ROOT / 'data' / 'segment_stats.csv', index_col=0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            seg_stats['expected_ltv'].plot(kind='bar', ax=ax,
                color=['#2ecc71','#3498db','#f39c12','#e74c3c'], edgecolor='black')
            ax.set_title('Expected LTV by Segment')
            ax.set_ylabel('LTV ($)')
            ax.tick_params(axis='x', rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            seg_stats['churn_rate'].plot(kind='bar', ax=ax,
                color=['#2ecc71','#3498db','#f39c12','#e74c3c'], edgecolor='black')
            ax.set_title('Churn Rate by Segment')
            ax.set_ylabel('Churn Rate')
            ax.tick_params(axis='x', rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col3:
            fig, ax = plt.subplots(figsize=(5,4))
            seg_stats['users'].plot(kind='bar', ax=ax,
                color=['#2ecc71','#3498db','#f39c12','#e74c3c'], edgecolor='black')
            ax.set_title('Users per Segment')
            ax.set_ylabel('Users')
            ax.tick_params(axis='x', rotation=20)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.subheader("Segment Summary Table")
        st.dataframe(seg_stats[['users','churn_rate','avg_monthly_rev',
                                  'expected_ltv','total_arr']].style.format({
            'churn_rate': '{:.1%}',
            'avg_monthly_rev': '${:.2f}',
            'expected_ltv': '${:.0f}',
            'total_arr': '${:,.0f}'
        }))
    except:
        st.warning("Run the segmentation notebook cell first")

with tab3:
    st.subheader("Cohort Retention Analysis")
    
    try:
        cohort_data = pd.read_csv(PROJECT_ROOT / 'data' / 'cohort_retention.csv')
        retention_pivot = cohort_data.pivot_table(
            index='cohort', columns='plan_tier', values='retention_rate'
        )
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(retention_pivot, annot=True, fmt='.1%',
                    cmap='RdYlGn', ax=ax, vmin=0.4, vmax=1.0,
                    linewidths=0.5)
        ax.set_title('Retention Rate by Cohort & Plan Tier')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        **Key Insights:**
        - 🏆 **Premium plan** retains 82-87% across ALL cohorts
        - 📉 **Monthly plan** retains only 50-55% — upgrade path critical
        - 📈 **Annual plan** improves retention with tenure (71% → 78%)
        """)
    except:
        st.warning("Run the cohort analysis notebook cell first")

with tab4:
    st.subheader("🔍 Individual User Risk Explorer")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Adjust user profile:**")
        active_days   = st.slider("Daily Active Days", 0, 30, 10)
        adoption      = st.slider("Feature Adoption Score", 0, 100, 40)
        last_login    = st.slider("Days Since Last Login", 0, 60, 10)
        payment_fail  = st.selectbox("Payment Failures", [0, 1, 2, 3])
        plan          = st.selectbox("Plan Tier", ['monthly', 'annual', 'premium'])
        goals         = st.checkbox("Goals Set", value=True)
        
        risk = (
            0.3 * (1 - active_days/30) +
            0.2 * (last_login/60) +
            0.2 * (payment_fail/3) +
            0.15 * (1 - adoption/100) +
            0.15 * (1 - int(goals))
        )
        
        plan_modifier = {'monthly': 1.3, 'annual': 0.85, 'premium': 0.6}
        risk *= plan_modifier[plan]
        risk = min(risk, 1.0)
    
    with col2:
        color = "🔴" if risk > 0.6 else "🟡" if risk > 0.3 else "🟢"
        tier  = "HIGH RISK" if risk > 0.6 else "MEDIUM RISK" if risk > 0.3 else "LOW RISK"
        
        st.metric("Churn Risk Score", f"{risk:.1%}")
        st.markdown(f"## {color} {tier}")
        
        st.markdown("**Risk Factor Breakdown:**")
        factors = {
            'Inactivity':       0.3 * (1 - active_days/30),
            'Login Recency':    0.2 * (last_login/60),
            'Payment Issues':   0.2 * (payment_fail/3),
            'Low Adoption':     0.15 * (1 - adoption/100),
            'No Goals Set':     0.15 * (1 - int(goals))
        }
        
        fig, ax = plt.subplots(figsize=(6,3))
        colors_f = ['#e74c3c' if v > 0.1 else '#f39c12' if v > 0.05 
                    else '#2ecc71' for v in factors.values()]
        ax.barh(list(factors.keys()), list(factors.values()),
                color=colors_f, edgecolor='black')
        ax.set_xlabel('Risk Contribution')
        ax.set_title('Risk Score Components')
        plt.tight_layout()
        st.pyplot(fig)

st.divider()
st.caption("SaaS Customer Intelligence Platform | Synthetic Dataset | 10,000 Users")