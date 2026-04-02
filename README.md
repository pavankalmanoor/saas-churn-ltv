# 📊 SaaS Customer Intelligence Platform
> Churn Prediction · LTV Analysis · Behavioral Segmentation · Product Analytics

End-to-end customer intelligence platform built on a synthetic SaaS dataset of 10,000 users, featuring churn prediction, RFM segmentation, LTV modeling, cohort retention analysis, A/B testing, and an interactive 4-tab Streamlit dashboard.

---

## 🎯 Project Overview

**Pipeline:**
```
Synthetic SaaS Dataset (10,000 users)
        ↓
RFM Feature Engineering
        ↓
K-Means Behavioral Segmentation + LTV Modeling
        ↓
Cohort Retention Analysis + A/B Test Framework
        ↓
XGBoost Churn Model + SHAP Explainability + MLflow
        ↓
4-Tab Interactive Streamlit Dashboard
```

---

## 📸 Dashboard Screenshots

### Overview & KPIs
![Dashboard Overview](data/Screenshot_2026-04-01_at_8_04_11_PM.png)

### LTV & Customer Segments
![LTV Segments](data/Screenshot_2026-04-01_at_8_04_16_PM.png)

### Cohort Retention Heatmap
![Cohort Retention](data/Screenshot_2026-04-01_at_8_04_27_PM.png)

### Individual User Risk Explorer
![User Explorer](data/Screenshot_2026-04-01_at_8_04_34_PM.png)

---

## 📊 Key Results

### Business KPIs
| Metric | Value |
|---|---|
| Total Users | 10,000 |
| Churned Users | 3,539 |
| Overall Churn Rate | 35.4% |
| Monthly Recurring Revenue | $294,886 |
| High Risk Users | 195 |

### Churn Model Performance
| Metric | Value |
|---|---|
| CV AUC | 0.673 ± 0.008 |
| Test AUC | 0.643 |
| Dataset | Synthetic (10K users, 16 features) |

### Customer Segments
| Segment | Users | Churn Rate | Expected LTV | ARR |
|---|---|---|---|---|
| 🏆 Champions | 1,779 | 24.0% | $140 | $719,001 |
| ⚠️ At Risk | 2,461 | 35.0% | $88 | $911,653 |
| 💚 Loyal Users | 2,827 | 35.0% | $79 | $939,695 |
| 🔴 About to Churn | 2,933 | 43.0% | $64 | $968,242 |

### Cohort Retention
| Plan | Avg Retention |
|---|---|
| Premium | 82–87% across all cohorts |
| Annual | 71–78% (improves with tenure) |
| Monthly | 50–55% (highest churn risk) |

### A/B Test — Onboarding Improvement
| Group | Churn Rate |
|---|---|
| Control (Standard) | 40.4% |
| Treatment (Improved) | 35.8% |
| Relative Lift | 11.4% reduction |
| p-value | 0.152 (not significant at n=500) |
| Required sample | ~1,200/arm for 80% power |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Data Generation** | NumPy, Pandas (synthetic SaaS behavioral data) |
| **Segmentation** | K-Means, RFM Analysis, MinMaxScaler |
| **LTV Modeling** | Revenue/churn rate formula, segment-level LTV |
| **Churn Prediction** | XGBoost, Stratified K-Fold CV |
| **Explainability** | SHAP (TreeExplainer) |
| **Experimentation** | SciPy chi-squared A/B testing |
| **Experiment Tracking** | MLflow |
| **Dashboard** | Streamlit (4 tabs) |
| **Visualization** | Matplotlib, Seaborn |

---

## 🚀 Quick Start

```bash
git clone https://github.com/pavankalmanoor/saas-churn-ltv
cd saas-churn-ltv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/01_saas_intelligence.ipynb

# Run dashboard
streamlit run src/app.py
```

---

## 📁 Project Structure

```
saas-churn-ltv/
├── data/
│   ├── saas_users.csv              # Synthetic dataset (10K users)
│   ├── segment_stats.csv           # RFM segment summary
│   ├── cohort_retention.csv        # Cohort analysis results
│   ├── eda_plots.png               # EDA visualizations
│   ├── segmentation_ltv.png        # LTV segmentation charts
│   ├── cohort_ab_test.png          # Cohort & A/B test plots
│   ├── shap_churn.png              # SHAP feature importance
│   └── confusion_matrix.png        # Model evaluation
├── notebooks/
│   └── 01_saas_intelligence.ipynb # Full analysis pipeline
├── src/
│   └── app.py                      # Streamlit dashboard
├── mlruns/                         # MLflow experiment logs
├── requirements.txt
└── README.md
```

---

## ⚙️ Feature Engineering

**16 features across 4 categories:**

**Engagement (6):**
- `daily_active_days` — activity in last 30 days
- `avg_session_mins` — session length
- `feature_adoption_score` — % of features used (0-100)
- `goals_set` — whether user set health/fitness goals
- `social_connections` — friend count on platform
- `streak_days` — consecutive active days

**Support & Payment (3):**
- `support_tickets_30d` — support burden
- `payment_failures` — billing issues
- `days_since_last_login` — recency signal

**Business (3):**
- `monthly_charge` — revenue per user
- `tenure_months` — customer age
- `plan_tier` — monthly/annual/premium

**RFM Derived (4):**
- `recency_score`, `frequency_score`, `monetary_score`, `rfm_score`

---

## 🔍 Key Findings

**Churn drivers (SHAP):**
- Plan tier is the strongest predictor — premium users 3x less likely to churn
- Goals set reduces churn significantly — product engagement matters
- Payment failures strongly predict churn — billing health = retention health
- RFM score captures composite engagement risk

**Segmentation insights:**
- Champions (1,779 users) have $140 LTV vs $64 for About-to-Churn — 2.2x gap
- 2,933 users "About to Churn" represent $968K ARR at risk
- Intervention priority: convert monthly → annual users (retention jumps 20%)

**A/B test:**
- 11.4% relative lift from improved onboarding didn't reach significance
- Honest reporting — would need 1,200 users/arm for 80% power at this effect size
- This is how real product experimentation works

---

## 💡 Interview Talking Points

**Why synthetic data?**
Real SaaS behavioral data is proprietary. Synthetic data lets us design realistic churn signals based on published SaaS research — DAU/MAU ratios, payment failure rates, and feature adoption benchmarks from industry reports.

**Why AUC 0.643?**
Realistic. The synthetic data has intentional noise to mimic real-world messiness. A model claiming 0.95 AUC on behavioral churn data would be overfit or leaking.

**Why report a non-significant A/B test?**
Because that's what real experimentation looks like. Reporting only significant results is p-hacking. Showing the required sample size calculation demonstrates statistical maturity.

**LTV formula:**
LTV = Monthly Revenue / Churn Rate — standard SaaS industry formula used by Stripe, Recurly, and Baremetrics.
