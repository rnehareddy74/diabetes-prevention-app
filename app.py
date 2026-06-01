import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ──
st.set_page_config(
    page_title="Diabetes Prevention Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Load models ──
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, 'models', 'risk_models.pkl'), 'rb') as f:
        risk = pickle.load(f)
    with open(os.path.join(base, 'models', 'glucose_models.pkl'), 'rb') as f:
        gluc = pickle.load(f)
    with open(os.path.join(base, 'models', 'intervention_engine.pkl'), 'rb') as f:
        eng = pickle.load(f)
    return risk, gluc, eng

@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    fndds = pd.read_csv(os.path.join(base, 'data', 'fndds_nutrition_lookup.csv'))
    swap  = pd.read_csv(os.path.join(base, 'data', 'food_swap_database_fndds.csv'))
    return fndds, swap

risk_pkg, gluc_pkg, eng_pkg = load_models()
fndds_df, swap_df           = load_data()

# ── Prediction functions ──
def predict_risk_score(user):
    nc_feats = risk_pkg['features_nc']
    model_nc = risk_pkg['model_nc']
    iso_reg  = risk_pkg['iso_reg']
    row = np.array([[user.get(f, 0) for f in nc_feats]])
    raw = model_nc.predict_proba(row)[:, 1]
    score = float((iso_reg.predict(raw) * 100).round(1)[0])
    band = subtitle = ''
    for b, info in risk_pkg['score_bands'].items():
        lo, hi = info['range']
        if lo <= score <= hi:
            band     = b
            subtitle = info['subtitle']
            break
    return score, band, subtitle

def predict_clinical_label(user):
    c_feats   = risk_pkg['features_c']
    model_c   = risk_pkg['model_c']
    label_dec = risk_pkg['label_dec']
    row  = np.array([[user.get(f, 0) for f in c_feats]])
    pred = int(model_c.predict(row)[0])
    probs = dict(zip(risk_pkg['label_order'],
                     model_c.predict_proba(row)[0].round(4)))
    return label_dec[pred], probs

def predict_glucose_spike(meal, user):
    features     = gluc_pkg['features']
    train_median = gluc_pkg['train_median']
    model_g      = gluc_pkg['model']
    combined = {**user, **meal}
    row = np.array([[combined.get(f, train_median.get(f, 0)) for f in features]])
    spike = float(model_g.predict(row)[0])
    if   spike < 20: cat = 'Low'
    elif spike < 40: cat = 'Moderate'
    elif spike < 70: cat = 'High'
    else:             cat = 'Very High'
    return round(spike, 1), cat

def get_spike_color(cat):
    return {'Low': '#2ecc71', 'Moderate': '#f39c12',
            'High': '#e67e22', 'Very High': '#e74c3c'}.get(cat, '#gray')

# ── Intervention engine ──
ADA_TARGETS = {
    'exercise_min_week': 150, 'sleep_hrs_min': 7, 'sleep_hrs_max': 9,
    'drinks_men': 14, 'drinks_women': 7,
}
DPP_URL    = 'https://www.cdc.gov/diabetes/prevention/find-a-program.html'
DOCTOR_URL = 'https://findahealthcenter.hrsa.gov/'

def get_care_tier(score, label, hba1c=None, fg=None, phq9=0, diagnosed=False):
    if diagnosed or label == 'diabetic' or score >= 80:
        tier = 3
    elif (hba1c and hba1c >= 6.5) or (fg and fg >= 126):
        tier = 3
    elif label in ['prediabetic','high_risk_prediabetic','early_insulin_resistance'] or score >= 35:
        tier = 2
    else:
        tier = 1
    if phq9 >= 10 and tier < 2:
        tier = 2
    return tier

def get_interventions(user, score, label):
    bench     = risk_pkg['benchmarks']
    risk_bench = bench.get(
        'diabetic' if label == 'diabetic' else
        'prediabetic' if label in ['prediabetic','high_risk_prediabetic','early_insulin_resistance'] else
        'normal', bench.get('normal', {}))

    interventions = []

    # Exercise
    ex_min  = user.get('total_exercise_min_week', 0)
    ex_gap  = max(0, 150 - ex_min)
    ex_bench = risk_bench.get('avg_exercise_min_week', 150)
    ex_status = 'Meeting Target' if ex_min >= 150 else 'Below Target' if ex_min >= 75 else 'Very Low'
    ex_recs = []
    if ex_gap > 0:
        ex_recs.append(f"Add {max(1, round(ex_gap/30))} x 30-min brisk walks per week to close the {ex_gap:.0f} min gap")
    ex_recs += ["Post-meal walks (15-20 min) reduce glucose spike by ~20%",
                "Resistance training 2x/week improves insulin sensitivity"]
    interventions.append({
        'domain': 'Exercise', 'icon': '🏃',
        'status': ex_status,
        'priority': 1 if ex_min < 75 else 2 if ex_min < 150 else 5,
        'user_value': f"{ex_min:.0f} min/week",
        'target': "150 min/week (ADA DPP)",
        'nhanes': f"People at your risk level average {ex_bench:.0f} min/week",
        'recs': ex_recs,
        'ada_rule': "ADA DPP: 150 min/week moderate-intensity activity"
    })

    # Sleep
    sleep     = user.get('avg_sleep_hrs', 7)
    sl_deficit = max(0, 7 - sleep)
    sl_bench  = risk_bench.get('pct_poor_sleep', 20)
    sl_status = 'Optimal' if 7 <= sleep <= 9 else 'Slightly Low' if sleep >= 6 else 'Very Low'
    sl_recs   = []
    if sl_deficit > 0:
        sl_recs.append(f"Target 7 hrs minimum — you have a {sl_deficit:.1f} hr nightly deficit")
    sl_recs += ["Consistent bed/wake times within 30 min regulates cortisol",
                "Avoid screens 1 hr before bed — blue light suppresses melatonin"]
    interventions.append({
        'domain': 'Sleep', 'icon': '😴',
        'status': sl_status,
        'priority': 1 if sleep < 6 else 2 if sleep < 7 else 5,
        'user_value': f"{sleep:.1f} hrs/night",
        'target': "7–9 hrs (ADA)",
        'nhanes': f"{sl_bench:.0f}% of people at your risk level report poor sleep",
        'recs': sl_recs,
        'ada_rule': "ADA: 7-9 hours of quality sleep for optimal glucose regulation"
    })

    # Mental Health
    phq9 = user.get('depression_score', 0)
    if   phq9 <= 4:  mh_cat = 'Minimal';  mh_status = 'Good'
    elif phq9 <= 9:  mh_cat = 'Mild';     mh_status = 'Mild Concern'
    elif phq9 <= 14: mh_cat = 'Moderate'; mh_status = 'Moderate Concern'
    else:             mh_cat = 'Severe';   mh_status = 'High Concern'
    mh_bench = risk_bench.get('pct_depression_moderate', 7)
    mh_recs  = []
    if phq9 <= 9:
        mh_recs += ["Exercise (30 min/day) reduces depression score by 2-3 points",
                    "10 min daily mindfulness shown effective in DPP studies"]
    else:
        mh_recs += ["Score suggests professional support — see care pathway below",
                    "Depression at this level impairs diabetes self-management"]
    interventions.append({
        'domain': 'Mental Health', 'icon': '🧠',
        'status': mh_status,
        'priority': 1 if phq9 >= 10 else 2 if phq9 >= 5 else 5,
        'user_value': f"PHQ-9: {phq9} ({mh_cat})",
        'target': "PHQ-9 < 5 (Minimal)",
        'nhanes': f"{mh_bench:.0f}% moderate depression at your risk level",
        'recs': mh_recs,
        'ada_rule': "ADA: PHQ-9 >= 10 requires provider referral"
    })

    # Alcohol
    drinks  = user.get('drinks_per_week', 0)
    sex     = 'male' if user.get('sex_enc', 1) == 1 else 'female'
    limit   = ADA_TARGETS['drinks_men'] if sex == 'male' else ADA_TARGETS['drinks_women']
    al_bench = risk_bench.get('pct_drinks_past_year', 75)
    al_status = 'Good' if drinks <= limit/2 else 'At Limit' if drinks <= limit else 'Above Limit'
    al_recs = []
    if drinks > limit:
        al_recs.append(f"Reduce to {limit} drinks/week (ADA limit for {sex}s)")
        al_recs.append("Never drink on empty stomach — causes reactive hypoglycemia")
    else:
        al_recs.append(f"Target {int(limit/2)} drinks/week for optimal glucose control")
    interventions.append({
        'domain': 'Alcohol', 'icon': '🍺',
        'status': al_status,
        'priority': 1 if drinks > limit else 3 if drinks > limit/2 else 5,
        'user_value': f"{drinks:.0f} drinks/week",
        'target': f"≤{limit} drinks/week (ADA)",
        'nhanes': f"{al_bench:.0f}% of people at your risk level drink alcohol",
        'recs': al_recs,
        'ada_rule': f"ADA: Limit alcohol to {limit} drinks/week for {sex}s"
    })

    # Smoking
    smoking = user.get('smoking_enc', 0)
    sm_map  = {0: 'Never Smoked', 1: 'Former Smoker', 2: 'Current Smoker'}
    sm_status = sm_map.get(smoking, 'Never Smoked')
    sm_bench  = risk_bench.get('pct_current_smoker', 15)
    sm_recs   = []
    if smoking == 2:
        sm_recs += ["Cessation reduces HbA1c by ~0.2% within weeks",
                    "CDC Smokefree (smokefree.gov) — free text/app support",
                    "Nicotine replacement preferred over vaping"]
    elif smoking == 1:
        sm_recs += ["Insulin resistance from smoking normalizes over 5-10 years",
                    "Maintaining non-smoking status is one of the highest-impact choices"]
    else:
        sm_recs += ["Non-smoking is protective for insulin sensitivity — maintain this"]
    interventions.append({
        'domain': 'Smoking', 'icon': '🚬',
        'status': sm_status,
        'priority': 1 if smoking == 2 else 5,
        'user_value': sm_status,
        'target': "Non-smoker",
        'nhanes': f"{sm_bench:.0f}% of people at your risk level currently smoke",
        'recs': sm_recs,
        'ada_rule': "ADA: Smoking raises insulin resistance and HbA1c"
    })

    # Diet
    diet_recs = []
    if label in ['diabetic', 'high_risk_prediabetic']:
        diet_recs += ["Fill half your plate with non-starchy vegetables (ADA Plate Method)",
                      "Limit carbohydrates to 45-60g per meal",
                      "Use Meal Checker tab to find food swaps"]
    elif label in ['prediabetic', 'early_insulin_resistance']:
        diet_recs += ["Reduce refined grains and added sugars",
                      "Add fiber-rich foods (beans, lentils) to slow glucose absorption",
                      "Choose whole fruit over fruit juice"]
    else:
        diet_recs += ["Maintain balanced meals with vegetables, lean protein, whole grains",
                      "Limit sugary drinks — choose water or unsweetened beverages"]
    dpp_pct = risk_bench.get('pct_meeting_dpp_target', 40)
    interventions.append({
        'domain': 'Diet', 'icon': '🥗',
        'status': 'Review Needed' if label in ['diabetic','high_risk_prediabetic'] else 'Monitor',
        'priority': 1 if label in ['diabetic','high_risk_prediabetic'] else 3,
        'user_value': 'Based on risk profile',
        'target': "ADA Plate Method",
        'nhanes': f"{100-dpp_pct:.0f}% at your risk level not meeting dietary targets",
        'recs': diet_recs,
        'ada_rule': "ADA Plate Method: half non-starchy veg, quarter lean protein, quarter carbs"
    })

    return sorted(interventions, key=lambda x: x['priority'])

# ── Status color helper ──
def status_color(status):
    green  = ['Meeting Target','Optimal','Good','Never Smoked','Monitor']
    yellow = ['Near Target','Slightly Low','Mild Concern','At Limit','Former Smoker']
    red    = ['Below Target','Very Low','Moderate Concern','High Concern',
              'Above Limit','Current Smoker','Review Needed']
    if status in green:  return '#2ecc71'
    if status in yellow: return '#f39c12'
    if status in red:    return '#e74c3c'
    return '#95a5a6'

# ── CSS ──
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; text-align: center; margin-bottom: 2rem;
        color: white;
    }
    .risk-card {
        background: white; border-radius: 12px; padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 1rem;
        border-left: 5px solid #3498db;
    }
    .intervention-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 0.8rem;
    }
    .band-badge {
        display: inline-block; padding: 0.4rem 1rem; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem; color: white;
    }
    .metric-row {
        display: flex; gap: 1rem; margin-bottom: 1rem;
    }
    .metric-box {
        background: #f8f9fa; border-radius: 8px; padding: 0.8rem 1.2rem;
        flex: 1; text-align: center;
    }
    .tier-banner {
        border-radius: 10px; padding: 1rem 1.5rem; margin-bottom: 1.5rem;
        color: white; font-weight: bold;
    }
    .disclaimer {
        background: #f8f9fa; border-radius: 8px; padding: 1rem;
        font-size: 0.8rem; color: #666; margin-top: 2rem;
        border-left: 3px solid #bdc3c7;
    }
    div[data-testid="stTabs"] button { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2rem;">🏥 Early Diabetes Prevention Platform</h1>
    <p style="margin:0.5rem 0 0; opacity:0.85;">
        Personalized risk assessment · Lifestyle interventions · Meal glucose prediction
    </p>
    <p style="margin:0.3rem 0 0; font-size:0.8rem; opacity:0.6;">
        Based on NHANES data · ADA 2023 Guidelines · USDA FNDDS 2021-2023
    </p>
</div>
""", unsafe_allow_html=True)

# ── Session state ──
if 'results' not in st.session_state:
    st.session_state.results = None
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Risk Assessment",
    "💡 Interventions",
    "🍽️ Meal Checker",
    "ℹ️ About"
])

# ══════════════════════════════════════════════
# TAB 1 — RISK ASSESSMENT
# ══════════════════════════════════════════════
with tab1:
    st.subheader("Enter Your Health Profile")

    with st.form("risk_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Basic Information**")
            age    = st.number_input("Age", 18, 90, 45, help="Years")
            sex    = st.selectbox("Sex", ["Male", "Female"])
            height = st.number_input("Height (cm)", 140, 220, 170)
            weight = st.number_input("Weight (kg)", 40, 200, 80)
            waist  = st.number_input("Waist Circumference (cm)", 50, 180, 90,
                                      help="Measured at navel level")
            diagnosed = st.checkbox("I have been diagnosed with diabetes or prediabetes")

            st.markdown("**Lifestyle**")
            exercise = st.number_input("Exercise (minutes/week)", 0, 1000, 90,
                                        help="All moderate activity combined")
            sleep    = st.number_input("Average Sleep (hours/night)", 3.0, 12.0, 7.0, step=0.5)
            sedentary = st.number_input("Sedentary Time (hours/day)", 0, 20, 8,
                                         help="Sitting/lying excluding sleep")
            smoking  = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            drinks   = st.number_input("Alcoholic Drinks per Week", 0, 60, 3)
            phq9     = st.slider("Depression Score (PHQ-9)", 0, 27, 3,
                                  help="0=Minimal  5-9=Mild  10-14=Moderate  15+=Severe")

        with col2:
            st.markdown("**Lab Results (Optional)**")
            st.info("Leave at 0 if you don't have lab results. Model A (lifestyle only) will be used.")
            hba1c    = st.number_input("HbA1c (%)", 0.0, 20.0, 0.0, step=0.1,
                                        help="Glycated hemoglobin — 3-month average blood sugar")
            fg       = st.number_input("Fasting Glucose (mg/dL)", 0, 500, 0,
                                        help="Blood sugar after 8+ hours fasting")
            homa_ir  = st.number_input("HOMA-IR", 0.0, 20.0, 0.0, step=0.1,
                                        help="Insulin resistance score — fasting glucose x insulin / 405")
            trig     = st.number_input("Triglycerides (mg/dL)", 0, 1000, 0)
            hdl      = st.number_input("HDL Cholesterol (mg/dL)", 0, 200, 0,
                                        help="Good cholesterol")
            total_chol = st.number_input("Total Cholesterol (mg/dL)", 0, 500, 0)

            st.markdown("**Computed Values**")
            bmi = weight / ((height/100)**2)
            whr = waist / height
            st.metric("BMI", f"{bmi:.1f}")
            st.metric("Waist-Height Ratio", f"{whr:.3f}")

        submitted = st.form_submit_button("🔍 Calculate My Risk", use_container_width=True)

    if submitted:
        # Build user profile
        smoking_enc = {'Never': 0, 'Former': 1, 'Current': 2}[smoking]
        sex_enc     = 1 if sex == 'Male' else 0

        user = {
            'age': age, 'sex_enc': sex_enc, 'bmi': bmi,
            'waist_cm': waist, 'waist_height_ratio': whr,
            'total_exercise_min_week': exercise,
            'avg_sleep_hrs': sleep, 'depression_score': phq9,
            'smoking_enc': smoking_enc, 'drinks_per_week': drinks,
            'sedentary_min_day': sedentary * 60,
            'sex': sex_enc,
        }
        user['age_bmi']        = age * bmi / 100
        user['exercise_sleep'] = exercise * sleep
        user['waist_age']      = waist * age / 100

        # Add lab values if provided
        has_labs = hba1c > 0 and fg > 0
        if has_labs:
            user.update({
                'hba1c': hba1c, 'fasting_glucose': fg,
                'homa_ir': homa_ir, 'triglycerides': trig,
                'hdl_cholesterol': hdl, 'total_cholesterol': total_chol
            })

        # Run models
        score, band, subtitle = predict_risk_score(user)

        if has_labs:
            label, probs = predict_clinical_label(user)
            model_used   = 'Clinical (with lab values)'
        else:
            label_map = {
                'Diabetes Unlikely'    : 'normal',
                'Early Warning'        : 'early_insulin_resistance',
                'Prediabetes Territory': 'prediabetic',
                'Diabetes Territory'   : 'diabetic'
            }
            label      = label_map.get(band, 'normal')
            probs      = None
            model_used = 'Lifestyle only (no lab values)'

        tier = get_care_tier(score, label, hba1c if has_labs else None,
                              fg if has_labs else None, phq9, diagnosed)

        st.session_state.results = {
            'score': score, 'band': band, 'subtitle': subtitle,
            'label': label, 'probs': probs, 'tier': tier,
            'model_used': model_used, 'has_labs': has_labs,
        }
        st.session_state.user_profile = user
        st.session_state.user_profile.update({
            'phq9_score': phq9, 'already_diagnosed': diagnosed,
            'hba1c': hba1c if has_labs else None,
            'fasting_glucose': fg if has_labs else None,
        })

    # Show results
    if st.session_state.results:
        r = st.session_state.results
        st.divider()
        st.subheader("Your Risk Assessment")

        band_colors = {
            'Diabetes Unlikely'    : '#2ecc71',
            'Early Warning'        : '#f39c12',
            'Prediabetes Territory': '#e67e22',
            'Diabetes Territory'   : '#e74c3c',
        }
        color = band_colors.get(r['band'], '#95a5a6')

        col1, col2 = st.columns([1, 1])

        with col1:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = r['score'],
                title = {'text': "Risk Score", 'font': {'size': 18}},
                number = {'font': {'size': 36, 'color': color}},
                gauge = {
                    'axis'  : {'range': [0, 100], 'tickwidth': 1},
                    'bar'   : {'color': color, 'thickness': 0.3},
                    'steps' : [
                        {'range': [0,  34], 'color': '#d5f5e3'},
                        {'range': [34, 59], 'color': '#fef9e7'},
                        {'range': [59, 79], 'color': '#fdebd0'},
                        {'range': [79, 100],'color': '#fadbd8'},
                    ],
                    'threshold': {
                        'line' : {'color': color, 'width': 4},
                        'thickness': 0.75,
                        'value': r['score']
                    }
                }
            ))
            fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="risk-card" style="border-left-color:{color}">
                <span class="band-badge" style="background:{color}">
                    {r['band']}
                </span>
                <p style="font-size:1.1rem; margin:0.8rem 0 0.3rem; color:#555;">
                    {r['subtitle']}
                </p>
                <hr style="margin:0.8rem 0; border-color:#eee">
                <p><b>Model used:</b> {r['model_used']}</p>
                <p><b>Clinical label:</b> {r['label'].replace('_',' ').title()}</p>
            </div>
            """, unsafe_allow_html=True)

            # Band legend
            st.markdown("**Risk Band Guide**")
            for b, c in band_colors.items():
                marker = "◀ You are here" if b == r['band'] else ""
                st.markdown(
                    f"<span style='color:{c}'>●</span> **{b}** {marker}",
                    unsafe_allow_html=True
                )

        # Care pathway
        tier_colors = {1: '#2ecc71', 2: '#f39c12', 3: '#e74c3c'}
        tier_labels = {
            1: '✅ Tier 1 — Self-Management',
            2: '⚠️ Tier 2 — DPP Program Recommended',
            3: '🚨 Tier 3 — Please See a Doctor'
        }
        tier_descs = {
            1: 'Your risk can be managed with lifestyle changes. Follow the recommendations below.',
            2: (f'You qualify for the CDC Diabetes Prevention Program. '
                f'It reduces diabetes risk by 58%. Medicare & most insurers cover it. '
                f'[Find a program near you]({DPP_URL})'),
            3: (f'Your results suggest speaking with a healthcare provider soon. '
                f'Request: HbA1c, fasting glucose, lipid panel. '
                f'[Find a low-cost health center]({DOCTOR_URL})')
        }
        st.markdown(f"""
        <div class="tier-banner" style="background:{tier_colors[r['tier']]}">
            {tier_labels[r['tier']]}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(tier_descs[r['tier']])

        if r['probs']:
            st.markdown("**Clinical Risk Breakdown** *(internal reference)*")
            prob_df = pd.DataFrame({
                'Risk Level': [k.replace('_',' ').title() for k in r['probs'].keys()],
                'Probability': [f"{v*100:.1f}%" for v in r['probs'].values()]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)

        st.info("👉 Go to the **Interventions** tab to see your personalized action plan")

# ══════════════════════════════════════════════
# TAB 2 — INTERVENTIONS
# ══════════════════════════════════════════════
with tab2:
    if not st.session_state.results:
        st.info("Please complete the Risk Assessment first.")
    else:
        r    = st.session_state.results
        user = st.session_state.user_profile
        interventions = get_interventions(user, r['score'], r['label'])

        st.subheader(f"Your Personalized Intervention Plan")
        st.markdown(f"**Risk band:** {r['band']} &nbsp;|&nbsp; **Priority domains:** {', '.join([i['domain'] for i in interventions[:3]])}")

        # Urgent domains
        urgent = [i for i in interventions if i['priority'] <= 2]
        if urgent:
            st.markdown("### 🎯 Focus Here First")
            cols = st.columns(len(urgent))
            for col, item in zip(cols, urgent):
                with col:
                    col.markdown(f"""
                    <div style="background:{status_color(item['status'])}22;
                                border-left:4px solid {status_color(item['status'])};
                                border-radius:8px; padding:0.8rem; text-align:center">
                        <div style="font-size:1.8rem">{item['icon']}</div>
                        <div style="font-weight:bold">{item['domain']}</div>
                        <div style="color:{status_color(item['status'])};font-weight:600">{item['status']}</div>
                        <div style="font-size:0.85rem;color:#666">{item['user_value']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("### 📊 All Domains")

        for item in interventions:
            with st.expander(
                f"{item['icon']} {item['domain']} — {item['status']} | {item['user_value']}",
                expanded=(item['priority'] <= 2)
            ):
                col1, col2, col3 = st.columns(3)
                col1.metric("Your Value",  item['user_value'])
                col2.metric("Target",      item['target'])
                sc = status_color(item['status'])
                col3.markdown(
                    f"<div style='background:{sc}22;border-radius:6px;padding:0.5rem;"
                    f"text-align:center;color:{sc};font-weight:bold'>"
                    f"{item['status']}</div>",
                    unsafe_allow_html=True
                )

                st.markdown(f"📊 **NHANES context:** {item['nhanes']}")
                st.markdown("**Recommendations:**")
                for rec in item['recs']:
                    st.markdown(f"• {rec}")
                st.caption(f"📋 {item['ada_rule']}")

        # Care pathway
        st.divider()
        tier_colors = {1:'#2ecc71', 2:'#f39c12', 3:'#e74c3c'}
        tier_labels = {
            1:'✅ Tier 1 — Self-Management',
            2:'⚠️ Tier 2 — DPP Program Recommended',
            3:'🚨 Tier 3 — Please See a Doctor'
        }
        st.markdown(f"""
        <div class="tier-banner" style="background:{tier_colors[r['tier']]}">
            {tier_labels[r['tier']]}
        </div>
        """, unsafe_allow_html=True)

        if r['tier'] == 2:
            st.markdown(f"**CDC Diabetes Prevention Program** — reduces diabetes risk by 58%. "
                        f"Covered by Medicare & most insurers. [Find a program]({DPP_URL})")
        elif r['tier'] == 3:
            st.markdown(f"**Please speak with a healthcare provider.** "
                        f"Request HbA1c, fasting glucose, and lipid panel. "
                        f"[Find a low-cost health center]({DOCTOR_URL})")
            st.markdown("**Bring to your appointment:**")
            for item in ["This risk assessment summary", "HbA1c if available",
                          "Current medications", "Family history of diabetes"]:
                st.markdown(f"• {item}")

        if user.get('phq9_score', 0) >= 10:
            st.error("🧠 Mental health score suggests professional support. "
                     "Please mention this to your healthcare provider. "
                     "Crisis support: **988 Suicide & Crisis Lifeline** (call or text 988)")

        st.markdown("""
        <div class="disclaimer">
        ⚕️ This tool is a screening aid only and does not replace medical advice.
        Always consult a qualified healthcare provider for diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — MEAL CHECKER
# ══════════════════════════════════════════════
with tab3:
    st.subheader("🍽️ Meal Glucose Checker")
    st.markdown("Search for a food, enter quantity, and see the predicted glucose spike + swap recommendation.")

    # Food search
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search food", placeholder="e.g. white bread, orange juice, rice...")
    with col2:
        meal_type = st.selectbox("Meal type", ["Breakfast", "Lunch", "Dinner", "Snack"])

    meal_hour = {'Breakfast': 8, 'Lunch': 13, 'Dinner': 19, 'Snack': 15}[meal_type]

    selected_food = None
    if query and len(query) >= 2:
        q   = query.lower().strip()
        matches = fndds_df[
            fndds_df['food_name'].str.lower().str.contains(q, na=False)
        ].head(10)

        if len(matches) == 0:
            st.warning("No foods found. Try a different search term.")
        else:
            food_names = matches['food_name'].tolist()
            chosen     = st.selectbox("Select food", food_names)
            selected_food = matches[matches['food_name'] == chosen].iloc[0]

    if selected_food is not None:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            qty  = st.number_input("Quantity", 0.25, 10.0, 1.0, step=0.25)
        with col2:
            st.metric("Portion size", str(selected_food['portion_desc']))
        with col3:
            baseline_glucose = st.number_input(
                "Current glucose (mg/dL)", 60, 300, 95,
                help="Your blood glucose right now if known, else leave at 95"
            )

        # Calculate meal macros
        carbs   = round(float(selected_food['carb_g_per_serving'])   * qty, 1)
        protein = round(float(selected_food['protein_g_per_serving'])* qty, 1)
        fat     = round(float(selected_food['fat_g_per_serving'])    * qty, 1)
        fiber   = round(float(selected_food['fiber_g_per_serving'])  * qty, 1)
        sugar   = round(float(selected_food['sugar_g_per_serving'])  * qty, 1)
        cals    = round(float(selected_food['energy_kcal_per_serving'])* qty, 0)

        # Display nutrition
        st.markdown("**Nutritional Content**")
        ncols = st.columns(6)
        for col, label, val, unit in zip(ncols,
            ['Calories','Carbs','Protein','Fat','Fiber','Sugar'],
            [cals, carbs, protein, fat, fiber, sugar],
            ['kcal','g','g','g','g','g']):
            col.metric(label, f"{val}{unit}")

        # Build meal dict for prediction
        high_carb  = 1 if carbs > 60 else 0
        low_fiber  = 1 if fiber < 3  else 0
        high_cal   = 1 if cals  > 500 else 0
        cfr        = carbs / (fiber + 1)

        user_prof  = st.session_state.user_profile if st.session_state.user_profile else {}
        age_val    = user_prof.get('age', 45)
        bmi_val    = user_prof.get('bmi', 27)
        hba1c_val  = user_prof.get('hba1c', 5.5) or 5.5
        fg_val     = user_prof.get('fasting_glucose', 95) or 95
        homa_val   = user_prof.get('homa_ir', 1.5) or 1.5
        hdl_val    = user_prof.get('hdl_cholesterol', 50) or 50
        trig_val   = user_prof.get('triglycerides', 120) or 120
        sex_val    = user_prof.get('sex_enc', 1)

        meal_dict = {
            'meal_carbs'     : carbs,
            'meal_protein'   : protein,
            'meal_fat'       : fat,
            'meal_fiber'     : fiber,
            'baseline_glucose': baseline_glucose,
            'high_carb_meal' : high_carb,
            'low_fiber_meal' : low_fiber,
            'high_cal_meal'  : high_cal,
            'time_to_peak_min': 60,
            'age'            : age_val,
            'sex'            : sex_val,
            'bmi'            : bmi_val,
            'hba1c'          : hba1c_val,
            'fasting_glucose': fg_val,
            'homa_ir'        : homa_val,
            'hdl_cholesterol': hdl_val,
            'triglycerides'  : trig_val,
            'hour'           : meal_hour,
            'carb_fiber_ratio': cfr,
            'ir_proxy'       : homa_val * carbs / 100,
            'carb_x_baseline': carbs * baseline_glucose / 100,
            'mt_breakfast'   : 1 if meal_type == 'Breakfast' else 0,
            'mt_lunch'       : 1 if meal_type == 'Lunch'     else 0,
            'mt_dinner'      : 1 if meal_type == 'Dinner'    else 0,
            'mt_snack'       : 1 if meal_type == 'Snack'     else 0,
        }

        spike, cat = predict_glucose_spike(meal_dict, {})
        spike_color = get_spike_color(cat)

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            # Spike gauge
            fig2 = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = spike,
                title = {'text': "Predicted Glucose Spike (mg/dL)"},
                delta = {'reference': 40, 'increasing': {'color': '#e74c3c'}},
                gauge = {
                    'axis' : {'range': [-20, 150]},
                    'bar'  : {'color': spike_color},
                    'steps': [
                        {'range': [-20, 20], 'color': '#d5f5e3'},
                        {'range': [20, 40],  'color': '#fef9e7'},
                        {'range': [40, 70],  'color': '#fdebd0'},
                        {'range': [70, 150], 'color': '#fadbd8'},
                    ],
                }
            ))
            fig2.update_layout(height=250, margin=dict(l=20,r=20,t=60,b=20))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown(
                f"<div style='text-align:center;background:{spike_color}22;"
                f"border-radius:8px;padding:0.5rem;color:{spike_color};"
                f"font-weight:bold;font-size:1.1rem'>{cat} Spike</div>",
                unsafe_allow_html=True
            )

        with col2:
            # Swap recommendation
            food_code = int(selected_food['food_code'])
            swap_rows = swap_df[swap_df['high_risk_food_code'] == food_code]

            if len(swap_rows) > 0:
                best_swap = swap_rows.sort_values('carb_saving_per_serving', ascending=False).iloc[0]
                swap_food_code = int(best_swap['swap_food_code'])
                swap_fndds = fndds_df[fndds_df['food_code'] == swap_food_code]

                st.markdown("### 🔄 Recommended Swap")
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown(f"**Current: {selected_food['food_name']}**")
                    st.markdown(f"Serving: {selected_food['portion_desc']}")
                    st.markdown(f"Carbs: **{carbs:.1f}g**")
                    st.markdown(f"Fiber: {fiber:.1f}g")
                    st.markdown(f"Sugar: {sugar:.1f}g")
                    st.markdown(f"Calories: {cals:.0f} kcal")

                with c2:
                    st.markdown(f"**Swap to: {best_swap['swap_food_name']}**")
                    st.markdown(f"Serving: {best_swap['swap_portion']}")
                    swap_carbs = round(float(best_swap['swap_carb_per_serving']) * qty, 1)
                    carb_save  = round(float(best_swap['carb_saving_per_serving']) * qty, 1)
                    fiber_gain = round(float(best_swap['fiber_gain_per_serving']) * qty, 1)
                    st.markdown(f"Carbs: **{swap_carbs:.1f}g**")
                    if not swap_fndds.empty:
                        sw_fiber = round(float(swap_fndds.iloc[0]['fiber_g_per_serving']) * qty, 1)
                        sw_sugar = round(float(swap_fndds.iloc[0]['sugar_g_per_serving']) * qty, 1)
                        sw_cals  = round(float(swap_fndds.iloc[0]['energy_kcal_per_serving']) * qty, 0)
                        st.markdown(f"Fiber: {sw_fiber:.1f}g")
                        st.markdown(f"Sugar: {sw_sugar:.1f}g")
                        st.markdown(f"Calories: {sw_cals:.0f} kcal")

                st.success(f"💚 This swap saves **{carb_save:.0f}g carbs** per serving"
                           + (f" and adds **{fiber_gain:.1f}g fiber**" if fiber_gain > 0 else ""))
                st.caption(f"📋 ADA Rule: {best_swap['ada_rule']}")

                # Estimate spike reduction
                if carb_save > 0:
                    swap_meal  = {**meal_dict, 'meal_carbs': max(0, carbs - carb_save),
                                   'meal_fiber': fiber + (fiber_gain if fiber_gain > 0 else 0)}
                    swap_meal['carb_fiber_ratio'] = swap_meal['meal_carbs'] / (swap_meal['meal_fiber'] + 1)
                    swap_meal['ir_proxy']          = homa_val * swap_meal['meal_carbs'] / 100
                    swap_spike, swap_cat           = predict_glucose_spike(swap_meal, {})
                    reduction  = spike - swap_spike
                    if reduction > 0:
                        st.info(f"📉 Estimated spike reduction: **{reduction:.0f} mg/dL** "
                                f"({cat} → {swap_cat})")
            else:
                st.markdown("### ✅ No Swap Needed")
                st.success(f"This food is not flagged as high-risk in the ADA swap database. "
                           f"Predicted spike is {spike:.0f} mg/dL ({cat}).")
                st.markdown(f"**ADA risk level:** {selected_food.get('ada_risk_level','low')}")

        st.caption(f"Model accuracy: MAE {gluc_pkg['mae']} mg/dL | "
                   f"R² {gluc_pkg['r2']} | Directional accuracy {gluc_pkg['dir_accuracy']}% | "
                   f"Based on 45 CGM patients")

# ══════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════
with tab4:
    st.subheader("About This Platform")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### What This Does
        This platform uses machine learning and clinical guidelines
        to provide personalized early diabetes prevention support.

        **Three integrated components:**
        1. **Risk Assessment** — Predicts diabetes risk from lifestyle
           or clinical features using models trained on NHANES data
        2. **Intervention Engine** — 6-domain lifestyle recommendations
           based on ADA 2023 guidelines and NHANES population benchmarks
        3. **Meal Checker** — Predicts post-meal glucose spike and
           suggests evidence-based food swaps from FNDDS/ADA database

        ### Data Sources
        - **NHANES** (National Health and Nutrition Examination Survey)
          — 6,330 US adults for risk model training
        - **USDA FNDDS 2021-2023** — 5,430 foods with full nutrition
        - **ADA Diabetes Plate Method 2023** — food swap guidelines
        - **CGM Dataset** — 45 patients, 1,640 meal sessions for
          glucose spike model
        """)

    with col2:
        st.markdown("""
        ### Model Performance

        | Model | Metric | Value |
        |---|---|---|
        | Risk Score (no labs) | AUC | 0.74 |
        | Clinical Label (with labs) | Accuracy | 99% |
        | Glucose Spike | MAE | 21 mg/dL |
        | Glucose Spike | Directional Acc | 72% |

        ### Limitations
        - Risk score model: 0.74 AUC is the known ceiling
          for lifestyle-only diabetes screening (matches ADA FINDRISC)
        - Glucose model: trained on 45 patients — individual
          variation not fully captured
        - Not validated on external datasets
        - Glucose model underpredicts very high spikes (>70 mg/dL)

        ### Disclaimer
        This tool is for educational and screening purposes only.
        It does not replace medical advice, diagnosis, or treatment.
        Always consult a qualified healthcare provider.
        """)

    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("High-risk foods in database", "558")
    col2.metric("FNDDS foods searchable",      "5,430")
    col3.metric("NHANES patients trained on",  "6,330")
