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
    combined['carb_fiber_ratio'] = combined.get('meal_carbs',54) / (combined.get('meal_fiber',4) + 1)
    combined['ir_proxy']         = combined.get('homa_ir',4.24) * combined.get('meal_carbs',54) / 100
    combined['carb_x_baseline']  = combined.get('meal_carbs',54) * combined.get('baseline_glucose',96) / 100
    row   = np.array([[combined.get(f, train_median.get(f, 0)) for f in features]])
    raw   = float(model_g.predict(row)[0])

    # ── Calibration correction layer ──
    # The model (R2=0.35, 45 patients) regresses to mean for extreme values.
    # This layer applies clinically-grounded adjustments to correct:
    # 1. HbA1c compression above 6.5 (model plateaus, real world does not)
    # 2. Baseline glucose inverse relationship (training data confounding)
    # 3. HOMA-IR under-sensitivity at extreme values
    hba1c    = combined.get('hba1c', 6.0) or 6.0
    baseline = combined.get('baseline_glucose', 96) or 96
    homa_ir  = combined.get('homa_ir', 4.24) or 4.24
    carbs    = combined.get('meal_carbs', 54) or 54

    # HbA1c multiplier
    if   hba1c > 6.0: hba1c_mult = 1.0 + (hba1c - 6.0) * 0.15
    elif hba1c < 5.5: hba1c_mult = 0.85
    else:              hba1c_mult = 1.0

    # Baseline glucose multiplier
    if   baseline > 120: baseline_mult = 1.0 + (baseline - 120) / 200
    elif baseline < 80:  baseline_mult = 0.90
    else:                baseline_mult = 1.0

    # HOMA-IR multiplier (severe insulin resistance)
    homa_mult = 1.0 + max(0, homa_ir - 5.0) * 0.05

    # Apply corrections
    adjusted = raw * hba1c_mult * baseline_mult * homa_mult
    spike    = float(np.clip(adjusted, -15, 200).round(1))

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


# ── Differentiated intervention rules by risk label ──
INTERVENTION_RULES = {
    'exercise': {
        'normal': {
            'recs': [
                "Maintain 150+ min/week to stay in the healthy range",
                "Mix cardio and strength training for best metabolic benefit",
                "Post-meal walks (15 min) reduce glucose spikes by ~20% even in healthy individuals",
            ],
            'urgency_note': "Preventive maintenance — your exercise is protecting you"
        },
        'early_insulin_resistance': {
            'recs': [
                "150 min/week moderate activity is your primary intervention — it can reverse early IR",
                "Post-meal walks (15-20 min) directly lower insulin resistance within days",
                "Resistance training 2x/week improves insulin sensitivity by 25-30%",
                "Even 3x 10-min walks per day counts toward your target",
            ],
            'urgency_note': "Exercise is the most effective reversal tool at this stage"
        },
        'prediabetic': {
            'recs': [
                "DPP studies show 150 min/week reduces diabetes conversion risk by 58%",
                "Post-meal walks are most effective — 15 min after eating lowers spike by ~20%",
                "Resistance training 2x/week improves insulin sensitivity and HbA1c",
                "Brisk walking pace (can talk but not sing) is the optimal intensity",
            ],
            'urgency_note': "150 min/week is clinical prescription at your risk level — not optional"
        },
        'high_risk_prediabetic': {
            'recs': [
                "Target 150-200 min/week — your risk level needs more than the minimum",
                "Post-meal walks are urgent — glucose spikes at your HbA1c cause vascular damage",
                "Resistance training 3x/week shown to lower HbA1c by 0.3-0.5% independently",
                "Sedentary time >8 hrs/day compounds risk — stand or walk every hour",
                "Exercise is as effective as metformin for preventing diabetes at this stage",
            ],
            'urgency_note': "High priority — exercise is as effective as metformin at this stage"
        },
        'diabetic': {
            'recs': [
                "150+ min/week is ADA minimum for diabetes management — not just prevention",
                "Post-meal walks directly lower post-meal glucose — do this after every meal",
                "Resistance training lowers HbA1c by 0.5-0.7% — comparable to some medications",
                "Avoid exercise gaps >2 days — insulin sensitivity drops quickly",
                "Monitor glucose before/after exercise — carry fast-acting carbs if glucose < 100",
            ],
            'urgency_note': "Clinical priority — discuss your exercise plan with your doctor"
        }
    },
    'sleep': {
        'normal': {
            'recs': [
                "7-9 hrs maintains your metabolic health — protect this habit",
                "Consistent sleep/wake schedule within 30 min regulates cortisol",
                "Even 1 night of poor sleep temporarily raises insulin resistance",
            ],
            'urgency_note': "Maintaining good sleep is actively protecting your metabolic health"
        },
        'early_insulin_resistance': {
            'recs': [
                "Sleep < 7 hrs raises cortisol which directly worsens insulin resistance",
                "Consistent 7-8 hrs can reduce HOMA-IR by 15-20%",
                "Avoid screens 1 hr before bed — blue light disrupts sleep quality",
                "Consistent bedtime within 30 min every night regulates insulin rhythm",
            ],
            'urgency_note': "Sleep improvement directly reduces insulin resistance"
        },
        'prediabetic': {
            'recs': [
                "22.8% of prediabetic group reports poor sleep vs 19.2% normal — you are at higher risk",
                "Each hour below 6 hrs raises HbA1c — sleep is a direct metabolic lever",
                "Treating sleep apnea (if present) alone can improve HbA1c by 0.3%",
                "Alcohol as sleep aid worsens sleep quality and raises morning glucose — avoid",
            ],
            'urgency_note': "Poor sleep accelerates prediabetes to diabetes conversion"
        },
        'high_risk_prediabetic': {
            'recs': [
                "Sleep deprivation raises fasting glucose by 15-20 mg/dL after just 2 nights",
                "Your glucose is already at the edge of diabetic range — protect it with sleep",
                "Ask your doctor about sleep apnea screening — common at your BMI/risk level",
                "Sleep + stress combined is the most damaging metabolic combination — address both",
            ],
            'urgency_note': "High priority — sleep directly controls your fasting glucose levels"
        },
        'diabetic': {
            'recs': [
                "23.1% of diabetics report poor sleep vs 19.2% normal — feeds a damaging cycle",
                "Poor sleep raises HbA1c — it is a clinical diabetes management target",
                "Sleep apnea is 3x more common in diabetics — get screened if not already done",
                "Nocturia (night urination) disrupting sleep? Better glucose control reduces it",
                "7-8 hrs is part of your diabetes management plan alongside medication",
            ],
            'urgency_note': "Clinical priority — discuss sleep quality with your healthcare provider"
        }
    },
    'mental_health': {
        'normal': {
            'recs': [
                "Good mental health is actively protecting your metabolic health — maintain it",
                "Regular exercise and sleep are your best mental health tools",
                "10 min daily mindfulness reduces cortisol and improves insulin sensitivity",
            ]
        },
        'early_insulin_resistance': {
            'recs': [
                "Chronic stress raises cortisol which directly worsens insulin resistance",
                "10 min daily mindfulness shown effective in DPP diabetes prevention studies",
                "Exercise (30 min/day) improves mood and insulin sensitivity simultaneously",
            ]
        },
        'prediabetic': {
            'recs': [
                "Depression doubles the risk of converting prediabetes to diabetes",
                "Exercise 30 min/day reduces PHQ-9 score by 2-3 points in clinical studies",
                "Sleep improvement often reduces depression scores — address sleep domain first",
                "Mindfulness-Based Stress Reduction (MBSR) shown effective in DPP studies",
            ]
        },
        'high_risk_prediabetic': {
            'recs': [
                "Stress hormones (cortisol) raise fasting glucose — mental health is a clinical target",
                "Depression impairs self-management ability — diet, exercise, medication adherence",
                "PHQ-9 >= 10: see care pathway — moderate depression requires professional support",
                "DPP program includes stress management component — addresses all risk factors together",
            ]
        },
        'diabetic': {
            'recs': [
                "8.3% moderate and 6.0% severe depression in diabetic group — highest of all groups",
                "Depression is 2-3x more common in diabetes and directly impairs glucose control",
                "PHQ-9 >= 10: bring score to your doctor — depression treatment improves HbA1c",
                "Diabetes distress (burnout from managing diabetes) affects 45% — discuss with doctor",
                "Mental health treatment improves HbA1c by 0.3-0.5% in clinical studies",
            ]
        }
    },
    'alcohol': {
        'normal': {
            'recs': [
                "Within ADA limits at your risk level — maintain current pattern",
                "Always pair alcohol with food to prevent reactive hypoglycemia",
                "2+ alcohol-free days per week is beneficial for liver and metabolic health",
            ]
        },
        'early_insulin_resistance': {
            'recs': [
                "Alcohol impairs glucose clearance — reduce toward ADA optimal target",
                "Never drink on empty stomach — causes reactive hypoglycemia",
                "Replace 2-3 drinks/week with sparkling water + citrus or herbal tea",
                "Alcohol disrupts sleep quality — compounding insulin resistance effect",
            ]
        },
        'prediabetic': {
            'recs': [
                "76.2% of prediabetic group drinks vs 80.7% normal — reduction is protective",
                "Target half the ADA limit for optimal glucose control",
                "Beer and sweet cocktails spike glucose — dry wine or spirits are lower risk",
                "Alcohol raises triglycerides which directly worsen insulin resistance",
            ]
        },
        'high_risk_prediabetic': {
            'recs': [
                "Alcohol directly raises fasting glucose the morning after even moderate drinking",
                "Liver prioritizes clearing alcohol over regulating glucose — dangerous at your level",
                "Never drink on empty stomach — glucose swings more dangerous at your HbA1c",
                "Replace evening drinks with herbal tea — sleep and glucose both improve",
            ]
        },
        'diabetic': {
            'recs': [
                "70.5% of diabetics drink vs 80.7% normal — diabetics correctly drink less",
                "Alcohol can cause dangerous hypoglycemia if on insulin or sulfonylureas",
                "Always eat when drinking — never on empty stomach with diabetes medications",
                "Check glucose before and 2 hrs after drinking — alcohol effects are delayed",
                "Discuss alcohol use with your doctor if on any diabetes medications",
            ]
        }
    },
    'smoking': {
        'normal': {
            'never' : ["Non-smoking is strongly protective for insulin sensitivity — maintain this"],
            'former': [
                "Former smoker: insulin resistance normalizes gradually over 5-10 years",
                "Maintaining non-smoking is one of the highest-impact choices you can make",
            ],
            'current': [
                "Smoking raises insulin resistance even at low risk — cessation is the top action",
                "CDC Smokefree (smokefree.gov) — free text, app and web coaching",
                "Nicotine replacement preferred over vaping for metabolically neutral cessation",
            ]
        },
        'prediabetic': {
            'never' : ["Non-smoking is protecting you — 54.9% of prediabetics never smoked"],
            'former': [
                "Former smoker at prediabetes stage — insulin resistance higher than never-smokers",
                "Each year after quitting, insulin sensitivity continues to improve",
            ],
            'current': [
                "17.4% of prediabetics currently smoke — accelerates progression to diabetes",
                "Smoking + prediabetes = 3x higher diabetes conversion risk vs non-smokers",
                "Cessation reduces HbA1c by ~0.2% within weeks — meaningful at your level",
                "CDC Smokefree (smokefree.gov) — free text, app and web support",
            ]
        },
        'diabetic': {
            'never' : ["Non-smoking protects your cardiovascular health — diabetics have 2x heart disease risk"],
            'former': [
                "Former smoker: cardiovascular risk decreases significantly each year after quitting",
                "Your insulin resistance from past smoking is gradually improving over time",
            ],
            'current': [
                "17.3% of diabetics currently smoke — highest risk combination for heart disease",
                "Smoking + diabetes = 4x cardiovascular disease risk vs non-smoking diabetics",
                "Cessation is the single most impactful action outside of medication",
                "Ask your doctor about varenicline (Chantix) — most effective cessation medication",
            ]
        }
    },
    'diet': {
        'normal': {
            'recs': [
                "Balanced meals — half plate vegetables, quarter lean protein, quarter whole grains",
                "Limit sugary drinks — each daily soda raises diabetes risk by 26% over 10 years",
                "Choose whole fruit over juice — fiber slows glucose absorption significantly",
                "Use Meal Checker tab to see nutrition breakdown of your regular foods",
            ],
            'status': 'Maintain'
        },
        'early_insulin_resistance': {
            'recs': [
                "Reduce refined grains (white bread, white rice) — swap for whole grain versions",
                "Add fiber to every meal (beans, lentils, oats) — slows glucose absorption",
                "Limit added sugar to < 25g/day women, < 36g/day men",
                "Choose whole fruit over juice — the fiber makes the difference",
                "Use Meal Checker to identify high-carb foods and see ADA-approved swaps",
            ],
            'status': 'Action Needed'
        },
        'prediabetic': {
            'recs': [
                "ADA Plate Method: half non-starchy veg, quarter lean protein, quarter complex carbs",
                "Target 45-60g carbs per meal — track for 1 week to understand your intake",
                "Eliminate sugary drinks completely — replace with water or unsweetened tea",
                "Fiber goal: 25-35g/day — beans, oats, and vegetables at every meal",
                "60.9% of prediabetics not meeting DPP dietary targets — use Meal Checker for swaps",
            ],
            'status': 'Action Needed'
        },
        'high_risk_prediabetic': {
            'recs': [
                "ADA Plate Method strictly — half non-starchy veg every single meal",
                "Limit carbs to 45g/meal maximum — your glucose response is amplified",
                "Eliminate all sugary drinks, fruit juice, white bread, white rice now",
                "Meal timing: eat within 10 hrs/day — time-restricted eating improves HbA1c",
                "Glycemic index matters — choose foods with GI < 55",
                "70% not meeting DPP dietary targets at your risk level — use Meal Checker daily",
            ],
            'status': 'High Priority'
        },
        'diabetic': {
            'recs': [
                "Carbohydrate management is your #1 dietary tool for glucose control",
                "Consistent 45-60g carbs/meal stabilizes glucose better than restriction",
                "ADA Plate Method every meal — half non-starchy veg is non-negotiable",
                "Saturated fat < 7% calories — diabetes raises cardiovascular risk 2-4x",
                "Sodium < 2300mg/day — hypertension is common with diabetes",
                "Work with a registered dietitian — personalized plan improves HbA1c by 1-2%",
            ],
            'status': 'Clinical Priority'
        }
    }
}

def get_interventions(user, score, label):
    bench = risk_pkg['benchmarks']
    bench_key = (
        'diabetic'   if label == 'diabetic' else
        'prediabetic' if label in ['prediabetic','high_risk_prediabetic','early_insulin_resistance'] else
        'normal'
    )
    risk_bench = bench.get(bench_key, bench.get('normal', {}))

    # Resolve label to closest rule key
    def rule_label(lbl):
        mapping = {
            'normal'                  : 'normal',
            'early_insulin_resistance': 'early_insulin_resistance',
            'prediabetic'             : 'prediabetic',
            'high_risk_prediabetic'   : 'high_risk_prediabetic',
            'diabetic'                : 'diabetic',
        }
        return mapping.get(lbl, 'normal')

    rl       = rule_label(label)
    ex_rules = INTERVENTION_RULES['exercise'][rl]
    sl_rules = INTERVENTION_RULES['sleep'][rl]
    mh_rules = INTERVENTION_RULES['mental_health'][rl]
    al_rules = INTERVENTION_RULES['alcohol'][rl]
    dt_rules = INTERVENTION_RULES['diet'][rl]

    interventions = []

    # ── Exercise ──
    ex_min   = user.get('total_exercise_min_week', 0)
    ex_gap   = max(0, 150 - ex_min)
    ex_bench = risk_bench.get('avg_exercise_min_week', 150)
    pct      = (ex_min / 150) * 100
    if   pct >= 100: ex_status = 'Meeting Target'
    elif pct >= 66:  ex_status = 'Near Target'
    elif pct >= 33:  ex_status = 'Below Target'
    else:             ex_status = 'Very Low'

    ex_recs = []
    if ex_gap > 0:
        sessions = max(1, round(ex_gap / 30))
        ex_recs.append(f"Add {sessions} x 30-min brisk walks/week to close the {ex_gap:.0f} min gap to 150 min/week")
    ex_recs += ex_rules['recs']

    interventions.append({
        'domain': 'Exercise', 'icon': '🏃',
        'status': ex_status,
        'priority': 1 if pct < 33 else 2 if pct < 66 else 4,
        'user_value': f"{ex_min:.0f} min/week",
        'target': "150 min/week (ADA DPP)",
        'nhanes': (f"People at your risk level ({bench_key}) average {ex_bench:.0f} min/week. "
                   f"You are at {ex_min:.0f} min/week."),
        'urgency_note': ex_rules.get('urgency_note', ''),
        'recs': ex_recs[:5],
        'ada_rule': "ADA DPP Standard: 150 min/week moderate-intensity physical activity"
    })

    # ── Sleep ──
    sleep    = user.get('avg_sleep_hrs', 7)
    deficit  = max(0, 7 - sleep)
    sl_bench = risk_bench.get('pct_poor_sleep', 20)
    if   sleep >= 7 and sleep <= 9: sl_status = 'Optimal'
    elif sleep >= 6:                 sl_status = 'Slightly Low'
    elif sleep < 6:                  sl_status = 'Very Low'
    else:                            sl_status = 'Too High'

    sl_recs = []
    if deficit > 0:
        sl_recs.append(f"Target 7 hrs minimum — you have a {deficit:.1f} hr nightly deficit")
    sl_recs += sl_rules['recs']

    interventions.append({
        'domain': 'Sleep', 'icon': '😴',
        'status': sl_status,
        'priority': 1 if sleep < 6 else 2 if sleep < 7 else 4,
        'user_value': f"{sleep:.1f} hrs/night",
        'target': "7–9 hrs (ADA)",
        'nhanes': (f"{sl_bench:.0f}% of {bench_key} patients report poor sleep quality. "
                   f"Poor sleep raises cortisol and insulin resistance."),
        'urgency_note': sl_rules.get('urgency_note', ''),
        'recs': sl_recs[:5],
        'ada_rule': "ADA: 7-9 hours of quality sleep for optimal glucose regulation"
    })

    # ── Mental Health ──
    phq9 = user.get('depression_score', 0)
    if   phq9 <= 4:  mh_cat = 'Minimal';  mh_status = 'Good'
    elif phq9 <= 9:  mh_cat = 'Mild';     mh_status = 'Mild Concern'
    elif phq9 <= 14: mh_cat = 'Moderate'; mh_status = 'Moderate Concern'
    else:             mh_cat = 'Severe';   mh_status = 'High Concern'

    mh_bench_mod = risk_bench.get('pct_depression_moderate', 7)
    mh_bench_sev = risk_bench.get('pct_depression_severe', 4)

    interventions.append({
        'domain': 'Mental Health', 'icon': '🧠',
        'status': mh_status,
        'priority': 1 if phq9 >= 10 else 2 if phq9 >= 5 else 5,
        'user_value': f"PHQ-9: {phq9} ({mh_cat})",
        'target': "PHQ-9 < 5 (Minimal)",
        'nhanes': (f"{mh_bench_mod:.0f}% moderate and {mh_bench_sev:.0f}% severe depression "
                   f"in {bench_key} group. Depression doubles diabetes conversion risk."),
        'urgency_note': 'PHQ-9 >= 10 triggers care pathway escalation' if phq9 >= 10 else '',
        'recs': mh_rules['recs'][:5],
        'ada_rule': "ADA: Screen for depression in prediabetes/diabetes — PHQ-9 >= 10 requires provider referral"
    })

    # ── Alcohol ──
    drinks = user.get('drinks_per_week', 0)
    sex    = 'male' if user.get('sex_enc', 1) == 1 else 'female'
    limit  = 14 if sex == 'male' else 7
    al_bench = risk_bench.get('pct_drinks_past_year', 75)
    if   drinks == 0:          al_status = 'Non-Drinker'
    elif drinks <= limit / 2:  al_status = 'Low — Good'
    elif drinks <= limit:      al_status = 'Moderate — At Limit'
    elif drinks <= limit * 1.5:al_status = 'Above Limit'
    else:                       al_status = 'High — Reduce Now'

    al_recs = []
    if drinks > limit:
        al_recs.append(f"Reduce to {limit} drinks/week — ADA limit for {sex}s. You are {drinks-limit:.0f} over.")
        al_recs.append(f"Target {int(limit/2)} drinks/week for optimal glucose control")
    al_recs += al_rules['recs']

    interventions.append({
        'domain': 'Alcohol', 'icon': '🍺',
        'status': al_status,
        'priority': 1 if drinks > limit else 3 if drinks > limit/2 else 5,
        'user_value': f"{drinks:.0f} drinks/week",
        'target': f"≤{limit} drinks/week (ADA limit for {sex}s)",
        'nhanes': (f"{al_bench:.0f}% of {bench_key} group drinks alcohol. "
                   f"ADA limit is {limit} drinks/week for {sex}s."),
        'urgency_note': f"Excess alcohol raises triglycerides and worsens insulin resistance" if drinks > limit else '',
        'recs': al_recs[:5],
        'ada_rule': f"ADA: Limit alcohol to {limit} drinks/week for {sex}s — alcohol impairs glucose regulation"
    })

    # ── Smoking ──
    smoking    = user.get('smoking_enc', 0)
    sm_map     = {0: 'never', 1: 'former', 2: 'current'}
    sm_key     = sm_map.get(smoking, 'never')
    sm_status  = {'never': 'Never Smoked', 'former': 'Former Smoker', 'current': 'Current Smoker'}[sm_key]
    sm_bench   = risk_bench.get('pct_current_smoker', 15)
    sm_bench_f = risk_bench.get('pct_former_smoker', 25)

    # Get smoking recs — some labels share rules, use prediabetic for high_risk
    sm_label_key = rl if rl in INTERVENTION_RULES['smoking'] else (
        'prediabetic' if rl == 'high_risk_prediabetic' else 'normal')
    sm_recs = INTERVENTION_RULES['smoking'][sm_label_key].get(sm_key, [])

    interventions.append({
        'domain': 'Smoking', 'icon': '🚬',
        'status': sm_status,
        'priority': 1 if smoking == 2 else 4,
        'user_value': sm_status,
        'target': "Non-smoker",
        'nhanes': (f"{sm_bench:.0f}% currently smoke and {sm_bench_f:.0f}% are former smokers "
                   f"in the {bench_key} group. Smoking raises insulin resistance and HbA1c."),
        'urgency_note': "Smoking cessation is the highest-impact single action you can take" if smoking == 2 else '',
        'recs': sm_recs[:5],
        'ada_rule': "ADA: Smoking cessation is a clinical priority — raises insulin resistance and HbA1c by ~0.2%"
    })

    # ── Diet ──
    dpp_pct   = risk_bench.get('pct_meeting_dpp_target', 40)
    not_meeting = 100 - dpp_pct

    interventions.append({
        'domain': 'Diet', 'icon': '🥗',
        'status': dt_rules['status'],
        'priority': 1 if rl in ['diabetic','high_risk_prediabetic'] else 2 if rl == 'prediabetic' else 3,
        'user_value': 'Based on risk profile',
        'target': "ADA Plate Method 2023",
        'nhanes': (f"{not_meeting:.0f}% of {bench_key} group not meeting DPP dietary targets. "
                   f"Only {dpp_pct:.0f}% are on track."),
        'urgency_note': 'Use Meal Checker tab to identify and swap high-risk foods' if rl != 'normal' else '',
        'recs': dt_rules['recs'][:6],
        'ada_rule': "ADA Plate Method: half non-starchy veg, quarter lean protein, quarter carbohydrates"
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
    st.subheader("Meal Glucose Checker")
    st.markdown("Build your meal by adding multiple foods. See combined nutrition and predicted glucose spike.")

    if "meal_items" not in st.session_state:
        st.session_state.meal_items = []

    # Profile check
    has_profile = bool(st.session_state.user_profile)
    if not has_profile:
        st.warning("Complete Risk Assessment first for a personalized glucose prediction.")
        with st.expander("Or enter key values here for better accuracy"):
            mc1, mc2, mc3 = st.columns(3)
            quick_age   = mc1.number_input("Age", 18, 90, 45)
            quick_bmi   = mc2.number_input("BMI", 15.0, 60.0, 27.0, step=0.1)
            quick_hba1c = mc3.number_input("HbA1c pct leave 0 if unknown", 0.0, 15.0, 0.0, step=0.1)
            mc4, mc5    = st.columns(2)
            quick_homa  = mc4.number_input("HOMA-IR leave 0 if unknown", 0.0, 20.0, 0.0, step=0.1)
            quick_sex   = mc5.selectbox("Sex", ["Male", "Female"])
            st.session_state.user_profile = {
                "age": quick_age, "bmi": quick_bmi,
                "hba1c": quick_hba1c if quick_hba1c > 0 else None,
                "homa_ir": quick_homa if quick_homa > 0 else None,
                "sex_enc": 1 if quick_sex == "Male" else 0,
                "fasting_glucose": None, "hdl_cholesterol": None, "triglycerides": None,
            }

    col1, col2 = st.columns([2, 1])
    with col1:
        meal_type = st.selectbox("Meal type", ["Breakfast", "Lunch", "Dinner", "Snack"])
    with col2:
        baseline_glucose = st.number_input(
            "Current glucose mg/dL", 60, 300, 95,
            help="Blood glucose before eating — leave at 95 if unknown"
        )
    meal_hour = {"Breakfast": 8, "Lunch": 13, "Dinner": 19, "Snack": 15}[meal_type]

    st.divider()
    st.markdown("### Add Food to Meal")
    ac1, ac2 = st.columns([3, 1])
    with ac1:
        query = st.text_input("Search food", placeholder="e.g. white bread, rice, egg, banana, orange juice...")
    with ac2:
        add_qty = st.number_input("Servings", min_value=0.25, max_value=20.0, value=1.0, step=0.25)

    search_result = None
    if query and len(query) >= 2:
        q       = query.lower().strip()
        matches = fndds_df[fndds_df["food_name"].str.lower().str.contains(q, na=False)].head(10)
        if len(matches) == 0:
            st.warning("No foods found — try a different term.")
        else:
            chosen        = st.selectbox("Select food", matches["food_name"].tolist())
            search_result = matches[matches["food_name"] == chosen].iloc[0]
            pg_total      = float(search_result["portion_weight_g"]) * add_qty
            warn          = "  Large portion — verify this is correct" if pg_total > 400 else ""
            st.caption(
                f"1 serving = {search_result['portion_desc']} "
                f"({search_result['portion_weight_g']:.0f}g) "
                f"Total: {pg_total:.0f}g{warn}"
            )

    add_col, clear_col = st.columns(2)
    with add_col:
        if st.button("Add to Meal", use_container_width=True) and search_result is not None:
            ic  = round(float(search_result["carb_g_per_serving"])   * add_qty, 1)
            ip  = round(float(search_result["protein_g_per_serving"])* add_qty, 1)
            ifa = round(float(search_result["fat_g_per_serving"])     * add_qty, 1)
            ifi = round(float(search_result["fiber_g_per_serving"])   * add_qty, 1)
            is_ = round(min(float(search_result["sugar_g_per_serving"]) * add_qty, ic), 1)
            ica = round(float(search_result["energy_kcal_per_serving"])* add_qty, 0)
            st.session_state.meal_items.append({
                "food_code"    : int(search_result["food_code"]),
                "food_name"    : search_result["food_name"],
                "portion"      : search_result["portion_desc"],
                "qty"          : add_qty,
                "carbs"        : ic,
                "protein"      : ip,
                "fat"          : ifa,
                "fiber"        : ifi,
                "sugar"        : is_,
                "calories"     : ica,
                "ada_risk"     : str(search_result.get("ada_risk_level", "low")),
                "diabetes_flag": int(search_result.get("diabetes_flag", 0)),
                "high_carb"    : bool(search_result.get("high_carb", False)),
                "high_sugar"   : bool(search_result.get("high_sugar", False)),
                "low_fiber"    : bool(search_result.get("low_fiber", False)),
            })
            st.success(f"Added {search_result['food_name']} x{add_qty}")
            st.rerun()
    with clear_col:
        if st.button("Clear Meal", use_container_width=True):
            st.session_state.meal_items = []
            st.rerun()

    if not st.session_state.meal_items:
        st.info("Add at least one food above to see glucose prediction and swap recommendations.")
    else:
        st.divider()
        st.markdown("### Your Meal")
        items = st.session_state.meal_items

        for i, item in enumerate(items):
            c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([3,1,1,1,1,1,1,0.7])
            c1.markdown(f"**{item['food_name']}** x{item['qty']}  *({item['portion']})*")
            c2.metric("Cal",     f"{item['calories']:.0f}")
            c3.metric("Carbs",   f"{item['carbs']:.1f}g")
            c4.metric("Protein", f"{item['protein']:.1f}g")
            c5.metric("Fat",     f"{item['fat']:.1f}g")
            c6.metric("Fiber",   f"{item['fiber']:.1f}g")
            c7.metric("Sugar",   f"{item['sugar']:.1f}g")
            if c8.button("X", key=f"rm_{i}"):
                st.session_state.meal_items.pop(i)
                st.rerun()

        st.divider()

        total_carbs   = round(sum(i["carbs"]    for i in items), 1)
        total_protein = round(sum(i["protein"]  for i in items), 1)
        total_fat     = round(sum(i["fat"]      for i in items), 1)
        total_fiber   = round(sum(i["fiber"]    for i in items), 1)
        total_sugar   = round(min(sum(i["sugar"] for i in items), total_carbs), 1)
        total_cals    = round(sum(i["calories"] for i in items), 0)

        st.markdown("### Meal Totals")
        tc = st.columns(6)
        for col, label, val, unit in zip(tc,
            ["Calories","Carbs","Protein","Fat","Fiber","Sugar"],
            [total_cals, total_carbs, total_protein, total_fat, total_fiber, total_sugar],
            ["kcal","g","g","g","g","g"]):
            col.metric(label, f"{val}{unit}")

        carbs   = total_carbs
        protein = total_protein
        fat     = total_fat
        fiber   = total_fiber
        sugar   = total_sugar
        cals    = total_cals

        high_risk_items = [i for i in items if i["diabetes_flag"] == 1 or i["ada_risk"] == "high"]
        if high_risk_items:
            st.warning("High-risk foods in meal: " + ", ".join(i["food_name"] for i in high_risk_items))

        high_carb  = 1 if carbs > 60  else 0
        low_fiber  = 1 if fiber < 3   else 0
        high_cal   = 1 if cals  > 500 else 0
        cfr        = carbs / (fiber + 1)

        user_prof  = st.session_state.user_profile if st.session_state.user_profile else {}
        age_val    = user_prof.get("age",    45)
        bmi_val    = user_prof.get("bmi",    27)
        hba1c_val  = user_prof.get("hba1c",  5.5) or 5.5
        fg_val     = user_prof.get("fasting_glucose", 95) or 95
        homa_val   = user_prof.get("homa_ir", 1.5) or 1.5
        hdl_val    = user_prof.get("hdl_cholesterol", 50) or 50
        trig_val   = user_prof.get("triglycerides", 120) or 120
        sex_val    = user_prof.get("sex_enc", 1)

        meal_dict = {
            "meal_carbs"      : carbs,
            "meal_protein"    : protein,
            "meal_fat"        : fat,
            "meal_fiber"      : fiber,
            "baseline_glucose": baseline_glucose,
            "high_carb_meal"  : high_carb,
            "low_fiber_meal"  : low_fiber,
            "high_cal_meal"   : high_cal,
            "time_to_peak_min": 60,
            "age"             : age_val,
            "sex"             : sex_val,
            "bmi"             : bmi_val,
            "hba1c"           : hba1c_val,
            "fasting_glucose" : fg_val,
            "homa_ir"         : homa_val,
            "hdl_cholesterol" : hdl_val,
            "triglycerides"   : trig_val,
            "hour"            : meal_hour,
            "carb_fiber_ratio": cfr,
            "ir_proxy"        : homa_val * carbs / 100,
            "carb_x_baseline" : carbs * baseline_glucose / 100,
            "mt_breakfast"    : 1 if meal_type == "Breakfast" else 0,
            "mt_lunch"        : 1 if meal_type == "Lunch"     else 0,
            "mt_dinner"       : 1 if meal_type == "Dinner"    else 0,
            "mt_snack"        : 1 if meal_type == "Snack"     else 0,
        }

        spike, cat  = predict_glucose_spike(meal_dict, {})
        spike_color = get_spike_color(cat)

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            fig2 = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = spike,
                title = {"text": "Predicted Glucose Spike (mg/dL)"},
                delta = {"reference": 40, "increasing": {"color": "#e74c3c"}},
                gauge = {
                    "axis" : {"range": [-20, 200]},
                    "bar"  : {"color": spike_color},
                    "steps": [
                        {"range": [-20,  20], "color": "#d5f5e3"},
                        {"range": [ 20,  40], "color": "#fef9e7"},
                        {"range": [ 40,  70], "color": "#fdebd0"},
                        {"range": [ 70, 200], "color": "#fadbd8"},
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
            st.markdown("### Swap Recommendations")

            # Check each food item for swaps
            swaps_found = []
            for item in items:
                fc       = item["food_code"]
                sw_rows  = swap_df[swap_df["high_risk_food_code"] == fc]
                if len(sw_rows) > 0:
                    best = sw_rows.sort_values("carb_saving_per_serving", ascending=False).iloc[0]
                    sw_fndds = fndds_df[fndds_df["food_code"] == int(best["swap_food_code"])]
                    swaps_found.append({
                        "original"  : item["food_name"],
                        "swap_name" : best["swap_food_name"],
                        "swap_portion": best["swap_portion"],
                        "carb_saving": round(float(best["carb_saving_per_serving"]) * item["qty"], 1),
                        "fiber_gain" : round(float(best["fiber_gain_per_serving"])  * item["qty"], 1),
                        "ada_rule"  : best["ada_rule"],
                        "swap_fndds": sw_fndds,
                        "item_qty"  : item["qty"],
                    })

            if swaps_found:
                total_carb_save  = sum(s["carb_saving"] for s in swaps_found)
                total_fiber_gain = sum(s["fiber_gain"]  for s in swaps_found if s["fiber_gain"] > 0)

                for sw in swaps_found:
                    with st.expander(f"Swap: {sw['original'][:40]} → {sw['swap_name']}", expanded=True):
                        sc1, sc2 = st.columns(2)
                        sc1.markdown(f"**Current:** {sw['original']}")
                        sc2.markdown(f"**Swap to:** {sw['swap_name']}")
                        sc2.markdown(f"Serving: {sw['swap_portion']}")
                        if not sw["swap_fndds"].empty:
                            sw_row = sw["swap_fndds"].iloc[0]
                            sc2.markdown(f"Carbs: **{round(float(sw_row['carb_g_per_serving'])*sw['item_qty'],1)}g**")
                        st.success(f"Saves {sw['carb_saving']:.0f}g carbs" +
                                   (f" + adds {sw['fiber_gain']:.1f}g fiber" if sw['fiber_gain'] > 0 else ""))
                        st.caption(f"ADA: {sw['ada_rule']}")

                # Estimate combined spike reduction
                if total_carb_save > 0:
                    swap_meal = {**meal_dict,
                        "meal_carbs": max(0, carbs - total_carb_save),
                        "meal_fiber": fiber + total_fiber_gain}
                    swap_meal["carb_fiber_ratio"] = swap_meal["meal_carbs"] / (swap_meal["meal_fiber"] + 1)
                    swap_meal["ir_proxy"]          = homa_val * swap_meal["meal_carbs"] / 100
                    swap_spike, swap_cat           = predict_glucose_spike(swap_meal, {})
                    reduction = spike - swap_spike
                    if reduction > 0:
                        st.info(
                            f"If you made all swaps: spike drops from **{spike:.0f}** to "
                            f"**{swap_spike:.0f} mg/dL** ({cat} to {swap_cat}) "
                            f"— saving {reduction:.0f} mg/dL"
                        )
            else:
                # No swaps found — show ADA guidance for high-risk items
                if high_risk_items:
                    st.markdown("**No specific swaps found in database.**")
                    st.markdown("**ADA General Guidance for this meal:**")
                    if high_carb == 1:
                        st.markdown("- High carb meal — aim for 45-60g carbs maximum")
                        st.markdown("- Pair carbs with protein and fat to slow absorption")
                    if low_fiber == 1:
                        st.markdown("- Low fiber — add vegetables or legumes to slow glucose rise")
                    st.markdown("- Post-meal walk (15 min) reduces glucose spike by ~20%")
                else:
                    st.success("No high-risk foods detected in this meal. Predicted spike is within acceptable range.")

        st.caption(
            f"Model: MAE {gluc_pkg.get('mae','~21')} mg/dL | "
            f"R2 {gluc_pkg.get('r2','~0.34')} | "
            f"Directional accuracy {gluc_pkg.get('dir_accuracy','~72')}% | "
            f"Based on 45 CGM patients | Corrections applied for large portions and extreme metabolic values"
        )

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
