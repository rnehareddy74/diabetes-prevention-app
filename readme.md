# Early Diabetes Prevention Platform

A machine learning powered diabetes prevention application built on NHANES population data, ADA 2023 clinical guidelines, USDA FNDDS nutritional data, and a CGM patient cohort. The platform provides personalized risk assessment, lifestyle intervention recommendations, and meal-level glucose spike prediction with food swap guidance.

---

## Project Structure

    diabetes-prevention-app/
        app.py
        requirements.txt
        README.md
        models/
            risk_models.pkl
            glucose_models.pkl
            intervention_engine.pkl
        data/
            nhanes_risk_model__1_.csv
            fndds_nutrition_lookup.csv
            food_swap_database_fndds.csv
            nhanes_lifestyle_benchmarks.csv
            cgmacros_meal_response.csv
            cgmacros_activity_response.csv
        Notebooks/
            step1_risk_models_final.ipynb
            step2_glucose_model_v3_final.ipynb
            step3_intervention_engine.ipynb

---

## Datasets

### 1. NHANES Risk Model Dataset

File: nhanes_risk_model__1_.csv
Source: National Health and Nutrition Examination Survey (CDC)
Shape: 6,330 rows, 49 columns
Purpose: Training data for Model A (risk score) and Model B (clinical label)

Label distribution:
    normal: 2,539 patients
    early_insulin_resistance: 1,152 patients
    prediabetic: 1,075 patients
    high_risk_prediabetic: 514 patients
    diabetic: 715 patients

Binary target for Model A (risk_group):
    0 normal: 3,691
    1 prediabetic: 1,589
    2 diabetic: 715

Key feature columns: age, sex, ethnicity, bmi, waist_cm, height_cm, weight_kg, waist_height_ratio, hba1c, fasting_glucose, insulin, homa_ir, triglycerides, hdl_cholesterol, ldl_cholesterol, total_cholesterol, total_exercise_min_week, avg_sleep_hrs, sedentary_min_day, smoking_status_clean, drinks_per_week, depression_score, depression_category, diabetes_diagnosed, told_prediabetes_before

Missing data challenges:
    fasting_glucose: 47.5% missing (lab test not administered to all)
    insulin and homa_ir: approximately 50% missing
    total_exercise_min_week: 22.8% missing
    avg_sleep_hrs: 14.7% missing
    drinks_per_week: 50.3% missing
    depression_score: 13.9% missing
    hba1c: 5.3% missing

XGBoost handles missing values natively without imputation for the final models.

---

### 2. USDA FNDDS Nutrition Lookup

File: fndds_nutrition_lookup.csv
Source: USDA Food and Nutrient Database for Dietary Studies 2021-2023
Shape: 5,430 rows, 28 columns
Purpose: Food search index and nutritional lookup for the Meal Checker

Columns: food_code, food_name, wweia_cat_num, wweia_cat_name, energy_kcal, carb_g, fiber_g, sugar_g, fat_g, sat_fat_g, protein_g, cholesterol_mg, portion_desc, portion_weight_g, energy_kcal_per_serving, carb_g_per_serving, fiber_g_per_serving, sugar_g_per_serving, fat_g_per_serving, sat_fat_g_per_serving, protein_g_per_serving, high_carb, high_sugar, low_fiber, high_sat_fat, diabetes_flag, ada_risk_level, source

Risk classification:
    556 high-risk foods (ada_risk_level: high)
    1,471 medium-risk foods
    3,403 low-risk foods
    1,715 foods with diabetes_flag = 1

Food categories: 171 WWEIA categories covering all major food groups

Data quality note: Some foods show sugar_g_per_serving slightly exceeding carb_g_per_serving due to how FNDDS handles lactose in dairy products. The app caps displayed sugar at total carbohydrates to prevent confusing output.

---

### 3. ADA Food Swap Database (FNDDS-linked)

File: food_swap_database_fndds.csv
Source: ADA Diabetes Plate Method 2023 mapped to FNDDS food codes
Shape: 1,290 rows, 21 columns
Purpose: Food swap recommendations in the Meal Checker

Columns: high_risk_food_code, high_risk_food_name, high_risk_category, high_risk_carb_per100g, high_risk_fiber_per100g, high_risk_portion, high_risk_portion_g, high_risk_carb_per_serving, swap_food_code, swap_food_name, swap_category, swap_carb_per100g, swap_fiber_per100g, swap_portion, swap_portion_g, swap_carb_per_serving, carb_saving_per_serving, fiber_gain_per_serving, ada_rule, swap_group, source

Coverage:
    558 unique high-risk food codes covered
    12 unique swap food names (by ADA Plate Method design)
    1,290 high-risk to swap pairings

Swap groups:
    sweet_foods: 860 pairings (cakes, cookies, candy, pastries, jams)
    refined_grains: 379 pairings (white bread, rolls, tortillas, pancakes)
    starchy_veg: 36 pairings (potatoes and starchy vegetables)
    beverages: 15 pairings (sugary drinks)

High-risk food categories:
    Cakes and pies: 258
    Cookies and brownies: 238
    Yeast breads: 215
    Doughnuts, sweet rolls, pastries: 175
    Jams, syrups, toppings: 89
    Rolls and buns: 76
    Pancakes, waffles, French toast: 70
    Candy containing chocolate: 51
    Candy not containing chocolate: 49
    Other starchy vegetables: 36

The four ADA rules embedded in the database:
    Rule 1: Replace refined grains with whole grains or non-starchy vegetables
    Rule 2: Eliminate added sugars and choose whole fruits instead
    Rule 3: Choose water or unsweetened tea and coffee over sugary drinks
    Rule 4: Choose non-starchy vegetables over starchy vegetables

Data quality notes:
    3 swap entries had negative fiber_gain_per_serving values (high-fiber bread swapped to lower-fiber alternative). These were set to zero for display since the swap is still valid for carbohydrate reduction.
    88.4% of high-risk FNDDS foods have no specific swap entry. For these the app shows ADA category-level guidance based on the food's high_carb, high_sugar, and low_fiber flags.

---

### 4. NHANES Lifestyle Benchmarks

File: nhanes_lifestyle_benchmarks.csv
Source: Derived from NHANES dataset aggregated by risk group
Shape: 3 rows, 22 columns
Purpose: Population comparison values used in the intervention engine

Groups: normal (3,691 patients), prediabetic (1,589 patients), diabetic (715 patients)

Columns used for differentiated comparisons:
    avg_exercise_min_week: normal 180, prediabetic 150, diabetic 120
    pct_poor_sleep: normal 19.2%, prediabetic 22.8%, diabetic 23.1%
    pct_never_smoked: normal 64.4%, prediabetic 54.9%, diabetic 49.4%
    pct_former_smoker: normal 23.1%, prediabetic 27.5%, diabetic 33.1%
    pct_current_smoker: normal 12.3%, prediabetic 17.4%, diabetic 17.3%
    pct_drinks_past_year: normal 80.7%, prediabetic 76.2%, diabetic 70.5%
    pct_depression_moderate: normal 7.3%, prediabetic 6.5%, diabetic 8.3%
    pct_depression_severe: normal 4.3%, prediabetic 3.1%, diabetic 6.0%
    pct_meeting_dpp_target: normal 49.5%, prediabetic 39.1%, diabetic 30.1%

Columns with identical values across all groups (used as population estimates only, not for individual comparison):
    avg_sleep_hrs: 8.0 for all groups
    avg_drinks_per_week: 0.19 for all groups
    pct_inactive: 0.0 for all groups
    pct_heavy_drinkers: 0.0 for all groups
    avg_depression_score: 3.0 for normal and diabetic, 2.0 for prediabetic

---

### 5. CGM Meal Response Dataset

File: cgmacros_meal_response.csv
Source: Continuous glucose monitor cohort with meal tracking
Shape: 1,640 rows, 31 columns
Purpose: Training data for Model C (glucose spike predictor)
Patients: 45
Meals per patient: mean 36.4, range 10 to 89

Columns: patient_id, timestamp, meal_type, meal_calories, meal_carbs, meal_protein, meal_fat, meal_fiber, net_carbs, baseline_glucose, peak_glucose, glucose_spike, time_to_peak_min, recovery_time_min, glucose_auc, high_carb_meal, low_fiber_meal, high_cal_meal, age, sex, bmi, hba1c, fasting_glucose, insulin, homa_ir, risk_group, risk_label, risk_stage, risk_stage_label, hdl_cholesterol, triglycerides

Patient risk distribution:
    high_risk_prediabetic: 563 meals from 16 patients
    diabetic: 439 meals from 13 patients
    normal: 311 meals from 9 patients
    early_insulin_resistance: 287 meals from 6 patients
    prediabetic: 40 meals from 1 patient

Meal type distribution:
    dinner: 469 meals
    breakfast: 434 meals
    lunch: 422 meals
    snack: 315 meals

Glucose spike statistics:
    mean: 37.44 mg/dL
    std: 38.91 mg/dL
    min: -48.96 mg/dL (high-protein/fat meals can cause mild glucose reduction)
    max: 212.20 mg/dL

Training data carbohydrate range:
    mean: 52.7g
    95th percentile: 94g
    max: 761g
    meals above 100g: 59 (3.6% of training data)

---

### 6. CGM Activity Response Dataset

File: cgmacros_activity_response.csv
Source: Continuous glucose monitor cohort with exercise session tracking
Shape: 933 rows, 19 columns
Purpose: Evidence-based exercise rules for the intervention engine (not used for model training)
Patients: 34

Columns: patient_id, start_time, duration_min, mets_raw_avg, mets_normalized_avg, exercise_intensity, heart_rate_avg, activity_calories, glucose_before, glucose_after, glucose_drop, glucose_drop_pct, timing, age, bmi, hba1c, homa_ir, risk_group, risk_label

Exercise intensity distribution:
    light: 637 sessions
    moderate: 258 sessions
    vigorous: 38 sessions

Timing distribution:
    fasted: 558 sessions
    post_meal: 375 sessions

Decision to use as rules rather than model: The activity dataset was evaluated for regression modeling but rejected for three reasons. First, glucose_before has a correlation of 0.58 with glucose_drop, indicating the dominant signal is regression to mean rather than exercise effect. High baseline glucose naturally drops and low baseline glucose naturally rises regardless of exercise. Second, only 38 vigorous sessions exist, insufficient to learn intensity-specific effects. Third, glucose_drop_pct is mathematically derived from glucose_before and glucose_drop and would constitute direct leakage.

Instead the dataset was used to derive evidence-based lookup rules controlling for baseline glucose range 80 to 120 mg/dL to isolate exercise effect from regression to mean.

Activity rules derived (baseline 80-120 mg/dL):
    fasted light (n=204, high confidence): mean glucose drop -9.3 mg/dL
    fasted moderate (n=71, high confidence): mean glucose drop -11.2 mg/dL
    fasted vigorous (n=8, low confidence): excluded from app
    post_meal light (n=105, high confidence): mean glucose drop -4.4 mg/dL
    post_meal moderate (n=48, medium confidence): mean glucose drop -9.3 mg/dL
    post_meal vigorous (n=10, low confidence): excluded from app

High baseline above 150 mg/dL (n=not filtered): mean drop -34.2 mg/dL
Low baseline below 90 mg/dL: mean change +3.0 mg/dL (caution warning triggered)

---

## ADA Rules and Guidelines Embedded

The following ADA 2023 standards are coded into the intervention engine:

Exercise: 150 minutes per week of moderate-intensity physical activity (ADA Diabetes Prevention Program standard). Post-meal walking specifically recommended for glucose spike reduction.

Sleep: 7 to 9 hours per night. Sleep below 6 hours raises insulin resistance and cortisol. Sleep apnea screening recommended for diabetics and high-risk prediabetics.

Mental Health: PHQ-9 scoring applied. Score 0 to 4 minimal, 5 to 9 mild, 10 to 14 moderate, 15 to 19 moderately severe, 20 to 27 severe. PHQ-9 at or above 10 triggers care pathway escalation and provider referral recommendation.

Alcohol: Maximum 14 drinks per week for men, 7 drinks per week for women. Optimal target is half the maximum limit for glucose control. Alcohol on empty stomach specifically flagged for hypoglycemia risk.

Smoking: Cessation reduces HbA1c by approximately 0.2% and improves insulin sensitivity within weeks. CDC Smokefree program referenced. Nicotine replacement therapy preferred over vaping for metabolically neutral cessation.

Diet: ADA Plate Method 2023. Half the plate non-starchy vegetables, one quarter lean protein, one quarter complex carbohydrates. Carbohydrate target 45 to 60g per meal. Four food swap rules described in the dataset section above.

Care Pathway:
    Tier 1 self-management: risk score below 35 or normal label
    Tier 2 DPP referral: risk score 35 to 79 or prediabetic, high-risk prediabetic, early insulin resistance label. CDC Diabetes Prevention Program reduces conversion risk by 58% over 2 to 3 years. Covered by Medicare and most private insurers.
    Tier 3 doctor referral: risk score 80 or above, diabetic label, HbA1c at or above 6.5%, fasting glucose at or above 126 mg/dL, BMI at or above 35 with high metabolic risk, or existing diagnosis.
    PHQ-9 at or above 10 escalates any tier to minimum Tier 2.

Resources referenced: CDC DPP program finder (cdc.gov/diabetes/prevention/find-a-program.html), HRSA health center finder (findahealthcenter.hrsa.gov), CDC Smokefree program (smokefree.gov), 988 Suicide and Crisis Lifeline.

---

## Data Cleaning and Feature Engineering

### Step 1 Notebook (NHANES Risk Models)

Encoding:
    sex column encoded from numeric (1.0, 2.0) to binary (1=male, 0=female)
    smoking_status_clean encoded from string (never, former, current) to ordinal (0, 1, 2)

Interaction features created:
    age_bmi = age multiplied by bmi divided by 100 (age-adjusted adiposity)
    exercise_sleep = total_exercise_min_week multiplied by avg_sleep_hrs (recovery capacity proxy)
    waist_age = waist_cm multiplied by age divided by 100 (visceral fat burden over time)

Binary target engineering:
    at_risk = 1 if risk_group greater than 0, else 0 (normal versus at-risk for Model A)

Missing value handling:
    XGBoost handles missing natively by learning optimal default directions at each split
    No imputation applied for Model A or Model B

Columns excluded from Model B to prevent circularity:
    risk_group, risk_stage, and risk_stage_label are all derived from HbA1c and fasting glucose thresholds. Including them alongside HbA1c and fasting_glucose as features would mean the model is predicting a label from the same values used to create it. Excluded from all models.

Data splits:
    Model A: 3-way split. 64% train, 16% calibration, 20% test. Stratified by at_risk label.
    Model B: 80/20 stratified train/test split.

Isotonic calibration:
    Fitted on the calibration set (never seen by XGBoost during training) to convert raw probabilities to well-calibrated 0 to 100 risk scores.
    Calibration validated by checking that actual at-risk percentage increases monotonically across score bands.

Feature importance stability:
    Checked via 10 bootstrap samples of 30 training patients each.
    Top 5 features (mt_breakfast, fasting_glucose, carb_fiber_ratio, hba1c, mt_lunch) all stable with coefficient of variation below 0.35.
    ir_proxy and low_fiber_meal identified as unstable but both have low mean importance below 0.04 and do not affect predictions.

---

### Step 2 Notebook (Glucose Spike Model)

Data quality fixes applied in order:

Fix 1: Fiber unit corruption. Patients 16, 18, and 19 had meal_fiber values up to 2,830g. Root cause: milligrams entered as grams during data collection. Fix: divide by 100 where meal_fiber exceeds 40g. This brings values into the realistic range of 0 to 38g matching other patients. A division of 1,000 was tested but produced values of 0.05 to 2.83g which is unrealistically low for full meals. Division by 100 produced values of 0.53 to 28.3g which matches the normal patient distribution.

Fix 2: net_carbs recomputed. The original net_carbs column was computed before the fiber fix, so it still contained corrupted values. After correcting meal_fiber, net_carbs was recomputed as meal_carbs minus meal_fiber clipped at zero. Correlation with glucose_spike improved from 0.131 to 0.231 after recomputation.

Fix 3: net_carbs then dropped. Correlation with meal_carbs was 0.99. Keeping both would dilute SHAP importance between near-identical features. net_carbs was dropped and meal_carbs plus meal_fiber kept separately.

Fix 4: Zero carb inconsistency. 16 meals had meal_carbs equal to zero but calories between 50 and 441. A 420-calorie meal with zero carbohydrates is a data entry error. These carb values were set to NaN and handled natively by XGBoost.

Fix 5: Leakage columns removed. peak_glucose equals baseline_glucose plus glucose_spike mathematically. glucose_auc is computed from the spike curve. recovery_time_min is measured after the spike occurs. None of these exist at prediction time. All three removed.

Fix 6: Redundant columns removed. insulin correlates 0.94 with homa_ir. HOMA-IR is the validated clinical metric. meal_calories correlates 0.92 with the macro-derived calorie calculation. risk_group, risk_stage, risk_label, and risk_stage_label are all derived from HbA1c and fasting glucose thresholds present as features. All removed.

Engineered features:
    hour: extracted from timestamp (dawn phenomenon, breakfast spikes are 2.7 times higher than snacks)
    carb_fiber_ratio: meal_carbs divided by (meal_fiber + 1) (fiber slows carb absorption, ratio is more informative than either alone)
    ir_proxy: homa_ir multiplied by meal_carbs divided by 100 (insulin resistance amplifies carb response)
    carb_x_baseline: meal_carbs multiplied by baseline_glucose divided by 100 (baseline glucose state amplifies response)
    is_breakfast: binary flag for hour 5 to 10, correlation 0.33 with spike. Dropped from final features because correlation with mt_breakfast dummy was 0.95. Keeping both would split SHAP importance identically.
    mt_breakfast, mt_dinner, mt_lunch, mt_snack: one-hot encoded from meal_type

Data split:
    Patient-level split required. With 45 patients a random 80/20 split places the same patient in both train and test. The model would then memorize per-patient glucose baselines rather than learning generalizable meal-to-spike relationships.
    Train: 36 patients, 1,261 meals. Test: 9 patients, 379 meals.
    Optuna patient-level CV: KFold(n_splits=5) applied to the list of training patient IDs, not individual meals.

Post-hoc calibration corrections applied in the app:
    HbA1c above 6.0: multiply predicted spike by (1.0 plus (hba1c minus 6.0) times 0.15)
    HbA1c below 5.5: multiply by 0.85
    Baseline glucose above 120: multiply by (1.0 plus (baseline minus 120) divided by 200)
    Baseline glucose below 80: multiply by 0.90
    HOMA-IR above 5.0: multiply by (1.0 plus (homa_ir minus 5.0) times 0.05)

Extrapolation for large carbohydrate meals:
    The model was trained on meals where 95% had fewer than 94g carbohydrates. Only 59 of 1,640 meals exceeded 100g. Above 94g the model collapses toward the training mean, producing predictions that decrease when carbohydrates increase.
    Fix: For carbohydrates above 94g, the model prediction at 94g is used as an anchor point. Additional carbohydrates above 94g are handled by linear extrapolation at 0.45 mg/dL per gram, derived from clinical literature showing approximately 4 to 5 mg/dL spike per 10g of additional carbohydrates.
    Result: Predictions are now monotonically increasing with carbohydrate load (20g=34, 60g=54, 100g=72, 150g=99, 200g=121 mg/dL).

---

### Step 3 Notebook (Intervention Engine)

The intervention engine is a rules engine, not a machine learning model. No training occurs in this notebook. The notebook builds, validates, and saves structured lookup tables and rule functions.

Rules coded:
    Six domain evaluators (diet, exercise, sleep, mental health, alcohol, smoking)
    Care pathway tier assignment logic
    Activity rules lookup from CGM activity dataset
    Food swap enrichment joining swap database to FNDDS nutrition data

Each domain evaluator returns: status, priority score 1 to 5, urgency level, user value, target value, NHANES population comparison, list of recommendations specific to the user's risk label, and the ADA rule citation.

Recommendation differentiation:
    Each of the six domains has five distinct recommendation sets covering normal, early_insulin_resistance, prediabetic, high_risk_prediabetic, and diabetic labels.
    NHANES statistics cited in recommendations use the exact values from the benchmarks file for the user's risk group.
    Smoking recommendations further differentiate by smoking status (never, former, current) within each risk label.

Priority ordering:
    Priority 1: domains where user value is furthest from target or risk is highest
    Priority 5: domains where user is at or near target
    Care pathway escalations override domain priority for mental health when PHQ-9 is at or above 10

Validated outputs:
    Engine tested on three example user profiles: normal risk, high-risk prediabetic, and diabetic with moderate depression
    Tier assignment validated for all combination of risk score, label, HbA1c, fasting glucose, PHQ-9, and already_diagnosed flag
    Swap coverage validated: 558 high-risk foods confirmed in lookup, FNDDS nutrition confirmed present for swap foods

---

## Model Performance Summary

Model A Risk Score (no lab values):
    Algorithm: XGBoost binary + isotonic calibration
    AUC: 0.74 (95% CI from 1,000 bootstrap resamples)
    Band calibration: Diabetes Unlikely 16.6% actual at-risk, Early Warning 46.9%, Prediabetes Territory 61.6%, Diabetes Territory 85.7%
    Youden threshold computed for optimal sensitivity/specificity balance
    Subgroup AUC analyzed by age group, sex, and ethnicity

Model B Clinical Label (with lab values):
    Algorithm: XGBoost multiclass
    Accuracy: 99%, Balanced accuracy: 99%
    Ablation study confirms HbA1c and fasting glucose contribute 20 percentage points beyond lipids and HOMA-IR alone
    High accuracy is valid: labels are ADA-defined thresholds, model learns those thresholds

Model C Glucose Spike:
    Algorithm: XGBoost regressor + post-hoc calibration
    MAE: 21 mg/dL (95% CI 19 to 23)
    R2: 0.34 (95% CI 0.25 to 0.41)
    Null model MAE (predict training mean for all): 26.4 mg/dL
    Improvement over null: 20.1%
    Directional accuracy: 72% of meal pairs ranked correctly at greater than 5 mg/dL threshold
    Category accuracy: 42% exact, 85% within one adjacent category
    very_high recall (above 70 mg/dL): approximately 18% due to class imbalance and small cohort
    Per-patient MAE analyzed: Patient 49 (HOMA-IR 9.21) drives most error due to sparse training coverage of severe insulin resistance

---

## Limitations

Risk score model ceiling: AUC 0.74 is the expected ceiling for lifestyle-only diabetes screening without laboratory values. This matches the FINDRISC literature range of 0.72 to 0.78. The model's role is to flag elevated risk and recommend testing, not to diagnose.

Glucose model cohort size: 45 patients is insufficient to capture individual biological variation, which accounts for approximately 65% of residual variance. Zeevi et al. (Cell 2015) showed that even with microbiome data, meal glucose prediction achieves R2 of approximately 0.60.

Glucose model very_high underprediction: The model systematically underpredicts spikes above 70 mg/dL due to regression to mean from the small cohort. Post-hoc calibration corrections and a composite detection rule partially address this.

No external validation: Both risk models were trained and tested on NHANES data only. No external dataset validation (BRFSS, UK Biobank) was performed. Bootstrap confidence intervals are provided as a substitute.

No temporal validation: NHANES survey year was not available in the provided dataset. A temporal train/test split was not possible.

NHANES benchmark data quality: Several benchmark columns show identical values across all three risk groups (sleep hours, drinks per week, inactivity rate). These appear to be synthetic or aggregated population estimates rather than measured group-level means. Only columns with differentiated values are used for individual comparison.

Activity model not built: The activity dataset showed severe regression-to-mean confounding (baseline_glucose correlation 0.58 with glucose_drop) and only 38 vigorous sessions. A regression model would predominantly learn baseline glucose effects rather than exercise effects. Evidence-based lookup rules with confidence levels were derived instead.

---

## Disclaimer

This application is a screening and educational tool only. It does not replace medical advice, clinical diagnosis, or treatment by a qualified healthcare provider. Risk scores and predictions are statistical estimates based on population data and may not reflect individual clinical reality. Users with concerning results should consult a qualified healthcare provider.
