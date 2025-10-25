# streamlit_app.py
import os
import pickle
from pathlib import Path
import pathlib

import streamlit as st
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import plotly.graph_objects as go

# NEW: SHAP + Matplotlib for patient-level plots and figure export
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Wedge

# ------------------------------------------------------------
# Streamlit page config
# ------------------------------------------------------------
st.set_page_config(layout="wide")
st.title('Mortality Risk Prediction for Hemodialysis Patients in Intensive Care Units')
st.write('Enter the following values for a hemodialysis patient in the Intensive Care Unit to predict their death risk during hospitalization:')

# ------------------------------------------------------------
# Utilities for Windows/Posix pickle compatibility
# ------------------------------------------------------------
pathlib.WindowsPath = pathlib.PosixPath

def convert_strings_to_paths(obj):
    if isinstance(obj, str) and ('path' in obj.lower() or '/' in obj):
        return Path(obj)
    elif isinstance(obj, dict):
        return {k: convert_strings_to_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_strings_to_paths(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_strings_to_paths(item) for item in obj)
    return obj

def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def loadsupport():
    def _load_pickle(filename, convert=True, extract_first=True):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Required file '{filename}' not found.")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if extract_first:
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            else:
                raise ValueError(f"Loaded data from '{filename}' is not a single-element list.")
        if convert and not isinstance(data, pd.DataFrame):
            data = convert_strings_to_paths(data)
        return data

    predictorall = _load_pickle('predictorall.pkl')
    predictorcardfull = _load_pickle('predictorcard.pkl')
    predictorsepsisfull = _load_pickle('predictorsepsis.pkl')
    emer = _load_pickle('emer.pkl', convert=False, extract_first=True)
    return predictorall, predictorcardfull, predictorsepsisfull, emer

predictor, card_predictor, sepsis_predictor, emer = loadsupport()

# ------------------------------------------------------------
# Data pre-processing
# ------------------------------------------------------------
emer_clean = emer.dropna(how='any', inplace=False)

# Features used for the UI (we’ll use each model’s own features for SHAP)
allcause_features = set(predictor.feature_metadata.get_features())
card_features = set(card_predictor.feature_metadata.get_features())
sepsis_features = set(sepsis_predictor.feature_metadata.get_features())
all_features = allcause_features.union(card_features).union(sepsis_features)
feature_names_list = list(all_features)

# Aggregate feature types (raw types from predictors)
feature_types = {}
feature_types.update(predictor.feature_metadata.type_map_raw)
feature_types.update(card_predictor.feature_metadata.type_map_raw)
feature_types.update(sepsis_predictor.feature_metadata.type_map_raw)

# ------------------------------------------------------------
# Display names mapping (from your original app)
# ------------------------------------------------------------
display_names = {
    '昏迷/意识丧失模糊': 'Coma(0/1)',
    '心肺复苏': 'Use of Cardiopulmonary Resuscitation(0/1)',
    '脓毒症': 'Sepsis(0/1)',
    '舒张压': 'Diastolic Blood Pressure(mmHg)',
    '高敏肌钙蛋白T(TnT-T)(发光法)': 'Troponin T(ng/ml)',
    '肌红蛋白(肌血球素)MYO': 'Myoglobin(ng/ml)',
    '中性分叶粒细胞NEUT#': 'Neutrophil Count(×10^9/L)',
    '凝血酶原活动度PT(%)': 'Prothrombin Time Activity(%)',
    '大血小板百分率': 'Platelet Large Cell Ratio(%)',
    '活化部分凝血活酶时间APTT': 'Activated Partial Thromboplastin Time(s)',
    'NAR': 'NAR',
    '收缩压': 'Systolic Blood Pressure(mmHg)',
    '肌酸激酶同工酶CK-MB(mass法)': 'Creatine Kinase-MB(ng/ml)',
    '阴离子间隙AG': 'Anion Gap(mmol/L)',
    'C反应蛋白': 'C-reactive protein(mg/L)',
    '平均血小板体积MPV': 'Mean Platelet Volume(fL)',
    '总胆红素TBIL( µmol/L)': 'Total bilirubin(µmol/L)',
    'PIV': 'PIV',
    'SIRI': 'SIRI',
    'CAR': 'CAR',
    '中性分叶粒细胞NEUT%': 'Neutrophil Percentage(%)',
    'B型钠尿肽前体(NT-proBNP)': 'NT-proBNP(pg/mL)',
    'NLR': 'NLR',
    '嗜酸细胞EO%': 'Eosinophil Percentage(%)',
    '平均红细胞血红蛋白浓度MCHC': 'Mean Corpuscular Hemoglobin Concentration(g/L)',
    '心衰': 'Heart Failure(0/1)'
}
for feature in feature_names_list:
    if feature not in display_names:
        display_names[feature] = feature  # fallback: original name

# ------------------------------------------------------------
# Bulk input UI (kept from your app)
# ------------------------------------------------------------
st.write('Alternatively, you can input all values separated by commas. If entered this way, the values will automatically populate each input field to ensure accurate recognition.')

bulk_input_order = [
    'Mean Corpuscular Hemoglobin Concentration(g/L)',
    'Creatine Kinase-MB(ng/ml)',
    'Neutrophil Percentage(%)',
    'Systolic Blood Pressure(mmHg)',
    'Troponin T(ng/ml)',
    'NAR',
    'Sepsis(0/1)',
    'NLR',
    'Platelet Large Cell Ratio(%)',
    'Coma(0/1)',
    'C-reactive protein(mg/L)',
    'PIV',
    'Neutrophil Count(×10^9/L)',
    'SIRI',
    'Total bilirubin(µmol/L)',
    'Prothrombin Time Activity(%)',
    'Diastolic Blood Pressure(mmHg)',
    'Mean Platelet Volume(fL)',
    'Myoglobin(ng/ml)',
    'Anion Gap(mmol/L)',
    'NT-proBNP(pg/mL)',
    'Use of Cardiopulmonary Resuscitation(0/1)',
    'CAR',
    'Eosinophil Percentage(%)',
    'Activated Partial Thromboplastin Time(s)',
    'Heart Failure(0/1)'
]
display_name_to_feature = {v: k for k, v in display_names.items()}
feature_names_list_ordered = []
for name in bulk_input_order:
    feature = display_name_to_feature.get(name)
    if feature:
        feature_names_list_ordered.append(feature)
    else:
        st.error(f"Display name '{name}' does not correspond to any feature.")
        st.stop()
num_features = len(feature_names_list_ordered)

st.write(', '.join(bulk_input_order))
st.write('Definition of some variables:')
st.write('(1) PIV: (Neutrophil Count * Platelet Count * Monocyte Count) / Lymphocyte Count')
st.write('(2) SIRI: (Neutrophil Count * Monocyte Count) / Lymphocyte Count')
st.write('(3) NAR: Neutrophil Count / Albumin')
st.write('(4) NLR: Neutrophil Count / Lymphocyte Count')
st.write('(5) CAR: C-Reactive Protein / Albumin')

bulk_input = st.text_area('Enter all values separated by commas. If entered this way, the values will automatically populate each input field to ensure accurate recognition.', value='', height=100)

# ------------------------------------------------------------
# Gauge plot function (kept)
# ------------------------------------------------------------
def create_ring_plot(probability, title):
    if probability < 0.33:
        color = 'green'
    elif probability < 0.66:
        color = 'orange'
    else:
        color = 'red'
    fig = go.Figure(go.Pie(values=[probability, 1 - probability],
                           hole=0.7,
                           marker=dict(colors=[color, 'lightgray']),
                           hoverinfo='none'))
    fig.update_traces(textinfo='none')
    fig.update_layout(
        annotations=[dict(text=f"{probability:.2%}", x=0.5, y=0.5, font_size=20, showarrow=False)],
        title_text=title,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ------------------------------------------------------------
# Input widgets
# ------------------------------------------------------------
input_data = {}
missing_features = []
columns_per_row = 6
rows = (num_features + columns_per_row - 1) // columns_per_row

# Process bulk input
if bulk_input.strip() != '':
    bulk_values = bulk_input.strip().split(',')
    if len(bulk_values) != len(feature_names_list_ordered):
        st.error('The number of values entered does not match the number of features.')
    else:
        for feature, value in zip(feature_names_list_ordered, bulk_values):
            feature_type = feature_types.get(feature, 'float')
            value = value.strip()
            if feature_type in ['int', 'float']:
                try:
                    value = float(value)
                except ValueError:
                    st.error(f"Invalid numeric value for {display_names.get(feature, feature)}: {value}")
                    st.stop()
            elif feature_type == 'datetime':
                try:
                    value = pd.to_datetime(value)
                except ValueError:
                    st.error(f"Invalid date format for {display_names.get(feature, feature)}: {value}")
                    st.stop()
            input_data[feature] = value

# Render inputs
for row in range(rows):
    cols = st.columns(columns_per_row)
    for idx in range(columns_per_row):
        feature_idx = row * columns_per_row + idx
        if feature_idx < num_features:
            feature = feature_names_list_ordered[feature_idx]
            feature_type = feature_types.get(feature, 'float')
            display_name = display_names.get(feature, feature)
            with cols[idx]:
                default_value = input_data.get(feature, '')
                if default_value == '':
                    if feature_type in ['int', 'float']:
                        value = st.number_input(f"{display_name}:", key=feature)
                    elif feature_type == 'object':
                        value = st.text_input(f"{display_name}:", key=feature)
                    elif feature_type == 'datetime':
                        value = st.date_input(f"{display_name}:", key=feature)
                    else:
                        value = st.text_input(f"{display_name}:", key=feature)
                else:
                    if feature_type in ['int', 'float']:
                        value = st.number_input(f"{display_name}:", value=float(default_value), key=feature)
                    elif feature_type == 'object':
                        value = st.text_input(f"{display_name}:", value=str(default_value), key=feature)
                    elif feature_type == 'datetime':
                        value = st.date_input(f"{display_name}:", value=pd.to_datetime(default_value), key=feature)
                    else:
                        value = st.text_input(f"{display_name}:", value=str(default_value), key=feature)
                input_data[feature] = value

# ------------------------------------------------------------
# SHAP helpers (robust, model-agnostic)
# ------------------------------------------------------------
def get_positive_label(predictor_obj):
    # For binary: class_labels = [negative, positive]
    try:
        labels = predictor_obj.class_labels
        if labels is not None and len(labels) == 2:
            return labels[1]
    except Exception:
        pass
    return 1  # safe default

def proba_callable_for_shap(predictor_obj, feature_names, positive_label):
    def f(X):
        # X arrives as numpy array from SHAP; reconvert to DataFrame with exact columns
        X_df = pd.DataFrame(X, columns=feature_names)
        proba = predictor_obj.predict_proba(X_df)
        # Normalize to DataFrame
        if isinstance(proba, pd.DataFrame):
            cols = list(proba.columns)
            # Try exact match of positive label
            if positive_label in proba.columns:
                arr = proba[positive_label].to_numpy()
            elif str(positive_label) in proba.columns:
                arr = proba[str(positive_label)].to_numpy()
            else:
                # fallback: try column '1', else last column
                if 1 in proba.columns:
                    arr = proba[1].to_numpy()
                elif '1' in proba.columns:
                    arr = proba['1'].to_numpy()
                else:
                    arr = proba.iloc[:, -1].to_numpy()
        else:
            # ndarray shape (n,2) or (n,)
            arr = np.asarray(proba)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                arr = arr[:, 1]
            else:
                arr = arr.reshape(-1)
        return arr  # 1D shape (n,)
    return f

def compute_local_shap_for_model(predictor_obj, x_row_df, background_df, feature_map):
    """
    predictor_obj: AutoGluon TabularPredictor
    x_row_df: DataFrame with exactly the model's features (one row)
    background_df: DataFrame with same columns (many rows)
    feature_map: dict original_feature_name -> display_name
    Returns: shap_df (mapped names), shap_exp (Explanation), base_value, fx (pos prob)
    """
    feature_names = list(x_row_df.columns)
    pos_label = get_positive_label(predictor_obj)
    f = proba_callable_for_shap(predictor_obj, feature_names, pos_label)

    # SHAP masker & explainer
    masker = shap.maskers.Independent(background_df, max_samples=min(200, len(background_df)))
    explainer = shap.Explainer(f, masker)  # model-agnostic; works for NN & trees

    exp = explainer(x_row_df)  # shap.Explanation
    # exp.values -> shape (1, n_features)
    shap_vals = np.array(exp.values).reshape(1, -1)[0]
    base_val = float(np.array(exp.base_values).reshape(-1)[-1])

    # actual predicted probability for sanity check
    fx = float(f(x_row_df.values)[0])

    # Build DataFrame and map to display names
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'value': [x_row_df.iloc[0][c] for c in feature_names],
        'shap': shap_vals
    })
    shap_df['abs_shap'] = np.abs(shap_df['shap'])
    shap_df['direction'] = np.where(shap_df['shap'] >= 0, '+', '−')
    shap_df.sort_values('abs_shap', ascending=False, inplace=True)

    # Map to display names for UI consistency
    shap_df['feature_display'] = shap_df['feature'].map(lambda x: feature_map.get(x, x))
    return shap_df, exp, base_val, fx

def plot_topk_bar(ax, shap_df, top_k=8, title=""):
    top = shap_df.head(top_k).copy()
    labels = [f"{r.feature_display} = {r.value}" for r in top.itertuples(index=False)]
    y = np.arange(len(top))[::-1]  # invert for top at top
    ax.barh(y, top['shap'].values[::-1], align='center')
    ax.set_yticks(y, labels=labels[::-1], fontsize=8)
    ax.axvline(0, color='k', linewidth=0.6)
    ax.set_xlabel("SHAP contribution to P(death)")
    if title:
        ax.set_title(title, fontsize=12)

def draw_donut(ax, prob: float, title: str):
    prob = float(np.clip(prob, 0.0, 1.0))
    color = 'green' if prob < 0.33 else ('orange' if prob < 0.66 else 'red')
    ax.add_patch(Wedge((0,0), 1.0, 0, 360, width=0.3, facecolor='lightgray', edgecolor='none'))
    ax.add_patch(Wedge((0,0), 1.0, 90, 90 - 360*prob, width=0.3, facecolor=color, edgecolor='none'))
    ax.text(0, 0, f"{prob:.0%}", ha='center', va='center', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.axis('off')

def export_figure6(input_display_pairs,
                   risk_allcause, risk_cardio, risk_infection,
                   shap_allcause, shap_cardio, shap_infect,
                   outfile_png="figure6_multiplot.png",
                   outfile_svg="figure6_multiplot.svg"):
    """
    Build a 3x3 layout:
      A: inputs table, B–D: donuts, E–G: SHAP bars
    """
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1])

    # Panel A: inputs table
    axA = fig.add_subplot(gs[0, 0])
    # Render as a table (two columns)
    cell_text = []
    cell_colors = []
    for name, val, imputed in input_display_pairs:
        flag = "•" if imputed else ""
        cell_text.append([name, f"{val}{flag}"])
        cell_colors.append(['white', '#FFF4E6' if imputed else 'white'])
    table = axA.table(cellText=cell_text, colLabels=["Variable", "Value"],
                      cellLoc='left', colLoc='left',
                      colColours=['#F2F2F2','#F2F2F2'],
                      cellColours=cell_colors, loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
    axA.axis('off'); axA.set_title("Entered inputs (• imputed)", fontsize=12)
    axA.text(-0.08, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight='bold', va='top')

    # B–D donuts
    axB = fig.add_subplot(gs[0, 1]); draw_donut(axB, risk_allcause, "All‑cause mortality")
    axB.text(-0.08, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight='bold', va='top')
    axC = fig.add_subplot(gs[0, 2]); draw_donut(axC, risk_cardio, "Cardiovascular mortality")
    axC.text(-0.08, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight='bold', va='top')
    axD = fig.add_subplot(gs[1, 1]); draw_donut(axD, risk_infection, "Infection‑related mortality")
    axD.text(-0.08, 1.05, "D", transform=axD.transAxes, fontsize=14, fontweight='bold', va='top')

    # E–G SHAP bars
    axE = fig.add_subplot(gs[1:, 0]); plot_topk_bar(axE, shap_allcause, top_k=8, title="All‑cause: top contributors")
    axE.text(-0.08, 1.05, "E", transform=axE.transAxes, fontsize=14, fontweight='bold', va='top')
    axF = fig.add_subplot(gs[1:, 1]); plot_topk_bar(axF, shap_cardio, top_k=8, title="Cardiovascular: top contributors")
    axF.text(-0.08, 1.05, "F", transform=axF.transAxes, fontsize=14, fontweight='bold', va='top')
    axG = fig.add_subplot(gs[1:, 2]); plot_topk_bar(axG, shap_infect, top_k=8, title="Infection‑related: top contributors")
    axG.text(-0.08, 1.05, "G", transform=axG.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(outfile_png, bbox_inches='tight')
    if outfile_svg is not None:
        fig.savefig(outfile_svg, bbox_inches='tight')
    return outfile_png, outfile_svg

# ------------------------------------------------------------
# Predict & Explain
# ------------------------------------------------------------
if st.button('Predict'):
    # 1) Build input row
    input_df = pd.DataFrame([input_data])

    # 2) Impute missing values (record what we imputed for transparency)
    missing_features = []
    missing_values_used = {}

    for feature in feature_names_list_ordered:
        if feature not in input_df.columns:
            input_df[feature] = np.nan
        if pd.isnull(input_df.loc[0, feature]) or input_df.loc[0, feature] == '':
            ftype = feature_types.get(feature, 'float')
            if ftype in ['int', 'float']:
                mean_value = emer_clean[feature].mean()
                input_df.loc[0, feature] = mean_value
                missing_features.append(display_names.get(feature, feature))
                missing_values_used[display_names.get(feature, feature)] = mean_value
            elif ftype == 'object':
                mode_value = emer_clean[feature].mode()[0]
                input_df.loc[0, feature] = mode_value
                missing_features.append(display_names.get(feature, feature))
                missing_values_used[display_names.get(feature, feature)] = mode_value
            else:
                input_df.loc[0, feature] = None
                missing_features.append(display_names.get(feature, feature))
                missing_values_used[display_names.get(feature, feature)] = None

    # Datetime normalization
    datetime_features = [f for f, t in feature_types.items() if t == 'datetime']
    for feature in datetime_features:
        if feature in input_df.columns:
            input_df[feature] = pd.to_datetime(input_df[feature])

    # 3) Model predictions
    prediction = predictor.predict(input_df)
    probability = predictor.predict_proba(input_df)
    card_prediction = card_predictor.predict(input_df)
    card_probability = card_predictor.predict_proba(input_df)
    sepsis_prediction = sepsis_predictor.predict(input_df)
    sepsis_probability = sepsis_predictor.predict_proba(input_df)

    # Extract P(positive)
    def _pick_pos(proba_df, predictor_obj):
        pos = get_positive_label(predictor_obj)
        if isinstance(proba_df, pd.DataFrame):
            if pos in proba_df.columns:
                return float(proba_df.iloc[0][pos])
            if str(pos) in proba_df.columns:
                return float(proba_df.iloc[0][str(pos)])
            if 1 in proba_df.columns:
                return float(proba_df.iloc[0][1])
            if '1' in proba_df.columns:
                return float(proba_df.iloc[0]['1'])
            return float(proba_df.iloc[0, -1])
        else:
            arr = np.asarray(proba_df)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return float(arr[0, 1])
            return float(arr[0])
    risk_of_death_probability = _pick_pos(probability, predictor)
    card_risk_probability = _pick_pos(card_probability, card_predictor)
    sepsis_risk_probability = _pick_pos(sepsis_probability, sepsis_predictor)

    # 4) Show results (gauges)
    st.subheader('Prediction Results')
    cols = st.columns(3)

    with cols[0]:
        st.markdown("### All-cause Mortality")
        prediction_text = 'Low risk of mortality' if prediction.iloc[0] == 0 else 'Elevated mortality risk, requiring intervention'
        st.write(f"**Prediction:** {prediction_text}")
        st.plotly_chart(create_ring_plot(risk_of_death_probability, "Probability of Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of mortality: {risk_of_death_probability:.2%}")

    with cols[1]:
        st.markdown("### Cardiovascular Mortality")
        card_prediction_text = 'Low risk of cardiovascular death' if card_prediction.iloc[0] == 0 else 'Elevated cardiovascular mortality risk, requiring intervention'
        st.write(f"**Prediction:** {card_prediction_text}")
        st.plotly_chart(create_ring_plot(card_risk_probability, "Probability of Cardiovascular Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of Cardiovascular mortality: {card_risk_probability:.2%}")

    with cols[2]:
        st.markdown("### Infection-related Mortality")
        sepsis_prediction_text = 'Low risk of infection-related mortality' if sepsis_prediction.iloc[0] == 0 else 'Elevated infection-related mortality risk, requiring intervention'
        st.write(f"**Prediction:** {sepsis_prediction_text}")
        st.plotly_chart(create_ring_plot(sepsis_risk_probability, "Probability of Infection-related Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of Infection-related mortality: {sepsis_risk_probability:.2%}")

    # 5) Show which variables were imputed
    if missing_features:
        st.warning("The following variables were missing and have been filled with average/mode values:")
        for var in missing_features:
            st.write(f"{var}: {missing_values_used[var]}")

    # 6) Patient‑specific SHAP (the key fix: use each model’s own features)
    st.markdown("### Patient‑specific feature contributions (SHAP)")
    st.caption("Bars to the right increase predicted risk; bars to the left decrease risk (relative to the model’s base rate).")

    # Prepare background samples per model (exact feature sets)
    allcause_feats = [f for f in predictor.features() if f in emer_clean.columns]
    cardio_feats   = [f for f in card_predictor.features() if f in emer_clean.columns]
    infect_feats   = [f for f in sepsis_predictor.features() if f in emer_clean.columns]

    # Ensure the input row covers these subsets
    x_allcause = input_df[allcause_feats].copy()
    x_cardio   = input_df[cardio_feats].copy()
    x_infect   = input_df[infect_feats].copy()

    # Background: sample rows with no missing values in those subsets
    bg_allcause = emer_clean[allcause_feats].dropna().sample(n=min(200, len(emer_clean)), random_state=0)
    bg_cardio   = emer_clean[cardio_feats].dropna().sample(n=min(200, len(emer_clean)), random_state=0)
    bg_infect   = emer_clean[infect_feats].dropna().sample(n=min(200, len(emer_clean)), random_state=0)

    shap_allcause_df, exp_allcause, base_allcause, fx_allcause = compute_local_shap_for_model(
        predictor, x_allcause, bg_allcause, display_names
    )
    shap_cardio_df, exp_cardio, base_cardio, fx_cardio = compute_local_shap_for_model(
        card_predictor, x_cardio, bg_cardio, display_names
    )
    shap_infect_df, exp_infect, base_infect, fx_infect = compute_local_shap_for_model(
        sepsis_predictor, x_infect, bg_infect, display_names
    )

    # Three columns: show bar plots + top tables
    ecol = st.columns(3)
    # All-cause
    with ecol[0]:
        st.markdown("**All‑cause: top drivers**")
        figA, axA = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar(axA, shap_allcause_df, top_k=8, title="")
        st.pyplot(figA, clear_figure=True)
        st.dataframe(shap_allcause_df[['feature_display','value','shap']].head(8).rename(
            columns={'feature_display':'feature','shap':'contribution'}))

    # Cardiovascular
    with ecol[1]:
        st.markdown("**Cardiovascular: top drivers**")
        figB, axB = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar(axB, shap_cardio_df, top_k=8, title="")
        st.pyplot(figB, clear_figure=True)
        st.dataframe(shap_cardio_df[['feature_display','value','shap']].head(8).rename(
            columns={'feature_display':'feature','shap':'contribution'}))

    # Infection-related
    with ecol[2]:
        st.markdown("**Infection‑related: top drivers**")
        figC, axC = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar(axC, shap_infect_df, top_k=8, title="")
        st.pyplot(figC, clear_figure=True)
        st.dataframe(shap_infect_df[['feature_display','value','shap']].head(8).rename(
            columns={'feature_display':'feature','shap':'contribution'}))

    # Optional: per-patient SHAP waterfall (compact) – one example (all-cause)
    with st.expander("Show SHAP waterfall plots (per‑patient)"):
        wcols = st.columns(3)
        with wcols[0]:
            st.markdown("All‑cause waterfall")
            plt.close('all')
            figW = plt.figure(figsize=(5, 3), dpi=150)
            shap.plots.waterfall(exp_allcause[0], show=False, max_display=10)
            st.pyplot(figW, clear_figure=True)
        with wcols[1]:
            st.markdown("Cardiovascular waterfall")
            plt.close('all')
            figW2 = plt.figure(figsize=(5, 3), dpi=150)
            shap.plots.waterfall(exp_cardio[0], show=False, max_display=10)
            st.pyplot(figW2, clear_figure=True)
        with wcols[2]:
            st.markdown("Infection‑related waterfall")
            plt.close('all')
            figW3 = plt.figure(figsize=(5, 3), dpi=150)
            shap.plots.waterfall(exp_infect[0], show=False, max_display=10)
            st.pyplot(figW3, clear_figure=True)

    # 7) Export camera‑ready Figure 6 (multi-panel)
    # Prepare inputs table for export (Panel A): mark imputed values
    ordered_display_pairs = []
    for feat in feature_names_list_ordered:
        disp = display_names.get(feat, feat)
        val = input_df.loc[0, feat]
        is_imp = disp in missing_values_used
        vs = f"{val:.3g}" if isinstance(val, float) else str(val)
        ordered_display_pairs.append((disp, vs, is_imp))

    if st.button("Export camera‑ready Figure 6 (PNG + SVG)"):
        png_path, svg_path = export_figure6(
            input_display_pairs=ordered_display_pairs,
            risk_allcause=float(risk_of_death_probability),
            risk_cardio=float(card_risk_probability),
            risk_infection=float(sepsis_risk_probability),
            shap_allcause=shap_allcause_df,
            shap_cardio=shap_cardio_df,
            shap_infect=shap_infect_df,
            outfile_png="figure6_multiplot.png",
            outfile_svg="figure6_multiplot.svg",
        )
        with open(png_path, "rb") as f:
            st.download_button("Download Figure 6 (PNG)", data=f, file_name="Figure6.png", mime="image/png")
        with open(svg_path, "rb") as f:
            st.download_button("Download Figure 6 (SVG)", data=f, file_name="Figure6.svg", mime="image/svg+xml")
