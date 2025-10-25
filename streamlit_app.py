# streamlit_app.py
# ------------------------------------------------------------
# Hemodialysis ICU mortality prediction + patient-level SHAP
# Robust per-model KernelExplainer + custom waterfall to avoid PIL errors
# ------------------------------------------------------------

import os
import pickle
from pathlib import Path
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
from autogluon.tabular import TabularPredictor
import plotly.graph_objects as go

# SHAP + plotting
import shap
import matplotlib
matplotlib.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Wedge

# ---------- Streamlit page config ----------
st.set_page_config(layout="wide")
st.title('Mortality Risk Prediction for Hemodialysis Patients in Intensive Care Units')
st.write('Enter the following values for a hemodialysis patient in the Intensive Care Unit to predict their death risk during hospitalization:')

# ---------- Windows/Posix pickle compatibility ----------
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

@st.cache_resource
def loadsupport():
    def _load_pickle(filename, convert=True, extract_first=True):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Required file '{filename}' not found.")
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if extract_first:
            # our pickles store a single object inside a one-element list
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

# ---------- Data pre-processing ----------
emer_clean = emer.dropna(how='any', inplace=False)

# features used by each model (restrict SHAP to these)
allcause_features = set(predictor.feature_metadata.get_features())
card_features     = set(card_predictor.feature_metadata.get_features())
sepsis_features   = set(sepsis_predictor.feature_metadata.get_features())
all_features      = allcause_features.union(card_features).union(sepsis_features)
feature_names_list = list(all_features)

# raw type map (union)
feature_types = {}
feature_types.update(predictor.feature_metadata.type_map_raw)
feature_types.update(card_predictor.feature_metadata.type_map_raw)
feature_types.update(sepsis_predictor.feature_metadata.type_map_raw)

# ---------- Display names mapping (from original app) ----------
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
for f in feature_names_list:
    display_names.setdefault(f, f)  # fallback to original name if missing

# ---------- Bulk input order & helpers ----------
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
    feat = display_name_to_feature.get(name)
    if feat is None:
        st.error(f"Display name '{name}' does not correspond to any feature.")
        st.stop()
    feature_names_list_ordered.append(feat)

num_features = len(feature_names_list_ordered)

st.write(', '.join(bulk_input_order))
st.write('Definition of some variables:')
st.write('(1) PIV: (Neutrophil Count * Platelet Count * Monocyte Count) / Lymphocyte Count')
st.write('(2) SIRI: (Neutrophil Count * Monocyte Count) / Lymphocyte Count')
st.write('(3) NAR: Neutrophil Count / Albumin')
st.write('(4) NLR: Neutrophil Count / Lymphocyte Count')
st.write('(5) CAR: C-Reactive Protein / Albumin')

bulk_input = st.text_area(
    'Enter all values separated by commas. If entered this way, the values will automatically populate each input field to ensure accurate recognition.',
    value='', height=100
)

# ---------- Gauge plot (kept) ----------
def create_ring_plot(probability, title):
    if probability < 0.33:
        color = 'green'
    elif probability < 0.66:
        color = 'orange'
    else:
        color = 'red'
    fig = go.Figure(go.Pie(
        values=[probability, 1 - probability],
        hole=0.7,
        marker=dict(colors=[color, 'lightgray']),
        hoverinfo='none'
    ))
    fig.update_traces(textinfo='none')
    fig.update_layout(
        annotations=[dict(text=f"{probability:.2%}", x=0.5, y=0.5, font_size=20, showarrow=False)],
        title_text=title, showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# ---------- Build input widgets ----------
input_data = {}
missing_features = []
columns_per_row = 6
rows = (num_features + columns_per_row - 1) // columns_per_row

# parse bulk input
if bulk_input.strip() != '':
    bulk_values = [x.strip() for x in bulk_input.strip().split(',')]
    if len(bulk_values) != num_features:
        st.error('The number of values entered does not match the number of features.')
    else:
        for feature, value in zip(feature_names_list_ordered, bulk_values):
            ftype = feature_types.get(feature, 'float')
            if ftype in ['int', 'float']:
                try:
                    value = float(value)
                except ValueError:
                    st.error(f"Invalid numeric value for {display_names.get(feature, feature)}: {value}")
                    st.stop()
            elif ftype == 'datetime':
                try:
                    value = pd.to_datetime(value)
                except ValueError:
                    st.error(f"Invalid date format for {display_names.get(feature, feature)}: {value}")
                    st.stop()
            input_data[feature] = value

# render inputs
for r in range(rows):
    cols = st.columns(columns_per_row)
    for idx in range(columns_per_row):
        k = r * columns_per_row + idx
        if k >= num_features:
            continue
        feature = feature_names_list_ordered[k]
        ftype = feature_types.get(feature, 'float')
        display_name = display_names.get(feature, feature)
        with cols[idx]:
            default_value = input_data.get(feature, '')
            if default_value == '':
                if ftype in ['int', 'float']:
                    value = st.number_input(f"{display_name}:", key=feature)
                elif ftype == 'object':
                    value = st.text_input(f"{display_name}:", key=feature)
                elif ftype == 'datetime':
                    value = st.date_input(f"{display_name}:", key=feature)
                else:
                    value = st.text_input(f"{display_name}:", key=feature)
            else:
                if ftype in ['int', 'float']:
                    value = st.number_input(f"{display_name}:", value=float(default_value), key=feature)
                elif ftype == 'object':
                    value = st.text_input(f"{display_name}:", value=str(default_value), key=feature)
                elif ftype == 'datetime':
                    value = st.date_input(f"{display_name}:", value=pd.to_datetime(default_value), key=feature)
                else:
                    value = st.text_input(f"{display_name}:", value=str(default_value), key=feature)
            input_data[feature] = value

# ---------- SHAP helpers ----------
def get_positive_label(predictor_obj):
    try:
        labels = predictor_obj.class_labels
        if labels is not None and len(labels) == 2:
            return labels[1]
    except Exception:
        pass
    return 1  # default

def proba_callable_for_shap(predictor_obj, feature_names, positive_label):
    # returns f(X) -> 1D array of P(positive)
    def f(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        proba = predictor_obj.predict_proba(X_df)
        if isinstance(proba, pd.DataFrame):
            if positive_label in proba.columns:
                arr = proba[positive_label].to_numpy()
            elif str(positive_label) in proba.columns:
                arr = proba[str(positive_label)].to_numpy()
            elif 1 in proba.columns:
                arr = proba[1].to_numpy()
            elif '1' in proba.columns:
                arr = proba['1'].to_numpy()
            else:
                arr = proba.iloc[:, -1].to_numpy()
        else:
            arr = np.asarray(proba)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                arr = arr[:, 1]
            else:
                arr = arr.reshape(-1)
        return arr.astype(float)
    return f

def compute_local_shap_kernel(predictor_obj, x_row_df, background_df, feature_map, max_bg=50):
    """
    Robust per-patient SHAP on probability scale using KernelExplainer.
    Returns: shap_df (with display names), expected_value (base), fx
    """
    feature_names = list(x_row_df.columns)
    pos_label = get_positive_label(predictor_obj)
    f = proba_callable_for_shap(predictor_obj, feature_names, pos_label)

    # background: exact columns, clean, capped for speed
    bg = background_df[feature_names].dropna()
    if len(bg) == 0:
        # fallback: use the row itself as background
        bg = x_row_df.copy()
    else:
        bg = bg.sample(n=min(max_bg, len(bg)), random_state=0)

    # KernelExplainer on probability scale
    explainer = shap.KernelExplainer(f, bg, link="identity")
    shap_vals = explainer.shap_values(x_row_df, nsamples="auto")
    # shap_vals can be list or ndarray depending on SHAP version
    shap_arr = np.array(shap_vals)
    if shap_arr.ndim == 3:
        # shape: (n_outputs, n_rows, n_features) -> pick output 0
        shap_row = shap_arr[0, 0, :]
    elif shap_arr.ndim == 2:
        # shape: (n_rows, n_features)
        shap_row = shap_arr[0, :]
    else:
        shap_row = shap_arr.reshape(-1)

    expected_value = float(np.array(explainer.expected_value).reshape(-1)[-1])
    fx = float(f(x_row_df.values)[0])

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'value': [x_row_df.iloc[0][c] for c in feature_names],
        'shap': shap_row.astype(float)
    })
    shap_df['abs_shap'] = shap_df['shap'].abs()
    shap_df.sort_values('abs_shap', ascending=False, inplace=True)
    shap_df['feature_display'] = shap_df['feature'].map(lambda x: feature_map.get(x, x))
    return shap_df, expected_value, fx

def plot_topk_bar(ax, shap_df, top_k=8, title=""):
    top = shap_df.head(top_k).copy()
    labels = [f"{r.feature_display} = {r.value}" for r in top.itertuples(index=False)]
    y = np.arange(len(top))[::-1]
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
      A: inputs table; B–D: donuts; E–G: SHAP bars.
    """
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1])

    # Panel A: inputs table
    axA = fig.add_subplot(gs[0, 0])
    cell_text, cell_colors = [], []
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

def plot_custom_waterfall(ax, shap_df, base_value, fx, top_k=10, title=""):
    """
    Draw a probability-scale waterfall without using shap.plots.waterfall (avoids PIL errors).
    base_value + sum(shap) ~= fx.
    """
    top = shap_df.head(top_k).copy()
    # sort so negative first -> leftward, positive next -> rightward for clean stacking
    top = top.sort_values('shap')
    contribs = top['shap'].values
    names = top['feature_display'].values

    # cumulative positions
    starts = [base_value]
    for s in contribs[:-1]:
        starts.append(starts[-1] + s)
    starts = np.array(starts)
    widths = np.abs(contribs)
    lefts = np.where(contribs >= 0, starts, starts + contribs)

    y = np.arange(len(top))
    colors = ['red' if s > 0 else 'blue' for s in contribs]
    ax.barh(y, widths, left=lefts, color=colors, align='center', edgecolor='none')
    ax.set_yticks(y, labels=[str(n) for n in names], fontsize=8)
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=0.8, label='base')
    ax.axvline(fx, color='black', linestyle='-', linewidth=1.0, label='prediction')
    ax.set_xlabel("Predicted probability")
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    # Ensure visible padding
    xmin = min(base_value, fx, lefts.min()) - 0.02
    xmax = max(base_value, fx, (lefts + widths).max()) + 0.02
    ax.set_xlim(xmin, xmax)

# ---------- Prediction & Explanations ----------
if st.button('Predict'):
    # build input row
    input_df = pd.DataFrame([input_data])

    # impute missing & record what we imputed
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

    # normalize datetimes
    for feature, ftype in feature_types.items():
        if ftype == 'datetime' and feature in input_df.columns:
            input_df[feature] = pd.to_datetime(input_df[feature])

    # predictions
    prediction = predictor.predict(input_df)
    probability = predictor.predict_proba(input_df)
    card_prediction = card_predictor.predict(input_df)
    card_probability = card_predictor.predict_proba(input_df)
    sepsis_prediction = sepsis_predictor.predict(input_df)
    sepsis_probability = sepsis_predictor.predict_proba(input_df)

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

    # results UI
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

    # imputed variables transparency
    if missing_features:
        st.warning("The following variables were missing and have been filled with average/mode values:")
        for var in missing_features:
            st.write(f"{var}: {missing_values_used[var]}")

    # ---------- Patient-specific SHAP ----------
    st.markdown("### Patient‑specific feature contributions (SHAP)")
    st.caption("Bars to the right increase predicted risk; bars to the left decrease risk (relative to the model’s base rate).")

    # exact per-model features & subsets
    allcause_feats = [f for f in predictor.features() if f in emer_clean.columns]
    cardio_feats   = [f for f in card_predictor.features() if f in emer_clean.columns]
    infect_feats   = [f for f in sepsis_predictor.features() if f in emer_clean.columns]

    x_allcause = input_df[allcause_feats].copy()
    x_cardio   = input_df[cardio_feats].copy()
    x_infect   = input_df[infect_feats].copy()

    bg_allcause = emer_clean[allcause_feats]
    bg_cardio   = emer_clean[cardio_feats]
    bg_infect   = emer_clean[infect_feats]

    shap_allcause_df, base_allcause, fx_allcause = compute_local_shap_kernel(
        predictor, x_allcause, bg_allcause, display_names, max_bg=50
    )
    shap_cardio_df, base_cardio, fx_cardio = compute_local_shap_kernel(
        card_predictor, x_cardio, bg_cardio, display_names, max_bg=50
    )
    shap_infect_df, base_infect, fx_infect = compute_local_shap_kernel(
        sepsis_predictor, x_infect, bg_infect, display_names, max_bg=50
    )

    # three columns of SHAP bars + tables
    ecol = st.columns(3)

    def _fmt_table(df):
        out = df[['feature_display','value','shap']].head(8).rename(
            columns={'feature_display':'feature','shap':'contribution'}
        ).copy()
        # numeric formatting for readability
        if 'contribution' in out.columns:
            out['contribution'] = out['contribution'].astype(float).round(4)
        return out

    with ecol[0]:
        st.markdown("**All‑cause: top drivers**")
        figA, axA = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar(axA, shap_allcause_df, top_k=8, title="")
        st.pyplot(figA, clear_figure=True)
        st.dataframe(_fmt_table(shap_allcause_df), use_container_width=True)

    with ecol[1]:
        st.markdown("**Cardiovascular: top drivers**")
        figB, axB = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar(axB, shap_cardio_df, top_k=8, title="")
        st.pyplot(figB, clear_figure=True)
        st.dataframe(_fmt_table(shap_cardio_df), use_container_width=True)

    with ecol[2]:
        st.markdown("**Infection‑related: top drivers**")
        figC, axC = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar(axC, shap_infect_df, top_k=8, title="")
        st.pyplot(figC, clear_figure=True)
        st.dataframe(_fmt_table(shap_infect_df), use_container_width=True)

    # ---------- Custom waterfall plots (no PIL error) ----------
    with st.expander("Show SHAP waterfall plots (per‑patient)"):
        wcols = st.columns(3)
        with wcols[0]:
            st.markdown("All‑cause waterfall")
            figW, axW = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall(axW, shap_allcause_df, base_allcause, fx_allcause, top_k=10, title="")
            st.pyplot(figW, clear_figure=True)
        with wcols[1]:
            st.markdown("Cardiovascular waterfall")
            figW2, axW2 = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall(axW2, shap_cardio_df, base_cardio, fx_cardio, top_k=10, title="")
            st.pyplot(figW2, clear_figure=True)
        with wcols[2]:
            st.markdown("Infection‑related waterfall")
            figW3, axW3 = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall(axW3, shap_infect_df, base_infect, fx_infect, top_k=10, title="")
            st.pyplot(figW3, clear_figure=True)

    # ---------- Export camera‑ready Figure 6 ----------
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
