# streamlit_app.py
# ------------------------------------------------------------
# Mortality prediction + per-patient interpretability (stable & fast)
# - Robust, small per-model background (imputed, summarized with shap.sample K=32)
# - KernelExplainer(link="logit"), nsamples capped (256)
# - Safe fallback local effects if SHAP degenerates
# - Predictions rendered before SHAP and persisted in session_state
# - Custom Matplotlib bar & waterfall (log-odds); camera-ready Figure 6 exporter
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
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Wedge

# -------------------- Streamlit page config --------------------
st.set_page_config(layout="wide")
st.title('Mortality Risk Prediction for Hemodialysis Patients in Intensive Care Units')
st.write('Enter the following values for a hemodialysis patient in the Intensive Care Unit to predict their death risk during hospitalization:')

# -------------------- Windows/Posix pickle compatibility --------------------
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
            if isinstance(data, list) and len(data) == 1:
                data = data[0]
            else:
                raise ValueError(f"Loaded data from '{filename}' is not a single-element list.")
        if convert and not isinstance(data, pd.DataFrame):
            data = convert_strings_to_paths(data)
        return data

    predictorall = _load_pickle('predictorall.pkl')
    predictorcard = _load_pickle('predictorcard.pkl')
    predictorsepsis = _load_pickle('predictorsepsis.pkl')
    emer = _load_pickle('emer.pkl', convert=False, extract_first=True)
    return predictorall, predictorcard, predictorsepsis, emer

predictor, card_predictor, sepsis_predictor, emer = loadsupport()

# -------------------- Feature sets & types --------------------
allcause_features = set(predictor.feature_metadata.get_features())
card_features     = set(card_predictor.feature_metadata.get_features())
sepsis_features   = set(sepsis_predictor.feature_metadata.get_features())
all_features      = allcause_features.union(card_features).union(sepsis_features)
feature_names_list = list(all_features)

feature_types = {}
feature_types.update(predictor.feature_metadata.type_map_raw)
feature_types.update(card_predictor.feature_metadata.type_map_raw)
feature_types.update(sepsis_predictor.feature_metadata.type_map_raw)

# -------------------- Display names mapping (from original app) --------------------
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
    display_names.setdefault(f, f)

# -------------------- Bulk input order & UI --------------------
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

# -------------------- Gauge plot --------------------
def create_ring_plot(probability, title):
    color = 'green' if probability < 0.33 else ('orange' if probability < 0.66 else 'red')
    fig = go.Figure(go.Pie(values=[probability, 1 - probability],
                           hole=0.7,
                           marker=dict(colors=[color, 'lightgray']),
                           hoverinfo='none'))
    fig.update_traces(textinfo='none')
    fig.update_layout(
        annotations=[dict(text=f"{probability:.2%}", x=0.5, y=0.5, font_size=20, showarrow=False)],
        title_text=title, showlegend=False, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# -------------------- Input widgets --------------------
input_data = {}
columns_per_row = 6
rows = (num_features + columns_per_row - 1) // columns_per_row

# Parse bulk input
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

# Render inputs
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

# -------------------- SHAP utilities --------------------
def get_positive_label(predictor_obj):
    try:
        labels = predictor_obj.class_labels
        if labels is not None and len(labels) == 2:
            return labels[1]
    except Exception:
        pass
    return 1

def proba_callable_for_shap(predictor_obj, feature_names, positive_label):
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

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s)

@st.cache_data(show_spinner=False)
def build_background(df_raw: pd.DataFrame, feature_list, target_rows: int = 500) -> pd.DataFrame:
    """Per-model background with simple imputation, then summarized to <= target_rows."""
    cols = [c for c in feature_list if c in df_raw.columns]
    if not cols:
        return pd.DataFrame(columns=feature_list)

    bg = df_raw[cols].copy()

    # Impute per column
    for c in cols:
        s = bg[c]
        if is_numeric_series(s):
            bg[c] = pd.to_numeric(s, errors='coerce')
            med = np.nanmedian(bg[c].values.astype(float))
            if not np.isfinite(med):
                med = 0.0
            bg[c] = bg[c].fillna(med)
        else:
            try:
                mode_val = s.mode(dropna=True).iloc[0]
            except Exception:
                mode_val = "missing"
            bg[c] = s.fillna(mode_val)

    # Summarize to at most target_rows (pre-sample) to keep memory bounded
    if len(bg) > target_rows:
        bg = bg.sample(n=target_rows, random_state=0).reset_index(drop=True)

    return bg

@st.cache_data(show_spinner=False)
def summarize_background(bg: pd.DataFrame, K: int = 32) -> pd.DataFrame:
    """Use shap.sample to reduce background to K rows (fast, works with mixed dtypes)."""
    if len(bg) <= K:
        return bg.reset_index(drop=True)
    return shap.sample(bg, K).reset_index(drop=True)

def compute_local_shap_logit_fast(
    predictor_obj,
    x_row_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    feature_map: dict,
    K_bg: int = 32,
    nsamples: int = 256
):
    """Stable single-patient SHAP on log-odds with small summarized background.
       Returns: shap_df, base_logit, pred_logit, base_prob, pred_prob
    """
    feature_names = list(x_row_df.columns)
    pos_label = get_positive_label(predictor_obj)

    bg_full = build_background(raw_df, feature_names, target_rows=500)
    bg = summarize_background(bg_full, K=K_bg)
    if len(bg) == 0:
        # As last resort, copy the row and jitter numeric cols a little
        bg = pd.DataFrame([x_row_df.iloc[0].to_dict() for _ in range(max(16, K_bg))])
        for c in feature_names:
            if is_numeric_series(pd.Series(bg[c])):
                bg[c] = pd.to_numeric(bg[c], errors='coerce').fillna(0.0) + np.random.normal(0, 1e-3, size=len(bg))

    fprob = proba_callable_for_shap(predictor_obj, feature_names, pos_label)
    explainer = shap.KernelExplainer(fprob, bg, link="logit")

    shap_vals = explainer.shap_values(x_row_df, nsamples=nsamples)
    svarr = np.array(shap_vals)
    if svarr.ndim == 3:
        phi = svarr[0, 0, :]
    elif svarr.ndim == 2:
        phi = svarr[0, :]
    else:
        phi = svarr.reshape(-1)

    base_logit = float(np.array(explainer.expected_value).reshape(-1)[-1])
    p_pred = float(np.clip(fprob(x_row_df.values)[0], 1e-8, 1 - 1e-8))
    pred_logit = float(np.log(p_pred / (1 - p_pred)))
    base_prob = float(1 / (1 + np.exp(-base_logit)))
    pred_prob = p_pred

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'value': [x_row_df.iloc[0][c] for c in feature_names],
        'shap_logit': phi.astype(float)
    })
    shap_df['abs_shap'] = shap_df['shap_logit'].abs()
    shap_df['feature_display'] = shap_df['feature'].map(lambda x: feature_map.get(x, x))
    shap_df.sort_values('abs_shap', ascending=False, inplace=True)

    # Degeneracy fallback
    if not np.isfinite(shap_df['abs_shap'].sum()) or shap_df['abs_shap'].sum() < 1e-8:
        base_logit_mean = float(np.log(np.clip(fprob(bg.values).mean(), 1e-8, 1-1e-8) /
                                       np.clip(1 - fprob(bg.values).mean(), 1e-8, 1-1e-8)))
        contribs = []
        for c in feature_names:
            bg_mod = bg.copy()
            bg_mod[c] = x_row_df.iloc[0][c]
            p_mean = float(np.clip(fprob(bg_mod.values).mean(), 1e-8, 1 - 1e-8))
            logit_mean = float(np.log(p_mean / (1 - p_mean)))
            contribs.append(logit_mean - base_logit_mean)
        shap_df['shap_logit'] = np.array(contribs, dtype=float)
        shap_df['abs_shap']  = np.abs(shap_df['shap_logit'])
        shap_df.sort_values('abs_shap', ascending=False, inplace=True)
        base_logit = base_logit_mean
        base_prob  = float(1 / (1 + np.exp(-base_logit)))

    return shap_df, base_logit, pred_logit, base_prob, pred_prob

# -------------------- Plot helpers (log-odds) --------------------
def plot_topk_bar_logit(ax, shap_df, top_k=8, title=""):
    top = shap_df.head(top_k).copy()
    labels = [f"{r.feature_display} = {r.value}" for r in top.itertuples(index=False)]
    y = np.arange(len(top))[::-1]
    vals = top['shap_logit'].values[::-1].astype(float)
    ax.barh(y, vals, align='center')
    ax.set_yticks(y, labels=labels[::-1], fontsize=8)
    ax.axvline(0, color='k', linewidth=0.6)
    ax.set_xlabel("SHAP (log-odds)")
    if title:
        ax.set_title(title, fontsize=12)
    vmax = np.nanmax(np.abs(vals)) if len(vals) else 0.0
    if np.isfinite(vmax) and vmax > 0:
        ax.set_xlim(-1.2*vmax, 1.2*vmax)

def draw_donut(ax, prob: float, title: str):
    prob = float(np.clip(prob, 0.0, 1.0))
    color = 'green' if prob < 0.33 else ('orange' if prob < 0.66 else 'red')
    ax.add_patch(Wedge((0,0), 1.0, 0, 360, width=0.3, facecolor='lightgray', edgecolor='none'))
    ax.add_patch(Wedge((0,0), 1.0, 90, 90 - 360*prob, width=0.3, facecolor=color, edgecolor='none'))
    ax.text(0, 0, f"{prob:.0%}", ha='center', va='center', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=12); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.axis('off')

def export_figure6(input_display_pairs,
                   risk_allcause, risk_cardio, risk_infection,
                   shap_allcause, shap_cardio, shap_infect,
                   outfile_png="figure6_multiplot.png",
                   outfile_svg="figure6_multiplot.svg"):
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1])

    axA = fig.add_subplot(gs[0, 0])
    cell_text, cell_colors = [], []
    for name, val, imputed in input_display_pairs:
        flag = "•" if imputed else ""
        cell_text.append([name, f"{val}{flag}"])
        cell_colors.append(['white', '#FFF4E6' if imputed else 'white'])
    table = axA.table(cellText=cell_text, colLabels=["Variable", "Value"],
                      cellLoc='left', colLoc='left',
                      cellColours=cell_colors, colColours=['#F2F2F2','#F2F2F2'],
                      loc='center')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.2)
    axA.axis('off'); axA.set_title("Entered inputs (• imputed)", fontsize=12)
    axA.text(-0.08, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight='bold', va='top')

    axB = fig.add_subplot(gs[0, 1]); draw_donut(axB, risk_allcause, "All‑cause mortality")
    axB.text(-0.08, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight='bold', va='top')
    axC = fig.add_subplot(gs[0, 2]); draw_donut(axC, risk_cardio, "Cardiovascular mortality")
    axC.text(-0.08, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight='bold', va='top')
    axD = fig.add_subplot(gs[1, 1]); draw_donut(axD, risk_infection, "Infection‑related mortality")
    axD.text(-0.08, 1.05, "D", transform=axD.transAxes, fontsize=14, fontweight='bold', va='top')

    axE = fig.add_subplot(gs[1:, 0]); plot_topk_bar_logit(axE, shap_allcause, top_k=8, title="All‑cause: top contributors")
    axE.text(-0.08, 1.05, "E", transform=axE.transAxes, fontsize=14, fontweight='bold', va='top')
    axF = fig.add_subplot(gs[1:, 1]); plot_topk_bar_logit(axF, shap_cardio,   top_k=8, title="Cardiovascular: top contributors")
    axF.text(-0.08, 1.05, "F", transform=axF.transAxes, fontsize=14, fontweight='bold', va='top')
    axG = fig.add_subplot(gs[1:, 2]); plot_topk_bar_logit(axG, shap_infect,   top_k=8, title="Infection‑related: top contributors")
    axG.text(-0.08, 1.05, "G", transform=axG.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(outfile_png, bbox_inches='tight')
    if outfile_svg is not None:
        fig.savefig(outfile_svg, bbox_inches='tight')
    return outfile_png, outfile_svg

def plot_custom_waterfall_logit(ax, shap_df, base_logit, pred_logit, base_prob, pred_prob, top_k=10, title=""):
    top = shap_df.head(top_k).copy().sort_values('shap_logit')
    phi = top['shap_logit'].values
    names = top['feature_display'].values
    starts = [base_logit]
    for s in phi[:-1]:
        starts.append(starts[-1] + s)
    starts = np.array(starts)
    widths = np.abs(phi)
    lefts = np.where(phi >= 0, starts, starts + phi)
    y = np.arange(len(top))
    colors = ['red' if s > 0 else 'blue' for s in phi]
    ax.barh(y, widths, left=lefts, color=colors, align='center', edgecolor='none')
    ax.set_yticks(y, labels=[str(n) for n in names], fontsize=8)
    ax.axvline(base_logit, color='gray', linestyle='--', linewidth=0.8, label=f'base (p={base_prob:.2f})')
    ax.axvline(pred_logit, color='black', linestyle='-', linewidth=1.0, label=f'pred (p={pred_prob:.2f})')
    ax.set_xlabel("Log-odds")
    if title: ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    xmin = min(base_logit, pred_logit, lefts.min()) - 0.25
    xmax = max(base_logit, pred_logit, (lefts + widths).max()) + 0.25
    ax.set_xlim(xmin, xmax)

# -------------------- Prediction & Explanation --------------------
if st.button('Predict'):
    # Build input row
    input_df = pd.DataFrame([input_data])

    # Impute + track
    missing_values_used = {}
    for feature in feature_names_list_ordered:
        if feature not in input_df.columns:
            input_df[feature] = np.nan
        if pd.isnull(input_df.loc[0, feature]) or input_df.loc[0, feature] == '':
            ftype = feature_types.get(feature, 'float')
            if ftype in ['int', 'float']:
                med = pd.to_numeric(emer[feature], errors='coerce').median()
                if not np.isfinite(med): med = 0.0
                input_df.loc[0, feature] = float(med)
            elif ftype == 'object':
                try:
                    mode_val = emer[feature].mode(dropna=True).iloc[0]
                except Exception:
                    mode_val = "missing"
                input_df.loc[0, feature] = mode_val
            else:
                input_df.loc[0, feature] = None
            missing_values_used[display_names.get(feature, feature)] = input_df.loc[0, feature]

    # Normalize datetime
    for feat, ftype in feature_types.items():
        if ftype == 'datetime' and feat in input_df.columns:
            input_df[feat] = pd.to_datetime(input_df[feat])

    # Predictions FIRST (so they never disappear)
    def _pick_pos(proba_df, predictor_obj):
        pos = get_positive_label(predictor_obj)
        if isinstance(proba_df, pd.DataFrame):
            if pos in proba_df.columns:      return float(proba_df.iloc[0][pos])
            if str(pos) in proba_df.columns: return float(proba_df.iloc[0][str(pos)])
            if 1 in proba_df.columns:        return float(proba_df.iloc[0][1])
            if '1' in proba_df.columns:      return float(proba_df.iloc[0]['1'])
            return float(proba_df.iloc[0, -1])
        arr = np.asarray(proba_df)
        if arr.ndim == 2 and arr.shape[1] >= 2: return float(arr[0, 1])
        return float(arr[0])

    pred_allcause = predictor.predict(input_df)
    prob_allcause = predictor.predict_proba(input_df)
    risk_allcause = _pick_pos(prob_allcause, predictor)

    pred_cardio   = card_predictor.predict(input_df)
    prob_cardio   = card_predictor.predict_proba(input_df)
    risk_cardio   = _pick_pos(prob_cardio, card_predictor)

    pred_infect   = sepsis_predictor.predict(input_df)
    prob_infect   = sepsis_predictor.predict_proba(input_df)
    risk_infect   = _pick_pos(prob_infect, sepsis_predictor)

    # Persist predictions in session_state to avoid “vanish” on reruns
    st.session_state["risks"] = dict(allcause=risk_allcause, cardio=risk_cardio, infect=risk_infect)
    st.session_state["pred_texts"] = dict(
        allcause=('Low risk of mortality' if pred_allcause.iloc[0] == 0 else 'Elevated mortality risk, requiring intervention'),
        cardio=('Low risk of cardiovascular death' if pred_cardio.iloc[0] == 0 else 'Elevated cardiovascular mortality risk, requiring intervention'),
        infect=('Low risk of infection-related mortality' if pred_infect.iloc[0] == 0 else 'Elevated infection-related mortality risk, requiring intervention')
    )
    st.session_state["input_df"] = input_df
    st.session_state["missing_values_used"] = missing_values_used

    # Show predictions immediately
    st.subheader('Prediction Results')
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### All-cause Mortality")
        st.write(f"**Prediction:** {st.session_state['pred_texts']['allcause']}")
        st.plotly_chart(create_ring_plot(st.session_state['risks']['allcause'], "Probability of Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of mortality: {st.session_state['risks']['allcause']:.2%}")
    with cols[1]:
        st.markdown("### Cardiovascular Mortality")
        st.write(f"**Prediction:** {st.session_state['pred_texts']['cardio']}")
        st.plotly_chart(create_ring_plot(st.session_state['risks']['cardio'], "Probability of Cardiovascular Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of Cardiovascular mortality: {st.session_state['risks']['cardio']:.2%}")
    with cols[2]:
        st.markdown("### Infection-related Mortality")
        st.write(f"**Prediction:** {st.session_state['pred_texts']['infect']}")
        st.plotly_chart(create_ring_plot(st.session_state['risks']['infect'], "Probability of Infection-related Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of Infection-related mortality: {st.session_state['risks']['infect']:.2%}")

    if len(missing_values_used) > 0:
        st.warning("The following variables were missing and were imputed for this run:")
        for k, v in missing_values_used.items():
            st.write(f"{k}: {v}")

    # -------------------- Patient-specific SHAP (fast & stable) --------------------
    st.markdown("### Patient‑specific feature contributions (SHAP)")
    st.caption("Bars show SHAP values in log‑odds (positive increases risk; negative decreases risk).")

    allcause_feats = [f for f in predictor.features() if f in emer.columns]
    cardio_feats   = [f for f in card_predictor.features() if f in emer.columns]
    infect_feats   = [f for f in sepsis_predictor.features() if f in emer.columns]

    x_allcause = input_df[allcause_feats].copy()
    x_cardio   = input_df[cardio_feats].copy()
    x_infect   = input_df[infect_feats].copy()

    # Compute explanations with spinner and guards
    with st.spinner("Computing patient‑level explanations..."):
        try:
            shap_allcause_df, base_allcause_logit, pred_allcause_logit, base_allcause_p, fx_allcause_p = compute_local_shap_logit_fast(
                predictor, x_allcause, emer, display_names, K_bg=32, nsamples=256
            )
        except Exception as e:
            st.error(f"All‑cause SHAP failed: {e}")
            shap_allcause_df = pd.DataFrame(columns=['feature_display','value','shap_logit','abs_shap'])
            base_allcause_logit = pred_allcause_logit = 0.0; base_allcause_p = fx_allcause_p = st.session_state['risks']['allcause']

        try:
            shap_cardio_df, base_cardio_logit, pred_cardio_logit, base_cardio_p, fx_cardio_p = compute_local_shap_logit_fast(
                card_predictor, x_cardio, emer, display_names, K_bg=32, nsamples=256
            )
        except Exception as e:
            st.error(f"Cardiovascular SHAP failed: {e}")
            shap_cardio_df = pd.DataFrame(columns=['feature_display','value','shap_logit','abs_shap'])
            base_cardio_logit = pred_cardio_logit = 0.0; base_cardio_p = fx_cardio_p = st.session_state['risks']['cardio']

        try:
            shap_infect_df, base_infect_logit, pred_infect_logit, base_infect_p, fx_infect_p = compute_local_shap_logit_fast(
                sepsis_predictor, x_infect, emer, display_names, K_bg=32, nsamples=256
            )
        except Exception as e:
            st.error(f"Infection‑related SHAP failed: {e}")
            shap_infect_df = pd.DataFrame(columns=['feature_display','value','shap_logit','abs_shap'])
            base_infect_logit = pred_infect_logit = 0.0; base_infect_p = fx_infect_p = st.session_state['risks']['infect']

    # Show three bar plots + tables
    ecol = st.columns(3)
    def _fmt_table(df):
        out = df[['feature_display','value','shap_logit']].head(8).rename(
            columns={'feature_display':'feature','shap_logit':'contribution (log-odds)'}
        ).copy()
        if len(out) > 0:
            out['contribution (log-odds)'] = out['contribution (log-odds)'].astype(float).round(3)
        return out

    with ecol[0]:
        st.markdown("**All‑cause: top drivers**")
        figA, axA = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar_logit(axA, shap_allcause_df, top_k=8, title="")
        st.pyplot(figA, clear_figure=True)
        st.dataframe(_fmt_table(shap_allcause_df), use_container_width=True)
    with ecol[1]:
        st.markdown("**Cardiovascular: top drivers**")
        figB, axB = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar_logit(axB, shap_cardio_df, top_k=8, title="")
        st.pyplot(figB, clear_figure=True)
        st.dataframe(_fmt_table(shap_cardio_df), use_container_width=True)
    with ecol[2]:
        st.markdown("**Infection‑related: top drivers**")
        figC, axC = plt.subplots(figsize=(4, 3), dpi=150)
        plot_topk_bar_logit(axC, shap_infect_df, top_k=8, title="")
        st.pyplot(figC, clear_figure=True)
        st.dataframe(_fmt_table(shap_infect_df), use_container_width=True)

    # Waterfalls (log-odds)
    with st.expander("Show SHAP waterfall plots (per‑patient)"):
        wcols = st.columns(3)
        with wcols[0]:
            st.markdown(f"All‑cause waterfall (base p={base_allcause_p:.2f} → pred p={fx_allcause_p:.2f})")
            figW, axW = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall_logit(axW, shap_allcause_df, base_allcause_logit, pred_allcause_logit, base_allcause_p, fx_allcause_p, top_k=10, title="")
            st.pyplot(figW, clear_figure=True)
        with wcols[1]:
            st.markdown(f"Cardiovascular waterfall (base p={base_cardio_p:.2f} → pred p={fx_cardio_p:.2f})")
            figW2, axW2 = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall_logit(axW2, shap_cardio_df, base_cardio_logit, pred_cardio_logit, base_cardio_p, fx_cardio_p, top_k=10, title="")
            st.pyplot(figW2, clear_figure=True)
        with wcols[2]:
            st.markdown(f"Infection‑related waterfall (base p={base_infect_p:.2f} → pred p={fx_infect_p:.2f})")
            figW3, axW3 = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall_logit(axW3, shap_infect_df, base_infect_logit, pred_infect_logit, base_infect_p, fx_infect_p, top_k=10, title="")
            st.pyplot(figW3, clear_figure=True)

    # Export camera‑ready Figure 6
    ordered_display_pairs = []
    for feat in feature_names_list_ordered:
        disp = display_names.get(feat, feat)
        val = input_df.loc[0, feat]
        is_imp = disp in st.session_state["missing_values_used"]
        vs = f"{val:.3g}" if isinstance(val, float) else str(val)
        ordered_display_pairs.append((disp, vs, is_imp))

    if st.button("Export camera‑ready Figure 6 (PNG + SVG)"):
        png_path, svg_path = export_figure6(
            input_display_pairs=ordered_display_pairs,
            risk_allcause=float(st.session_state['risks']['allcause']),
            risk_cardio=float(st.session_state['risks']['cardio']),
            risk_infection=float(st.session_state['risks']['infect']),
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
