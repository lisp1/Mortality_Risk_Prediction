# streamlit_app.py
# ------------------------------------------------------------
# Mortality prediction + per-patient interpretability (stable SHAP)
# Fundamental fixes:
#  - Robust background builder per model (imputation + dtype harmonization)
#  - KernelExplainer(link="logit") with sufficient nsamples
#  - Degeneracy guard: local log-odds effects fallback if SHAP≈0
#  - Custom log-odds bar + waterfall plots (no blank images)
#  - Camera-ready Figure 6 exporter retained
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
matplotlib.use("Agg")  # safe for Streamlit
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

# ---------- Do NOT drop all-NaN rows globally (kept for reference only) ----------
# Previous line caused background collapse:
# emer_clean = emer.dropna(how='any', inplace=False)  # <- removes most rows when 100+ cols
# Instead keep the raw DF; we will build per-model backgrounds with imputation.

# ---------- Feature sets and types ----------
allcause_features = set(predictor.feature_metadata.get_features())
card_features     = set(card_predictor.feature_metadata.get_features())
sepsis_features   = set(sepsis_predictor.feature_metadata.get_features())
all_features      = allcause_features.union(card_features).union(sepsis_features)
feature_names_list = list(all_features)

# raw type maps from predictors (union)
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
    display_names.setdefault(f, f)

# ---------- Bulk input order & UI ----------
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

# ---------- Gauge plot ----------
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

# ---------- Input widgets ----------
input_data = {}
columns_per_row = 6
rows = (num_features + columns_per_row - 1) // columns_per_row

# Parse bulk input (if provided)
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

# ---------- SHAP helpers: robust background & logit explanations ----------
def get_positive_label(predictor_obj):
    try:
        labels = predictor_obj.class_labels
        if labels is not None and len(labels) == 2:
            return labels[1]
    except Exception:
        pass
    return 1

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s)

def build_background(raw_df: pd.DataFrame,
                     feature_list,
                     target_size: int = 200) -> pd.DataFrame:
    """Per-model background with imputation & dtype harmonization.
       Never returns 0 rows; if needed, synthesize jittered samples."""
    df = raw_df.copy()
    cols = [c for c in feature_list if c in df.columns]
    if len(cols) == 0:
        # if predictor expects features not in emer, synthesize from current inputs later
        return pd.DataFrame(columns=feature_list)

    bg = df[cols].copy()

    # Impute per column (median for numeric; mode/constant for non-numeric)
    for c in cols:
        s = bg[c]
        if is_numeric_series(s):
            # coerce numeric
            bg[c] = pd.to_numeric(s, errors='coerce')
            med = np.nanmedian(bg[c].values.astype(float)) if np.isfinite(np.nanmedian(pd.to_numeric(s, errors='coerce'))).any() else 0.0
            if not np.isfinite(med):
                med = 0.0
            bg[c] = bg[c].fillna(med)
        else:
            # treat as category/text
            try:
                mode_val = s.mode(dropna=True).iloc[0]
            except Exception:
                mode_val = "missing"
            bg[c] = s.fillna(mode_val).astype(str)

    # Drop any residual NaNs and sample
    if len(bg) == 0:
        return pd.DataFrame(columns=feature_list)

    # De-duplicate and sample
    bg = bg.drop_duplicates()
    if len(bg) > target_size:
        bg = bg.sample(n=target_size, random_state=0)
    # Ensure we keep columns order exactly as feature_list
    bg = bg[[c for c in feature_list if c in bg.columns]]

    return bg.reset_index(drop=True)

def ensure_background(bg: pd.DataFrame,
                      x_row_df: pd.DataFrame,
                      raw_df: pd.DataFrame,
                      feature_list,
                      min_rows: int = 50) -> pd.DataFrame:
    """Guarantee a non-degenerate background >= min_rows.
       If not enough, synthesize jittered rows around real distribution."""
    # Rebuild if empty
    if bg is None or len(bg) == 0:
        bg = build_background(raw_df, feature_list, target_size=min_rows)

    # If still too small, synthesize jitter from raw_df stats or from x_row_df
    if len(bg) < min_rows:
        needed = min_rows - len(bg)
        synth = pd.DataFrame(columns=feature_list)
        for c in feature_list:
            if c not in raw_df.columns:
                # fallback from x value
                base = x_row_df.iloc[0][c]
                if isinstance(base, (int, float, np.number)):
                    synth[c] = base + np.random.normal(0, 1e-3, size=needed)
                else:
                    synth[c] = [str(base)] * needed
                continue

            s = raw_df[c]
            if is_numeric_series(s):
                arr = pd.to_numeric(s, errors='coerce').dropna().values
                if len(arr) >= 5:
                    mu, sd = np.nanmedian(arr), np.nanstd(arr)
                    sd = 1.0 if not np.isfinite(sd) or sd == 0 else sd
                    synth[c] = np.random.normal(mu, 0.3*sd, size=needed)
                else:
                    base = x_row_df.iloc[0][c]
                    base = float(base) if isinstance(base, (int, float, np.number)) else 0.0
                    synth[c] = base + np.random.normal(0, 1.0, size=needed)
            else:
                vals = s.dropna().astype(str).unique()
                if len(vals) == 0:
                    vals = [str(x_row_df.iloc[0][c])]
                synth[c] = np.random.choice(vals, size=needed)
        bg = pd.concat([bg, synth], axis=0, ignore_index=True)

    # Column order
    bg = bg[[c for c in feature_list if c in bg.columns]]
    return bg

def proba_callable_for_shap(predictor_obj, feature_names, positive_label):
    """f(X) -> P(positive) 1D array"""
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

def compute_local_shap_logit(
    predictor_obj,
    x_row_df: pd.DataFrame,
    bg_df_raw: pd.DataFrame,
    raw_df: pd.DataFrame,
    feature_map: dict,
    min_bg_rows: int = 50
):
    """Stable single-patient SHAP on log-odds with degeneracy guards."""
    feature_names = list(x_row_df.columns)
    pos_label = get_positive_label(predictor_obj)
    # Build / ensure robust background
    bg = build_background(bg_df_raw if bg_df_raw is not None else raw_df, feature_names, target_size=200)
    bg = ensure_background(bg, x_row_df, raw_df, feature_names, min_rows=min_bg_rows)

    # KernelExplainer on LOGIT link
    fprob = proba_callable_for_shap(predictor_obj, feature_names, pos_label)
    explainer = shap.KernelExplainer(fprob, bg, link="logit")

    nsamples = max(512, 2 * (len(feature_names) ** 2))
    shap_vals = explainer.shap_values(x_row_df, nsamples=nsamples)

    svarr = np.array(shap_vals)
    if svarr.ndim == 3:      # (n_outputs, n_rows, n_features)
        phi = svarr[0, 0, :]
    elif svarr.ndim == 2:    # (n_rows, n_features)
        phi = svarr[0, :]
    else:                    # (n_features,)
        phi = svarr.reshape(-1)

    # base & pred in logit space
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

    # Degeneracy guard: if sum|phi|≈0, compute local effects as fallback
    if not np.isfinite(shap_df['abs_shap'].sum()) or shap_df['abs_shap'].sum() < 1e-8:
        # Local effects on log-odds: mean logit difference when setting feature to x vs background
        # (independent-feature approximation of SHAP)
        base_logit_mean = float(np.log(np.clip(fprob(bg.values).mean(), 1e-8, 1-1e-8) / np.clip(1 - fprob(bg.values).mean(), 1e-8, 1-1e-8)))
        contribs = []
        for j, c in enumerate(feature_names):
            bg_mod = bg.copy()
            bg_mod[c] = x_row_df.iloc[0][c]  # set feature to patient's value
            p_mean = float(np.clip(fprob(bg_mod.values).mean(), 1e-8, 1 - 1e-8))
            logit_mean = float(np.log(p_mean / (1 - p_mean)))
            contribs.append(logit_mean - base_logit_mean)
        shap_df['shap_logit'] = np.array(contribs, dtype=float)
        shap_df['abs_shap'] = np.abs(shap_df['shap_logit'])
        shap_df.sort_values('abs_shap', ascending=False, inplace=True)
        # Recompute base/pred annotations from actual prediction
        base_logit = base_logit_mean
        base_prob  = float(1 / (1 + np.exp(-base_logit)))

    return shap_df, base_logit, pred_logit, base_prob, pred_prob

# ---------- Plotting helpers (log-odds) ----------
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
    # If values are very small, widen x-limits a bit so bars are visible
    vmax = np.nanmax(np.abs(vals)) if len(vals) else 0.0
    if np.isfinite(vmax) and vmax > 0:
        ax.set_xlim(-1.2*vmax, 1.2*vmax)

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
    """A: inputs; B–D: donuts; E–G: SHAP bars (log-odds)."""
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1])

    # Panel A
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

    # E–G bars
    axE = fig.add_subplot(gs[1:, 0]); plot_topk_bar_logit(axE, shap_allcause, top_k=8, title="All‑cause: top contributors")
    axE.text(-0.08, 1.05, "E", transform=axE.transAxes, fontsize=14, fontweight='bold', va='top')
    axF = fig.add_subplot(gs[1:, 1]); plot_topk_bar_logit(axF, shap_cardio, top_k=8, title="Cardiovascular: top contributors")
    axF.text(-0.08, 1.05, "F", transform=axF.transAxes, fontsize=14, fontweight='bold', va='top')
    axG = fig.add_subplot(gs[1:, 2]); plot_topk_bar_logit(axG, shap_infect, top_k=8, title="Infection‑related: top contributors")
    axG.text(-0.08, 1.05, "G", transform=axG.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.tight_layout()
    fig.savefig(outfile_png, bbox_inches='tight')
    if outfile_svg is not None:
        fig.savefig(outfile_svg, bbox_inches='tight')
    return outfile_png, outfile_svg

def plot_custom_waterfall_logit(ax, shap_df, base_logit, pred_logit, base_prob, pred_prob,
                                top_k=10, title=""):
    """Probability model explained in log-odds. Bars stack from base → pred."""
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
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=8)
    xmin = min(base_logit, pred_logit, lefts.min()) - 0.25
    xmax = max(base_logit, pred_logit, (lefts + widths).max()) + 0.25
    ax.set_xlim(xmin, xmax)

# ---------- Predictions + explanations ----------
if st.button('Predict'):
    # Build input row
    input_df = pd.DataFrame([input_data])

    # Impute missing values and record
    missing_features = []
    missing_values_used = {}
    for feature in feature_names_list_ordered:
        if feature not in input_df.columns:
            input_df[feature] = np.nan
        if pd.isnull(input_df.loc[0, feature]) or input_df.loc[0, feature] == '':
            ftype = feature_types.get(feature, 'float')
            if ftype in ['int', 'float']:
                # robust impute from emer (median)
                med = pd.to_numeric(emer[feature], errors='coerce').median()
                if not np.isfinite(med):
                    med = 0.0
                input_df.loc[0, feature] = float(med)
            elif ftype == 'object':
                try:
                    mode_value = emer[feature].mode(dropna=True).iloc[0]
                except Exception:
                    mode_value = "missing"
                input_df.loc[0, feature] = mode_value
            else:
                input_df.loc[0, feature] = None
            missing_features.append(display_names.get(feature, feature))
            missing_values_used[display_names.get(feature, feature)] = input_df.loc[0, feature]

    # Normalize datetime
    for feature, ftype in feature_types.items():
        if ftype == 'datetime' and feature in input_df.columns:
            input_df[feature] = pd.to_datetime(input_df[feature])

    # Predictions
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

    risk_allcause = _pick_pos(probability, predictor)
    risk_cardio   = _pick_pos(card_probability, card_predictor)
    risk_infect   = _pick_pos(sepsis_probability, sepsis_predictor)

    # Display results
    st.subheader('Prediction Results')
    cols = st.columns(3)
    with cols[0]:
        st.markdown("### All-cause Mortality")
        prediction_text = 'Low risk of mortality' if prediction.iloc[0] == 0 else 'Elevated mortality risk, requiring intervention'
        st.write(f"**Prediction:** {prediction_text}")
        st.plotly_chart(create_ring_plot(risk_allcause, "Probability of Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of mortality: {risk_allcause:.2%}")
    with cols[1]:
        st.markdown("### Cardiovascular Mortality")
        card_text = 'Low risk of cardiovascular death' if card_prediction.iloc[0] == 0 else 'Elevated cardiovascular mortality risk, requiring intervention'
        st.write(f"**Prediction:** {card_text}")
        st.plotly_chart(create_ring_plot(risk_cardio, "Probability of Cardiovascular Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of Cardiovascular mortality: {risk_cardio:.2%}")
    with cols[2]:
        st.markdown("### Infection-related Mortality")
        sepsis_text = 'Low risk of infection-related mortality' if sepsis_prediction.iloc[0] == 0 else 'Elevated infection-related mortality risk, requiring intervention'
        st.write(f"**Prediction:** {sepsis_text}")
        st.plotly_chart(create_ring_plot(risk_infect, "Probability of Infection-related Mortality"), use_container_width=True)
        st.write(f"Predicted Risk of Infection-related mortality: {risk_infect:.2%}")

    # Imputation transparency
    if len(missing_features) > 0:
        st.warning("The following variables were missing and were imputed for this run:")
        for var in missing_features:
            st.write(f"{var}: {missing_values_used[var]}")

    # ---------- Per-patient SHAP (log-odds) with robust background ----------
    st.markdown("### Patient‑specific feature contributions (SHAP)")
    st.caption("Bars show SHAP values in log‑odds (positive increases risk; negative decreases risk).")

    # Per-model feature subsets
    allcause_feats = [f for f in predictor.features() if f in emer.columns]
    cardio_feats   = [f for f in card_predictor.features() if f in emer.columns]
    infect_feats   = [f for f in sepsis_predictor.features() if f in emer.columns]

    x_allcause = input_df[allcause_feats].copy()
    x_cardio   = input_df[cardio_feats].copy()
    x_infect   = input_df[infect_feats].copy()

    # Raw per-model backgrounds (before robustification)
    bg_allcause_raw = emer[allcause_feats] if set(allcause_feats).issubset(emer.columns) else None
    bg_cardio_raw   = emer[cardio_feats]   if set(cardio_feats).issubset(emer.columns)   else None
    bg_infect_raw   = emer[infect_feats]   if set(infect_feats).issubset(emer.columns)   else None

    shap_allcause_df, base_allcause_logit, pred_allcause_logit, base_allcause_p, fx_allcause_p = compute_local_shap_logit(
        predictor, x_allcause, bg_allcause_raw, emer, display_names, min_bg_rows=80
    )
    shap_cardio_df, base_cardio_logit, pred_cardio_logit, base_cardio_p, fx_cardio_p = compute_local_shap_logit(
        card_predictor, x_cardio, bg_cardio_raw, emer, display_names, min_bg_rows=80
    )
    shap_infect_df, base_infect_logit, pred_infect_logit, base_infect_p, fx_infect_p = compute_local_shap_logit(
        sepsis_predictor, x_infect, bg_infect_raw, emer, display_names, min_bg_rows=80
    )

    # Three columns with bars + tables
    ecol = st.columns(3)

    def _fmt_table(df):
        out = df[['feature_display','value','shap_logit']].head(8).rename(
            columns={'feature_display':'feature','shap_logit':'contribution (log-odds)'}
        ).copy()
        out['contribution (log-odds)'] = out['contribution (log-odds)'].astype(float).round(4)
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

    # Custom waterfalls (log-odds) – always visible
    with st.expander("Show SHAP waterfall plots (per‑patient)"):
        wcols = st.columns(3)
        with wcols[0]:
            st.markdown(f"All‑cause waterfall (base p={base_allcause_p:.2f} → pred p={fx_allcause_p:.2f})")
            figW, axW = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall_logit(axW, shap_allcause_df, base_allcause_logit, pred_allcause_logit,
                                        base_allcause_p, fx_allcause_p, top_k=10, title="")
            st.pyplot(figW, clear_figure=True)
        with wcols[1]:
            st.markdown(f"Cardiovascular waterfall (base p={base_cardio_p:.2f} → pred p={fx_cardio_p:.2f})")
            figW2, axW2 = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall_logit(axW2, shap_cardio_df, base_cardio_logit, pred_cardio_logit,
                                        base_cardio_p, fx_cardio_p, top_k=10, title="")
            st.pyplot(figW2, clear_figure=True)
        with wcols[2]:
            st.markdown(f"Infection‑related waterfall (base p={base_infect_p:.2f} → pred p={fx_infect_p:.2f})")
            figW3, axW3 = plt.subplots(figsize=(5, 3), dpi=150)
            plot_custom_waterfall_logit(axW3, shap_infect_df, base_infect_logit, pred_infect_logit,
                                        base_infect_p, fx_infect_p, top_k=10, title="")
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
            risk_allcause=float(risk_allcause),
            risk_cardio=float(risk_cardio),
            risk_infection=float(risk_infect),
            shap_allcause=shap_allcause_df,
            shap_cardio=shap_cardio_df,
            shap_infect=shap_infect_df,
            outfile_png="figure6_multiplot.png",
            outfile_svg="figure6_multiplot.svg",
        )
        with open(png_path, "rb") as f:
            st.download_button("Download Figure 6 (PNG)", data=f, file_name="Figure6.png", mime="image/png")
        with open(svg_path, "RB") as f:
            st.download_button("Download Figure 6 (SVG)", data=f, file_name="Figure6.svg", mime="image/svg+xml")
