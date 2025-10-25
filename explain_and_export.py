# explain_and_export.py
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Wedge, Circle
from typing import Dict, Tuple, Optional, List

def _ensure_2d_frame(x: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for SHAP wrappers.")
    return x

def _proba_wrapper_for_shap(predictor, feature_names: List[str]):
    """
    Returns a function f(X) -> P(class=1) suitable for SHAP.
    X will arrive as a numpy array; convert to DataFrame with given columns.
    """
    def f(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        # Predictor may expect raw features; ensure consistent interface
        proba = predictor.predict_proba(X_df)
        # proba can be a DataFrame (columns [0,1]) or ndarray; get P(class=1)
        if isinstance(proba, pd.DataFrame):
            if 1 in proba.columns:
                return proba[[1]].values
            # Some models may name columns like '1'
            if '1' in proba.columns:
                return proba[['1']].values
            # Fallback to last column
            return proba.iloc[:, [-1]].values
        else:
            # ndarray of shape (n,2) or (n,)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, [1]]
            return proba.reshape(-1, 1)
    return f

def compute_local_shap(
    predictor,
    x_row: pd.DataFrame,
    background_df: pd.DataFrame,
    prefers_tree: bool = True,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Compute local SHAP for a single row (class=1).
    Returns:
      shap_df: DataFrame with columns ['feature','value','shap','abs_shap','direction']
      base_value: SHAP base value (float)
      fx: predicted probability for class=1 (float)
    """
    x_row = _ensure_2d_frame(x_row)
    feature_names = list(x_row.columns)
    # SHAP masker (independent) from representative background
    masker = shap.maskers.Independent(background_df[feature_names], max_samples=min(200, len(background_df)))

    # Prefer TreeExplainer if model is tree-based; else fall back to model-agnostic Explainer
    explainer = None
    if prefers_tree:
        try:
            explainer = shap.TreeExplainer(predictor._trainer.load_base_model(predictor._trainer.get_model_best()))  # may fail for non-tree/stacked
        except Exception:
            explainer = None

    if explainer is None:
        f = _proba_wrapper_for_shap(predictor, feature_names)
        explainer = shap.Explainer(f, masker)  # Kernel/Partition/Permutation chosen by SHAP

    exp = explainer(x_row)   # returns shap.Explanation
    # SHAP returns values for outputs; pick the last / positive class where applicable
    shap_values = exp.values
    base_values = exp.base_values
    # If multioutput, pick last dimension as class=1
    if isinstance(shap_values, list):
        # SHAP older API may return a list; choose last
        sv = shap_values[-1]
        bv = base_values[-1] if isinstance(base_values, list) else base_values
    else:
        sv = shap_values
        bv = base_values

    # `sv` shape -> (1, n_features) or (1, n_features, 1)
    sv = np.array(sv).reshape(1, -1)
    bv = float(np.array(bv).reshape(-1)[-1])

    # Compute predicted probability for class 1
    f = _proba_wrapper_for_shap(predictor, feature_names)
    fx = float(f(x_row.values)[0, 0])

    shap_df = pd.DataFrame({
        'feature': feature_names,
        'value': [x_row.iloc[0, i] for i in range(len(feature_names))],
        'shap': sv[0, :],
    })
    shap_df['abs_shap'] = shap_df['shap'].abs()
    shap_df['direction'] = np.where(shap_df['shap'] >= 0, '+', '−')
    shap_df.sort_values('abs_shap', ascending=False, inplace=True)
    return shap_df, bv, fx

def _draw_donut(ax, prob: float, title: str):
    """
    Draw a ring (donut) representing probability 0..1
    """
    prob = float(np.clip(prob, 0.0, 1.0))
    # Color by risk
    color = 'green' if prob < 0.33 else ('orange' if prob < 0.66 else 'red')
    # Background ring
    ax.add_patch(Wedge((0,0), 1.0, 0, 360, width=0.3, facecolor='lightgray', edgecolor='none'))
    # Foreground arc
    ax.add_patch(Wedge((0,0), 1.0, 90, 90 - 360*prob, width=0.3, facecolor=color, edgecolor='none'))
    # Center text
    ax.text(0, 0, f"{prob:.0%}", ha='center', va='center', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.axis('off')

def _bar_patient_shap(ax, shap_df: pd.DataFrame, top_k: int = 8, title: str = ""):
    """
    Horizontal bar plot for top |SHAP| contributors (signed).
    Right (+) increases risk; left (−) decreases risk.
    """
    top = shap_df.head(top_k).copy()
    # Display name includes value for readability
    labels = [f"{r.feature} = {r.value}" for r in top.itertuples(index=False)]
    y = np.arange(len(top))
    ax.barh(y, top['shap'], align='center')
    ax.set_yticks(y, labels=labels, fontsize=8)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel("SHAP contribution to P(death)")
    if title:
        ax.set_title(title, fontsize=12)
    ax.invert_yaxis()

def _table_inputs(ax, ordered_pairs: List[Tuple[str, str, bool]]):
    """
    Render a table of display_name -> value, flagging imputed items.
    ordered_pairs: list of (display_name, value_str, is_imputed)
    """
    # Build rows
    cell_text = []
    cell_colors = []
    for name, val, imp in ordered_pairs:
        flag = "•" if imp else ""
        cell_text.append([name, f"{val}{flag}"])
        if imp:
            cell_colors.append(['white', '#FFF4E6'])  # subtle highlight for imputed
        else:
            cell_colors.append(['white', 'white'])
    table = ax.table(cellText=cell_text,
                     colLabels=["Variable", "Value"],
                     cellLoc='left', colLoc='left',
                     colColours=['#F2F2F2', '#F2F2F2'],
                     cellColours=cell_colors,
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    ax.axis('off')
    ax.set_title("Entered inputs (• imputed)", fontsize=12)

def export_figure6(
    input_display_pairs: List[Tuple[str, str, bool]],
    risk_allcause: float,
    risk_cardio: float,
    risk_infection: float,
    shap_allcause: pd.DataFrame,
    shap_cardio: pd.DataFrame,
    shap_infect: pd.DataFrame,
    outfile_png: str = "figure6_multiplot.png",
    outfile_svg: Optional[str] = "figure6_multiplot.svg",
):
    """
    Build a 2x3 (or 3x3) grid figure: A inputs, B–D donuts, E–G SHAP bars.
    Saves to PNG (and SVG if requested). Returns output paths.
    """
    plt.close('all')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    gs = gridspec.GridSpec(3, 3, height_ratios=[1,1,1])

    # Panel A: Inputs table
    axA = fig.add_subplot(gs[0, 0])
    _table_inputs(axA, input_display_pairs)
    axA.text(-0.08, 1.05, "A", transform=axA.transAxes, fontsize=14, fontweight='bold', va='top')

    # Panels B–D: Donuts
    axB = fig.add_subplot(gs[0, 1])
    _draw_donut(axB, risk_allcause, "All‑cause mortality")
    axB.text(-0.08, 1.05, "B", transform=axB.transAxes, fontsize=14, fontweight='bold', va='top')

    axC = fig.add_subplot(gs[0, 2])
    _draw_donut(axC, risk_cardio, "Cardiovascular mortality")
    axC.text(-0.08, 1.05, "C", transform=axC.transAxes, fontsize=14, fontweight='bold', va='top')

    axD = fig.add_subplot(gs[1, 1])
    _draw_donut(axD, risk_infection, "Infection‑related mortality")
    axD.text(-0.08, 1.05, "D", transform=axD.transAxes, fontsize=14, fontweight='bold', va='top')

    # Panels E–G: Patient‑specific SHAP
    axE = fig.add_subplot(gs[1:, 0])
    _bar_patient_shap(axE, shap_allcause, top_k=8, title="All‑cause: top contributors")
    axE.text(-0.08, 1.05, "E", transform=axE.transAxes, fontsize=14, fontweight='bold', va='top')

    axF = fig.add_subplot(gs[1:, 1])
    _bar_patient_shap(axF, shap_cardio, top_k=8, title="Cardiovascular: top contributors")
    axF.text(-0.08, 1.05, "F", transform=axF.transAxes, fontsize=14, fontweight='bold', va='top')

    axG = fig.add_subplot(gs[1:, 2])
    _bar_patient_shap(axG, shap_infect, top_k=8, title="Infection‑related: top contributors")
    axG.text(-0.08, 1.05, "G", transform=axG.transAxes, fontsize=14, fontweight='bold', va='top')

    fig.tight_layout()

    fig.savefig(outfile_png, bbox_inches='tight')
    if outfile_svg is not None:
        fig.savefig(outfile_svg, bbox_inches='tight')
    return outfile_png, outfile_svg
