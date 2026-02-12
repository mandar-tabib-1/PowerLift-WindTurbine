"""
Inference-Time Visualization Module for Fuhrlander FL2500 Predictive Maintenance
================================================================================
Standalone functions for generating presentation-quality plots during inference.
Designed for agentic setups where an LLM agent calls these functions.

All functions:
  - Accept numpy arrays / dicts (no dependency on training code)
  - Return matplotlib Figure objects (and optionally save to disk)
  - Use Agg backend (no GUI required)
  - Are importable as a module

Functions:
  1. plot_turbine_health_dashboard  - 4-panel single-turbine health overview
  2. plot_confusion_report          - Confusion matrix + classification report
  3. plot_feature_waterfall         - SHAP per-sample contribution bar chart
  4. plot_multi_model_summary       - All-models-at-a-glance for one observation
  5. plot_rul_trend                 - RUL over time with optional ground truth
  6. generate_inference_report      - Top-level: generates all plots for a turbine

Usage:
    from inference_viz import generate_inference_report
    report = generate_inference_report(turbine_id=83, ...)
    print(report['text_summary'])
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# =============================================================================
# 1. SINGLE-TURBINE HEALTH DASHBOARD
# =============================================================================

def plot_turbine_health_dashboard(
    timestamps,
    health_indicator,
    gmm_states,
    fault_probabilities,
    rul_predictions,
    failure_threshold,
    turbine_id="Unknown",
    multi_class_probs=None,
    true_labels=None,
    save_path=None,
):
    """
    4-panel single-turbine health dashboard.

    Panel 1: Health Indicator over time, colored by GMM state, with threshold.
    Panel 2: Fault probability over time (binary classifier P(anomalous)).
    Panel 3: RUL prediction over time with urgency coloring.
    Panel 4: Current status summary with recommendation.

    Parameters
    ----------
    timestamps : array-like, shape (N,)
    health_indicator : np.ndarray, shape (N,)
    gmm_states : np.ndarray, shape (N,), values {0,1,2}
    fault_probabilities : np.ndarray, shape (N,), P(anomalous)
    rul_predictions : np.ndarray, shape (N,), may contain NaN
    failure_threshold : float
    turbine_id : int or str
    multi_class_probs : np.ndarray, shape (N,3), optional
    true_labels : np.ndarray, shape (N,), optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(18, 16))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 1, 1, 0.6],
                           hspace=0.35, wspace=0.3)

    fig.suptitle(
        f'Turbine {turbine_id} - Health Assessment Dashboard',
        fontsize=16, fontweight='bold', y=0.98
    )

    state_colors = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728'}
    state_labels = {0: 'Healthy', 1: 'Degrading', 2: 'Critical'}

    # --- Panel 1: Health Indicator over time (full width) ---
    ax1 = fig.add_subplot(gs[0, :])
    for state in [0, 1, 2]:
        mask = gmm_states == state
        if np.any(mask):
            ax1.scatter(timestamps[mask], health_indicator[mask],
                        s=4, alpha=0.5, color=state_colors[state],
                        label=state_labels[state])
    ax1.axhline(y=failure_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Failure Threshold ({failure_threshold:.4f})')
    if true_labels is not None:
        fault_mask = true_labels == 2
        if np.any(fault_mask):
            ax1.scatter(timestamps[fault_mask], health_indicator[fault_mask],
                        s=20, marker='x', color='black', alpha=0.7,
                        label='Actual Fault', zorder=10)
    ax1.set_ylabel('Health Indicator', fontsize=11)
    ax1.set_title('Health Indicator Over Time (Autoencoder Reconstruction Error)',
                  fontweight='bold', fontsize=12)
    ax1.legend(markerscale=4, fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30, labelsize=8)

    # --- Panel 2: Fault probability (left) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(range(len(fault_probabilities)), fault_probabilities,
                     alpha=0.3, color='red')
    ax2.plot(fault_probabilities, linewidth=0.5, color='darkred')
    ax2.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5,
                label='Decision Boundary (0.5)')
    ax2.set_ylabel('P(Anomalous)', fontsize=11)
    ax2.set_xlabel('Sample Index', fontsize=10)
    ax2.set_title('Fault Probability (Binary Classifier)', fontweight='bold')
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Multi-class probabilities (right) ---
    ax3 = fig.add_subplot(gs[1, 1])
    if multi_class_probs is not None:
        class_names = ['Healthy', 'Pre-Fault', 'Fault']
        class_colors = ['#2ca02c', '#ff7f0e', '#d62728']
        for i, (name, color) in enumerate(zip(class_names, class_colors)):
            ax3.plot(multi_class_probs[:, i], linewidth=0.7, color=color,
                     alpha=0.7, label=name)
        ax3.set_ylabel('Probability', fontsize=11)
        ax3.set_xlabel('Sample Index', fontsize=10)
        ax3.set_title('Multi-Class Probabilities (RandomForest)', fontweight='bold')
        ax3.set_ylim(0, 1.05)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
        ax3.text(0.5, 0.5, 'Multi-class probabilities\nnot provided',
                 ha='center', va='center', fontsize=12, color='grey',
                 transform=ax3.transAxes)

    # --- Panel 4: RUL prediction (full width) ---
    ax4 = fig.add_subplot(gs[2, :])
    valid = ~np.isnan(rul_predictions)
    if np.any(valid):
        rul_vals = rul_predictions[valid]
        ts_valid = timestamps[valid] if hasattr(timestamps, '__getitem__') else np.arange(len(rul_predictions))[valid]
        colors_rul = []
        for r in rul_vals:
            if r < 50:
                colors_rul.append('#d62728')
            elif r < 200:
                colors_rul.append('#ff7f0e')
            else:
                colors_rul.append('#2ca02c')
        ax4.scatter(ts_valid, rul_vals, s=3, c=colors_rul, alpha=0.5)
    if true_labels is not None:
        fault_mask = true_labels == 2
        if np.any(fault_mask):
            ts_fault = timestamps[fault_mask] if hasattr(timestamps, '__getitem__') else np.arange(len(true_labels))[fault_mask]
            ax4.scatter(ts_fault, np.zeros(np.sum(fault_mask)),
                        color='black', s=15, marker='x', alpha=0.7,
                        label='Actual Fault', zorder=10)
    ax4.set_ylabel('Predicted RUL (hours)', fontsize=11)
    ax4.set_title('Remaining Useful Life Prediction', fontweight='bold', fontsize=12)
    legend_elements = [
        Patch(facecolor='#2ca02c', alpha=0.7, label='Safe (>200h)'),
        Patch(facecolor='#ff7f0e', alpha=0.7, label='Warning (50-200h)'),
        Patch(facecolor='#d62728', alpha=0.7, label='Critical (<50h)'),
    ]
    ax4.legend(handles=legend_elements, fontsize=9, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=30, labelsize=8)

    # --- Panel 5: Status summary box (full width) ---
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')

    latest_hi = health_indicator[-1]
    latest_state = int(gmm_states[-1])
    latest_fp = fault_probabilities[-1]
    latest_rul = rul_predictions[-1] if not np.isnan(rul_predictions[-1]) else None

    if latest_state == 2 or latest_fp > 0.7:
        urgency = "CRITICAL - IMMEDIATE INSPECTION RECOMMENDED"
        box_color = '#ffcccc'
    elif latest_state == 1 or latest_fp > 0.3:
        urgency = "WARNING - SCHEDULE MAINTENANCE WITHIN PREDICTED RUL"
        box_color = '#fff3cd'
    else:
        urgency = "NORMAL OPERATION"
        box_color = '#d4edda'

    multi_str = ""
    if multi_class_probs is not None:
        mp = multi_class_probs[-1]
        multi_str = f"\n  Multi-Class:     P(H)={mp[0]:.3f}  P(PF)={mp[1]:.3f}  P(F)={mp[2]:.3f}"

    summary = (
        f"CURRENT STATUS - Turbine {turbine_id}\n"
        f"{'='*55}\n"
        f"  Health Indicator:  {latest_hi:.4f}  (threshold: {failure_threshold:.4f})\n"
        f"  GMM State:         {state_labels[latest_state]}\n"
        f"  Fault Probability: {latest_fp:.3f}"
        f"{multi_str}\n"
        f"  Predicted RUL:     {f'{latest_rul:.0f} hours' if latest_rul else 'N/A'}\n"
        f"{'='*55}\n"
        f"  ASSESSMENT: {urgency}"
    )
    ax5.text(0.02, 0.95, summary, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor=box_color, alpha=0.9))

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# =============================================================================
# 2. CONFUSION MATRIX AND CLASSIFICATION REPORT
# =============================================================================

def plot_confusion_report(
    y_true,
    y_pred,
    class_names=None,
    title="Classification Performance",
    normalize=True,
    save_path=None,
):
    """
    2-panel figure: confusion matrix heatmap + classification report table.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : list, optional
    title : str
    normalize : bool
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    if class_names is None:
        n_classes = max(len(np.unique(y_true)), len(np.unique(y_pred)))
        if n_classes <= 2:
            class_names = ['Healthy', 'Anomalous']
        else:
            class_names = ['Healthy', 'Pre-Fault', 'Fault']

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                              gridspec_kw={'width_ratios': [1.2, 1]})
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Left: Confusion matrix heatmap
    ax = axes[0]
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums,
                            where=row_sums != 0,
                            out=np.zeros_like(cm, dtype=float))
        annot = np.array([[f'{cm[i, j]}\n({cm_norm[i, j]:.1%})'
                           for j in range(len(class_names))]
                          for i in range(len(class_names))])
        sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=1)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title('Confusion Matrix', fontweight='bold')

    # Right: Classification report as text
    ax2 = axes[1]
    ax2.axis('off')
    ax2.text(0.05, 0.95, report_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax2.set_title('Classification Report', fontweight='bold')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# =============================================================================
# 3. FEATURE CONTRIBUTION WATERFALL (SHAP-based)
# =============================================================================

def plot_feature_waterfall(
    feature_names,
    feature_values,
    shap_values,
    prediction_label="",
    prediction_prob=None,
    top_n=15,
    save_path=None,
):
    """
    Waterfall-style bar chart showing feature contributions to a prediction.

    Parameters
    ----------
    feature_names : list of str (27 names)
    feature_values : np.ndarray, shape (27,)
    shap_values : np.ndarray, shape (27,)
    prediction_label : str
    prediction_prob : float, optional
    top_n : int
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    abs_shap = np.abs(shap_values)
    top_idx = np.argsort(abs_shap)[::-1][:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # Left: Waterfall with feature values
    colors = ['#d62728' if shap_values[i] > 0 else '#1f77b4' for i in top_idx]
    y_pos = np.arange(len(top_idx))

    ax1.barh(y_pos, shap_values[top_idx], color=colors, alpha=0.85)
    labels = [f'{feature_names[i]} = {feature_values[i]:.2f}' for i in top_idx]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
    ax1.set_xlabel('SHAP Value (impact on prediction)', fontsize=11)

    title = f'Feature Contributions: {prediction_label}'
    if prediction_prob is not None:
        title += f' (P={prediction_prob:.3f})'
    ax1.set_title(title, fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='x')

    # Right: Color-coded bar chart (no values, cleaner)
    colors_right = ['#d62728' if shap_values[i] > 0 else '#1f77b4' for i in top_idx]
    ax2.barh(y_pos, shap_values[top_idx], color=colors_right, alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
    ax2.invert_yaxis()
    ax2.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
    ax2.set_xlabel('SHAP Value', fontsize=11)
    ax2.set_title('Top Feature Contributions', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')

    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.85, label='Pushes toward FAULT'),
        Patch(facecolor='#1f77b4', alpha=0.85, label='Pushes toward HEALTHY'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# =============================================================================
# 4. MULTI-MODEL SUMMARY PANEL
# =============================================================================

def plot_multi_model_summary(
    health_indicator,
    failure_threshold,
    gmm_state,
    binary_prob,
    multi_prob,
    rul_hours,
    feature_importances=None,
    feature_names=None,
    turbine_id="Unknown",
    timestamp=None,
    save_path=None,
):
    """
    Single-observation summary panel showing all model outputs at a glance.

    Parameters
    ----------
    health_indicator : float
    failure_threshold : float
    gmm_state : int (0, 1, or 2)
    binary_prob : list [P(healthy), P(anomalous)]
    multi_prob : list [P(healthy), P(pre-fault), P(fault)]
    rul_hours : float
    feature_importances : np.ndarray (27,), optional
    feature_names : list, optional
    turbine_id : int or str
    timestamp : str, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3)

    state_label_map = {0: 'HEALTHY', 1: 'DEGRADING', 2: 'CRITICAL'}
    state_color_map = {0: '#2ca02c', 1: '#ff7f0e', 2: '#d62728'}

    ts_str = timestamp if timestamp else 'Latest Observation'
    fig.suptitle(
        f'Turbine {turbine_id} - Multi-Model Assessment\n{ts_str}',
        fontsize=14, fontweight='bold'
    )

    # --- Top-left: HI gauge ---
    ax1 = fig.add_subplot(gs[0, 0])
    bar_color = state_color_map.get(gmm_state, '#999999')
    ax1.barh(['Health\nIndicator'], [health_indicator], color=bar_color,
             alpha=0.8, height=0.5)
    ax1.axvline(x=failure_threshold, color='red', linestyle='--', linewidth=2.5,
                label=f'Threshold ({failure_threshold:.4f})')
    x_max = max(health_indicator * 1.3, failure_threshold * 1.5)
    ax1.set_xlim(0, x_max)
    ax1.set_title('Health Indicator vs Threshold', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.text(health_indicator + x_max * 0.02, 0, f'{health_indicator:.4f}',
             va='center', fontsize=12, fontweight='bold', color=bar_color)
    ax1.grid(True, alpha=0.3, axis='x')
    hi_status = "ABOVE THRESHOLD" if health_indicator > failure_threshold else "Below Threshold"
    ax1.set_xlabel(f'Status: {hi_status}', fontsize=10,
                   color='red' if health_indicator > failure_threshold else 'green')

    # --- Top-right: Multi-class probability bar ---
    ax2 = fig.add_subplot(gs[0, 1])
    class_names = ['Healthy', 'Pre-Fault', 'Fault']
    class_colors = ['#2ca02c', '#ff7f0e', '#d62728']
    bars = ax2.bar(class_names, multi_prob, color=class_colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    for bar, prob in zip(bars, multi_prob):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{prob:.3f}', ha='center', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('Probability', fontsize=11)
    ax2.set_title(f'Multi-Class Prediction: {state_label_map.get(gmm_state, "?")}',
                  fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0.5, color='grey', linestyle=':', alpha=0.4)

    # --- Bottom-left: RUL display ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis('off')

    rul_display = rul_hours if not np.isnan(rul_hours) else 0
    if rul_display < 50:
        rul_color = '#d62728'
        rul_urgency = 'CRITICAL'
    elif rul_display < 200:
        rul_color = '#ff7f0e'
        rul_urgency = 'WARNING'
    else:
        rul_color = '#2ca02c'
        rul_urgency = 'NORMAL'

    ax3.text(0.5, 0.65, f'{rul_display:.0f}', transform=ax3.transAxes,
             fontsize=64, ha='center', va='center', fontweight='bold',
             color=rul_color)
    ax3.text(0.5, 0.35, 'hours remaining', transform=ax3.transAxes,
             fontsize=14, ha='center', va='center', color='grey')
    ax3.text(0.5, 0.18, rul_urgency, transform=ax3.transAxes,
             fontsize=18, ha='center', va='center', fontweight='bold',
             color=rul_color,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=rul_color, linewidth=2))
    ax3.text(0.5, 0.05, f'Binary P(anomalous): {binary_prob[1]:.3f}',
             transform=ax3.transAxes, fontsize=11, ha='center', va='center',
             color='grey')
    ax3.set_title('Remaining Useful Life', fontweight='bold', fontsize=12)

    # --- Bottom-right: Top feature importance or summary ---
    ax4 = fig.add_subplot(gs[1, 1])
    if feature_importances is not None and feature_names is not None:
        n_show = min(10, len(feature_importances))
        top_idx = np.argsort(np.abs(feature_importances))[-n_show:]
        colors_feat = ['#d62728' if feature_importances[i] > 0 else '#1f77b4'
                       for i in top_idx]
        ax4.barh(range(n_show), feature_importances[top_idx],
                 color=colors_feat, alpha=0.8)
        ax4.set_yticks(range(n_show))
        ax4.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
        ax4.set_xlabel('Importance / SHAP Value', fontsize=10)
        ax4.set_title('Top Feature Contributions', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.axvline(x=0, color='grey', linestyle='--', alpha=0.4)
    else:
        ax4.axis('off')
        summary_box = (
            f"MODEL SUMMARY\n"
            f"{'='*35}\n"
            f"Autoencoder HI:  {health_indicator:.4f}\n"
            f"GMM State:       {state_label_map.get(gmm_state, '?')}\n"
            f"P(anomalous):    {binary_prob[1]:.3f}\n"
            f"P(healthy):      {multi_prob[0]:.3f}\n"
            f"P(pre-fault):    {multi_prob[1]:.3f}\n"
            f"P(fault):        {multi_prob[2]:.3f}\n"
            f"RUL:             {rul_display:.0f} hours"
        )
        ax4.text(0.1, 0.9, summary_box, transform=ax4.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax4.set_title('Model Summary', fontweight='bold', fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# =============================================================================
# 5. RUL TREND PLOT
# =============================================================================

def plot_rul_trend(
    timestamps,
    rul_predictions,
    true_hours_to_fault=None,
    failure_events=None,
    turbine_id="Unknown",
    save_path=None,
):
    """
    RUL prediction over time with optional ground truth overlay.

    Parameters
    ----------
    timestamps : array-like, shape (N,)
    rul_predictions : np.ndarray, shape (N,), may contain NaN
    true_hours_to_fault : np.ndarray, shape (N,), optional
    failure_events : np.ndarray, shape (N,) boolean mask, optional
    turbine_id : int or str
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    valid = ~np.isnan(rul_predictions)
    if np.any(valid):
        ax.plot(timestamps[valid], rul_predictions[valid],
                linewidth=0.8, color='purple', alpha=0.7,
                label='Predicted RUL')

    if true_hours_to_fault is not None:
        valid_true = true_hours_to_fault < 900
        if np.any(valid_true):
            ax.plot(timestamps[valid_true], true_hours_to_fault[valid_true],
                    linewidth=1.2, color='blue', alpha=0.6, linestyle='--',
                    label='Actual Hours to Fault')

    if failure_events is not None and np.any(failure_events):
        for ts in timestamps[failure_events]:
            ax.axvline(x=ts, color='red', alpha=0.3, linewidth=0.8)
        ax.axvline(x=timestamps[failure_events][0], color='red', alpha=0.3,
                   linewidth=0.8, label='Actual Fault Events')

    ax.axhline(y=50, color='orange', linestyle=':', alpha=0.5,
               label='Critical RUL (50h)')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Remaining Useful Life (hours)', fontsize=11)
    ax.set_title(f'Turbine {turbine_id} - RUL Prediction Over Time',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30, labelsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    return fig


# =============================================================================
# 6. CONVENIENCE: FULL INFERENCE REPORT
# =============================================================================

def generate_inference_report(
    turbine_id,
    timestamps,
    health_indicator,
    gmm_states,
    fault_probabilities,
    multi_class_probs,
    rul_predictions,
    failure_threshold,
    feature_names,
    feature_importances=None,
    shap_values_single=None,
    feature_values_single=None,
    true_labels=None,
    true_hours_to_fault=None,
    output_dir=None,
):
    """
    Generate a complete set of inference visualizations for one turbine.

    This is the PRIMARY function an LLM agent would call.

    Parameters
    ----------
    turbine_id : int or str
    timestamps : array-like (N,)
    health_indicator : np.ndarray (N,)
    gmm_states : np.ndarray (N,)
    fault_probabilities : np.ndarray (N,) - P(anomalous)
    multi_class_probs : np.ndarray (N,3)
    rul_predictions : np.ndarray (N,)
    failure_threshold : float
    feature_names : list of str (27)
    feature_importances : np.ndarray (27,), optional - SHAP or MDI importances
    shap_values_single : np.ndarray (27,), optional - SHAP for latest sample
    feature_values_single : np.ndarray (27,), optional - feature vals for latest
    true_labels : np.ndarray (N,), optional
    true_hours_to_fault : np.ndarray (N,), optional
    output_dir : str, optional

    Returns
    -------
    dict
        Keys: dashboard_path, confusion_path, waterfall_path, summary_path,
              rul_trend_path, text_summary
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)

    prefix = f'inference_turbine_{turbine_id}'
    paths = {}

    # 1. Dashboard
    dash_path = os.path.join(output_dir, f'{prefix}_dashboard.png')
    plot_turbine_health_dashboard(
        timestamps, health_indicator, gmm_states,
        fault_probabilities, rul_predictions, failure_threshold,
        turbine_id=turbine_id, multi_class_probs=multi_class_probs,
        true_labels=true_labels, save_path=dash_path,
    )
    paths['dashboard_path'] = dash_path
    print(f"  Saved: {dash_path}")

    # 2. Confusion matrix (only if true labels available)
    if true_labels is not None:
        y_pred_multi = np.argmax(multi_class_probs, axis=1)
        cm_path = os.path.join(output_dir, f'{prefix}_confusion.png')
        plot_confusion_report(
            true_labels, y_pred_multi,
            class_names=['Healthy', 'Pre-Fault', 'Fault'],
            title=f'Turbine {turbine_id} - Classification Performance',
            save_path=cm_path,
        )
        paths['confusion_path'] = cm_path
        print(f"  Saved: {cm_path}")

    # 3. SHAP waterfall (only if shap values provided)
    if shap_values_single is not None and feature_values_single is not None:
        wf_path = os.path.join(output_dir, f'{prefix}_shap_waterfall.png')
        latest_pred = 'ANOMALOUS' if fault_probabilities[-1] > 0.5 else 'HEALTHY'
        plot_feature_waterfall(
            feature_names=feature_names,
            feature_values=feature_values_single,
            shap_values=shap_values_single,
            prediction_label=latest_pred,
            prediction_prob=fault_probabilities[-1],
            save_path=wf_path,
        )
        paths['waterfall_path'] = wf_path
        print(f"  Saved: {wf_path}")

    # 4. Multi-model summary (latest observation)
    latest_rul = rul_predictions[-1] if not np.isnan(rul_predictions[-1]) else 999.0
    summary_path = os.path.join(output_dir, f'{prefix}_summary.png')
    plot_multi_model_summary(
        health_indicator=health_indicator[-1],
        failure_threshold=failure_threshold,
        gmm_state=int(gmm_states[-1]),
        binary_prob=[1 - fault_probabilities[-1], fault_probabilities[-1]],
        multi_prob=multi_class_probs[-1].tolist(),
        rul_hours=latest_rul,
        feature_importances=shap_values_single,
        feature_names=feature_names,
        turbine_id=turbine_id,
        save_path=summary_path,
    )
    paths['summary_path'] = summary_path
    print(f"  Saved: {summary_path}")

    # 5. RUL trend
    rul_path = os.path.join(output_dir, f'{prefix}_rul_trend.png')
    plot_rul_trend(
        timestamps, rul_predictions,
        true_hours_to_fault=true_hours_to_fault,
        failure_events=(true_labels == 2) if true_labels is not None else None,
        turbine_id=turbine_id,
        save_path=rul_path,
    )
    paths['rul_trend_path'] = rul_path
    print(f"  Saved: {rul_path}")

    # 6. Text summary
    state_labels = {0: 'HEALTHY', 1: 'DEGRADING', 2: 'CRITICAL'}
    latest_hi = health_indicator[-1]
    latest_state = int(gmm_states[-1])
    latest_fp = fault_probabilities[-1]

    text_summary = (
        f"INFERENCE REPORT - Turbine {turbine_id}\n"
        f"{'='*55}\n"
        f"Time range: {timestamps[0]} to {timestamps[-1]}\n"
        f"Observations: {len(timestamps)}\n\n"
        f"LATEST STATUS:\n"
        f"  Health Indicator: {latest_hi:.4f} (threshold: {failure_threshold:.4f})\n"
        f"  GMM State: {state_labels[latest_state]}\n"
        f"  Fault Probability: {latest_fp:.3f}\n"
        f"  Multi-Class: P(H)={multi_class_probs[-1][0]:.3f}  "
        f"P(PF)={multi_class_probs[-1][1]:.3f}  "
        f"P(F)={multi_class_probs[-1][2]:.3f}\n"
        f"  Predicted RUL: {f'{latest_rul:.0f} hours' if not np.isnan(rul_predictions[-1]) else 'N/A'}\n\n"
        f"DISTRIBUTION OVER ASSESSMENT PERIOD:\n"
        f"  Healthy:   {np.mean(gmm_states == 0) * 100:.1f}%\n"
        f"  Degrading: {np.mean(gmm_states == 1) * 100:.1f}%\n"
        f"  Critical:  {np.mean(gmm_states == 2) * 100:.1f}%\n"
        f"  Mean fault probability: {np.mean(fault_probabilities):.3f}\n"
    )

    if true_labels is not None:
        y_pred_m = np.argmax(multi_class_probs, axis=1)
        from sklearn.metrics import accuracy_score, f1_score
        text_summary += (
            f"\nVALIDATION (vs ground truth):\n"
            f"  Accuracy: {accuracy_score(true_labels, y_pred_m):.4f}\n"
            f"  Weighted F1: {f1_score(true_labels, y_pred_m, average='weighted', zero_division=0):.4f}\n"
        )

    paths['text_summary'] = text_summary
    return paths
