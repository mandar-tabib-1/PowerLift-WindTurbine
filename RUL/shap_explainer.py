"""
SHAP Explainability Module for Wind Turbine Predictive Maintenance
===================================================================
Standalone module for computing and visualizing SHAP-based feature importance.

Can be used:
  1. During training - called from wind_turbine_pm_fuhrlander.py main pipeline
  2. Standalone demo - imported and called with pre-trained models + test data
  3. Agentic setup  - explain_single_sample() for real-time per-sample explanations

Usage (standalone):
    from shap_explainer import SHAPExplainer

    explainer = SHAPExplainer(
        binary_clf=binary_clf,
        multi_clf=multi_clf,
        feature_names=ALL_FEATURES,
        output_dir='RUL/'
    )
    # Full analysis on test set (generates plots + report text)
    report_text = explainer.run_full_analysis(X_test, y_test)

    # Single sample explanation (for agentic/demo inference)
    result = explainer.explain_single_sample(X_test[0:1], sample_id=0)
    print(result['explanation_text'])

Author: Wind Turbine PM Expert
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os
import warnings
warnings.filterwarnings('ignore')


class SHAPExplainer:
    """
    SHAP-based model explainability for wind turbine fault prediction.

    Wraps both the binary (GradientBoosting) and multi-class (RandomForest)
    classifiers to provide global and local SHAP explanations.

    Attributes:
        binary_clf: Trained GradientBoostingClassifier (healthy vs anomalous)
        multi_clf:  Trained RandomForestClassifier (healthy/pre-fault/fault)
        feature_names: List of 27 feature names
        output_dir: Directory for saving plots
    """

    def __init__(self, binary_clf, multi_clf, feature_names, output_dir=None):
        self.binary_clf = binary_clf
        self.multi_clf = multi_clf
        self.feature_names = list(feature_names)
        self.output_dir = output_dir or os.path.dirname(__file__)

        # SHAP explainers (lazy-initialized on first use)
        self._binary_explainer = None
        self._multi_explainer = None
        self._binary_shap_values = None
        self._multi_shap_values = None
        self._X_explain = None

    # ------------------------------------------------------------------
    # Explainer initialization
    # ------------------------------------------------------------------

    def _init_binary_explainer(self, X_background=None):
        """Initialize SHAP TreeExplainer for binary classifier."""
        if self._binary_explainer is None:
            print("  Initializing SHAP TreeExplainer (binary classifier)...")
            self._binary_explainer = shap.TreeExplainer(self.binary_clf)
        return self._binary_explainer

    def _init_multi_explainer(self, X_background=None):
        """Initialize SHAP TreeExplainer for multi-class classifier."""
        if self._multi_explainer is None:
            print("  Initializing SHAP TreeExplainer (multi-class classifier)...")
            self._multi_explainer = shap.TreeExplainer(self.multi_clf)
        return self._multi_explainer

    # ------------------------------------------------------------------
    # Core SHAP computation
    # ------------------------------------------------------------------

    def compute_shap_values(self, X, max_samples=2000):
        """
        Compute SHAP values for both classifiers on given data.

        Args:
            X: Feature matrix, shape (N, 27). Can be numpy array or DataFrame.
            max_samples: Max samples for SHAP computation (subsampled if larger).

        Returns:
            dict with keys:
                'binary_shap': SHAP values for binary classifier, shape (N, 27)
                'multi_shap': SHAP values for multi-class, shape (N, 27, 3)
                'X_used': The (possibly subsampled) feature matrix used
                'sample_indices': Indices used if subsampled
        """
        # Convert to DataFrame for readable SHAP plots
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X.copy()

        # Subsample if too large (SHAP on full dataset is slow)
        if len(X_df) > max_samples:
            np.random.seed(42)
            indices = np.random.choice(len(X_df), max_samples, replace=False)
            indices = np.sort(indices)
            X_sample = X_df.iloc[indices].reset_index(drop=True)
            print(f"  Subsampled {len(X_df):,} -> {max_samples:,} for SHAP")
        else:
            indices = np.arange(len(X_df))
            X_sample = X_df

        self._X_explain = X_sample

        # Binary SHAP
        binary_exp = self._init_binary_explainer()
        print("  Computing SHAP values (binary classifier)...")
        self._binary_shap_values = binary_exp.shap_values(X_sample)

        # Multi-class SHAP
        multi_exp = self._init_multi_explainer()
        print("  Computing SHAP values (multi-class classifier)...")
        self._multi_shap_values = multi_exp.shap_values(X_sample)

        return {
            'binary_shap': self._binary_shap_values,
            'multi_shap': self._multi_shap_values,
            'X_used': X_sample,
            'sample_indices': indices,
        }

    # ------------------------------------------------------------------
    # Global importance (mean |SHAP|)
    # ------------------------------------------------------------------

    def get_global_importance(self):
        """
        Get global feature importance as mean |SHAP value| per feature.

        Returns:
            DataFrame with columns [feature, binary_importance, multi_importance_class0/1/2]
        """
        if self._binary_shap_values is None:
            raise RuntimeError("Call compute_shap_values() first.")

        # Binary: shape (N, 27) for class 1 or list of 2 arrays
        if isinstance(self._binary_shap_values, list):
            binary_imp = np.mean(np.abs(self._binary_shap_values[1]), axis=0)
        else:
            binary_imp = np.mean(np.abs(self._binary_shap_values), axis=0)

        # Multi-class: list of 3 arrays each (N, 27)
        multi_imp = {}
        class_names = ['Healthy', 'Pre-Fault', 'Fault']
        if isinstance(self._multi_shap_values, list):
            for c in range(len(self._multi_shap_values)):
                multi_imp[class_names[c]] = np.mean(
                    np.abs(self._multi_shap_values[c]), axis=0
                )
        else:
            multi_imp['Overall'] = np.mean(
                np.abs(self._multi_shap_values), axis=0
            )

        result = pd.DataFrame({
            'feature': self.feature_names,
            'binary_shap_importance': binary_imp,
        })
        for name, imp in multi_imp.items():
            result[f'multi_shap_{name}'] = imp

        result = result.sort_values('binary_shap_importance', ascending=False)
        return result.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(self, X=None, y=None, max_samples=2000):
        """
        Generate all SHAP plots and save to output_dir.

        Args:
            X: Feature matrix (N, 27). If None, uses previously computed data.
            y: True labels (optional, for coloring).
            max_samples: Max samples for SHAP computation.

        Returns:
            List of saved file paths.
        """
        if X is not None:
            self.compute_shap_values(X, max_samples=max_samples)

        if self._binary_shap_values is None:
            raise RuntimeError("No SHAP values computed. Pass X or call compute_shap_values().")

        saved_files = []
        X_df = self._X_explain

        # --- 1. Binary SHAP Summary (beeswarm) ---
        saved_files.append(self._plot_binary_summary(X_df))

        # --- 2. Binary SHAP Bar (global importance) ---
        saved_files.append(self._plot_binary_bar(X_df))

        # --- 3. Multi-class SHAP Summary ---
        saved_files.append(self._plot_multi_summary(X_df))

        # --- 4. Top feature dependence plots ---
        saved_files.append(self._plot_dependence(X_df))

        # --- 5. Combined dashboard ---
        saved_files.append(self._plot_dashboard(X_df, y))

        return saved_files

    def _plot_binary_summary(self, X_df):
        """SHAP beeswarm plot for binary classifier."""
        path = os.path.join(self.output_dir, 'shap_binary_summary.png')
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap_vals = self._binary_shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # class 1 = anomalous
        shap.summary_plot(shap_vals, X_df, show=False, max_display=20)
        plt.title('SHAP Feature Importance (Binary: Healthy vs Anomalous)',
                  fontweight='bold', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(path)}")
        return path

    def _plot_binary_bar(self, X_df):
        """SHAP bar chart (mean |SHAP|) for binary classifier."""
        path = os.path.join(self.output_dir, 'shap_binary_bar.png')
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.sca(ax)
        shap_vals = self._binary_shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap.summary_plot(shap_vals, X_df, plot_type='bar',
                         show=False, max_display=20)
        plt.title('Mean |SHAP| Feature Importance (Binary Classifier)',
                  fontweight='bold', fontsize=12, pad=15)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(path)}")
        return path

    def _plot_multi_summary(self, X_df):
        """SHAP bar chart for multi-class classifier (per-class importance)."""
        path = os.path.join(self.output_dir, 'shap_multiclass_bar.png')

        if isinstance(self._multi_shap_values, list) and len(self._multi_shap_values) == 3:
            class_names = ['Healthy', 'Pre-Fault', 'Fault']
            fig, axes = plt.subplots(1, 3, figsize=(20, 8))
            for c in range(3):
                plt.sca(axes[c])
                shap.summary_plot(self._multi_shap_values[c], X_df,
                                 plot_type='bar', show=False, max_display=15)
                axes[c].set_title(f'Class: {class_names[c]}',
                                 fontweight='bold', fontsize=11)
            fig.suptitle('SHAP Feature Importance by Class (Multi-class RF)',
                        fontweight='bold', fontsize=14, y=1.02)
            plt.tight_layout()
        else:
            fig, ax = plt.subplots(figsize=(12, 8))
            plt.sca(ax)
            shap.summary_plot(self._multi_shap_values, X_df,
                             plot_type='bar', show=False, max_display=20)
            plt.title('SHAP Feature Importance (Multi-class RF)',
                      fontweight='bold', fontsize=12, pad=15)
            plt.tight_layout()

        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(path)}")
        return path

    def _plot_dependence(self, X_df):
        """SHAP dependence plots for top 4 features."""
        path = os.path.join(self.output_dir, 'shap_dependence.png')

        shap_vals = self._binary_shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:4]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for i, feat_idx in enumerate(top_indices):
            ax = axes[i // 2][i % 2]
            plt.sca(ax)
            shap.dependence_plot(
                feat_idx, shap_vals, X_df,
                ax=ax, show=False
            )
            ax.set_title(f'SHAP Dependence: {self.feature_names[feat_idx]}',
                        fontweight='bold', fontsize=10)

        fig.suptitle('SHAP Dependence Plots (Top 4 Features, Binary Classifier)',
                    fontweight='bold', fontsize=13, y=1.01)
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(path)}")
        return path

    def _plot_dashboard(self, X_df, y=None):
        """Combined SHAP dashboard comparing MDI vs SHAP importance."""
        path = os.path.join(self.output_dir, 'shap_dashboard.png')

        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(
            'SHAP Explainability Dashboard - Fuhrlander FL2500 Fault Prediction',
            fontweight='bold', fontsize=14, y=1.005
        )

        shap_vals = self._binary_shap_values
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        # --- Panel 1: MDI vs SHAP comparison ---
        ax = axes[0, 0]
        mdi_imp = self.binary_clf.feature_importances_
        sorted_idx = np.argsort(mean_abs_shap)[-15:]
        y_pos = np.arange(len(sorted_idx))

        # Normalize both to [0,1] for comparison
        mdi_norm = mdi_imp[sorted_idx] / (mdi_imp.max() + 1e-10)
        shap_norm = mean_abs_shap[sorted_idx] / (mean_abs_shap.max() + 1e-10)

        ax.barh(y_pos - 0.2, mdi_norm, 0.35,
                label='MDI (Gini)', color='steelblue', alpha=0.8)
        ax.barh(y_pos + 0.2, shap_norm, 0.35,
                label='SHAP (mean |SHAP|)', color='coral', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx],
                          fontsize=8)
        ax.set_xlabel('Normalized Importance')
        ax.set_title('MDI (Gini) vs SHAP Importance (Top 15)',
                     fontweight='bold', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='x')

        # --- Panel 2: SHAP beeswarm ---
        plt.sca(axes[0, 1])
        shap.summary_plot(shap_vals, X_df, show=False, max_display=15)
        axes[0, 1].set_title('SHAP Beeswarm (Binary: Anomalous Class)',
                            fontweight='bold', fontsize=11)

        # --- Panel 3: Multi-class SHAP bar (Fault class only) ---
        ax = axes[1, 0]
        if isinstance(self._multi_shap_values, list) and len(self._multi_shap_values) >= 3:
            fault_shap = np.mean(np.abs(self._multi_shap_values[2]), axis=0)
            sorted_fault = np.argsort(fault_shap)[-15:]
            ax.barh(range(len(sorted_fault)),
                   fault_shap[sorted_fault], color='red', alpha=0.7)
            ax.set_yticks(range(len(sorted_fault)))
            ax.set_yticklabels([self.feature_names[i] for i in sorted_fault],
                              fontsize=8)
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('SHAP Importance for FAULT Class (RF)',
                        fontweight='bold', fontsize=11)
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'Multi-class SHAP\nnot available',
                   ha='center', va='center', fontsize=14)
            ax.set_title('SHAP Importance for FAULT Class',
                        fontweight='bold', fontsize=11)

        # --- Panel 4: Correlation between SHAP and feature value ---
        ax = axes[1, 1]
        top_feat_idx = np.argmax(mean_abs_shap)
        feat_name = self.feature_names[top_feat_idx]
        ax.scatter(X_df.iloc[:, top_feat_idx], shap_vals[:, top_feat_idx],
                  alpha=0.3, s=5, c='purple')
        ax.set_xlabel(f'{feat_name} (feature value)')
        ax.set_ylabel(f'SHAP value for {feat_name}')
        ax.set_title(f'SHAP vs Feature Value: {feat_name}',
                     fontweight='bold', fontsize=11)
        ax.axhline(y=0, color='grey', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(path)}")
        return path

    # ------------------------------------------------------------------
    # Single-sample explanation (for agentic / real-time demo)
    # ------------------------------------------------------------------

    def explain_single_sample(self, X_single, sample_id=0,
                              save_plot=True, verbose=True):
        """
        Explain a single prediction. Designed for agentic / demo inference.

        Args:
            X_single: Feature vector, shape (1, 27) or (27,).
                      Can be numpy array or DataFrame row.
            sample_id: Identifier for this sample (for plot titles/filenames).
            save_plot: If True, saves waterfall + force plot as PNG.
            verbose: If True, prints explanation text.

        Returns:
            dict with keys:
                'binary_prediction': 0 or 1
                'binary_probability': [p_healthy, p_anomalous]
                'multi_prediction': 0, 1, or 2
                'multi_probability': [p_healthy, p_prefault, p_fault]
                'binary_shap_values': array (27,) SHAP for binary
                'multi_shap_values': list of 3 arrays (27,) SHAP per class
                'top_features_pushing_fault': list of (feature, shap_value, feat_value)
                'top_features_pushing_healthy': list of (feature, shap_value, feat_value)
                'explanation_text': Human-readable explanation string
                'plot_path': Path to saved plot (if save_plot=True)
        """
        # Ensure shape (1, 27)
        if isinstance(X_single, pd.DataFrame):
            x_arr = X_single.values
        elif isinstance(X_single, pd.Series):
            x_arr = X_single.values.reshape(1, -1)
        else:
            x_arr = np.asarray(X_single)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)

        x_df = pd.DataFrame(x_arr, columns=self.feature_names)

        # Predictions
        binary_pred = self.binary_clf.predict(x_arr)[0]
        binary_prob = self.binary_clf.predict_proba(x_arr)[0]
        multi_pred = self.multi_clf.predict(x_arr)[0]
        multi_prob = self.multi_clf.predict_proba(x_arr)[0]

        # SHAP values
        binary_exp = self._init_binary_explainer()
        multi_exp = self._init_multi_explainer()

        b_shap = binary_exp.shap_values(x_df)
        m_shap = multi_exp.shap_values(x_df)

        # Extract single-sample SHAP
        if isinstance(b_shap, list):
            b_shap_single = b_shap[1][0]  # class 1 (anomalous)
        else:
            b_shap_single = b_shap[0]

        if isinstance(m_shap, list):
            m_shap_single = [m_shap[c][0] for c in range(len(m_shap))]
        else:
            m_shap_single = [m_shap[0]]

        # Sort features by SHAP impact
        feature_vals = x_arr[0]
        sorted_by_shap = np.argsort(b_shap_single)

        top_fault = []  # Features pushing toward fault (positive SHAP)
        top_healthy = []  # Features pushing toward healthy (negative SHAP)

        for idx in sorted_by_shap[::-1]:  # Most positive first
            if b_shap_single[idx] > 0:
                top_fault.append((
                    self.feature_names[idx],
                    float(b_shap_single[idx]),
                    float(feature_vals[idx])
                ))
        for idx in sorted_by_shap:  # Most negative first
            if b_shap_single[idx] < 0:
                top_healthy.append((
                    self.feature_names[idx],
                    float(b_shap_single[idx]),
                    float(feature_vals[idx])
                ))

        # Build explanation text
        label_map = {0: 'HEALTHY', 1: 'PRE-FAULT', 2: 'FAULT'}
        binary_label = 'ANOMALOUS' if binary_pred == 1 else 'HEALTHY'

        lines = []
        lines.append(f"=== SHAP Explanation for Sample {sample_id} ===")
        lines.append(f"")
        lines.append(f"Binary prediction: {binary_label} "
                     f"(P(anomalous)={binary_prob[1]:.3f})")
        lines.append(f"Multi-class prediction: {label_map.get(multi_pred, '?')} "
                     f"(P=[H:{multi_prob[0]:.3f}, PF:{multi_prob[1]:.3f}, "
                     f"F:{multi_prob[2]:.3f}])")
        lines.append(f"")
        lines.append(f"Top features PUSHING TOWARD FAULT:")
        for feat, sv, fv in top_fault[:5]:
            lines.append(f"  {feat:>35s}: SHAP={sv:+.4f}  value={fv:.3f}")
        lines.append(f"")
        lines.append(f"Top features PUSHING TOWARD HEALTHY:")
        for feat, sv, fv in top_healthy[:5]:
            lines.append(f"  {feat:>35s}: SHAP={sv:+.4f}  value={fv:.3f}")

        explanation_text = "\n".join(lines)

        if verbose:
            print(explanation_text)

        # Save waterfall plot
        plot_path = None
        if save_plot:
            plot_path = self._plot_single_sample(
                x_df, b_shap, binary_exp, sample_id, binary_label, binary_prob
            )

        return {
            'binary_prediction': int(binary_pred),
            'binary_probability': binary_prob.tolist(),
            'multi_prediction': int(multi_pred),
            'multi_probability': multi_prob.tolist(),
            'binary_shap_values': b_shap_single.tolist(),
            'multi_shap_values': [s.tolist() for s in m_shap_single],
            'top_features_pushing_fault': top_fault[:10],
            'top_features_pushing_healthy': top_healthy[:10],
            'explanation_text': explanation_text,
            'plot_path': plot_path,
        }

    def _plot_single_sample(self, x_df, b_shap, binary_exp, sample_id,
                            binary_label, binary_prob):
        """Save waterfall + bar plot for a single sample."""
        path = os.path.join(
            self.output_dir, f'shap_sample_{sample_id}.png'
        )

        if isinstance(b_shap, list):
            shap_vals_single = b_shap[1][0]
        else:
            shap_vals_single = b_shap[0]

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Left: horizontal bar of SHAP values for this sample
        ax = axes[0]
        sorted_idx = np.argsort(np.abs(shap_vals_single))[-15:]
        colors = ['red' if shap_vals_single[i] > 0 else 'blue'
                  for i in sorted_idx]
        ax.barh(range(len(sorted_idx)),
                shap_vals_single[sorted_idx], color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_idx)))
        labels = []
        for i in sorted_idx:
            labels.append(
                f"{self.feature_names[i]} = {x_df.iloc[0, i]:.2f}"
            )
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('SHAP value (impact on anomaly prediction)')
        ax.set_title(
            f'Sample {sample_id}: {binary_label} '
            f'(P(anom)={binary_prob[1]:.3f})',
            fontweight='bold', fontsize=11
        )
        ax.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='x')

        # Right: feature values as a table-like bar chart
        ax2 = axes[1]
        top_n = 15
        importance_order = np.argsort(np.abs(shap_vals_single))[::-1][:top_n]
        feat_data = []
        for idx in importance_order:
            feat_data.append({
                'Feature': self.feature_names[idx],
                'Value': x_df.iloc[0, idx],
                'SHAP': shap_vals_single[idx],
                'Direction': 'Fault' if shap_vals_single[idx] > 0 else 'Healthy'
            })
        feat_df = pd.DataFrame(feat_data)

        bar_colors = ['#d62728' if d == 'Fault' else '#1f77b4'
                      for d in feat_df['Direction']]
        ax2.barh(range(len(feat_df)), feat_df['SHAP'],
                color=bar_colors, alpha=0.8)
        ax2.set_yticks(range(len(feat_df)))
        ax2.set_yticklabels(feat_df['Feature'], fontsize=8)
        ax2.set_xlabel('SHAP value')
        ax2.set_title('Top 15 Feature Contributions', fontweight='bold',
                      fontsize=11)
        ax2.axvline(x=0, color='grey', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', alpha=0.8, label='Pushes toward FAULT'),
            Patch(facecolor='#1f77b4', alpha=0.8, label='Pushes toward HEALTHY')
        ]
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {os.path.basename(path)}")
        return path

    # ------------------------------------------------------------------
    # Full analysis (called from main pipeline or standalone)
    # ------------------------------------------------------------------

    def run_full_analysis(self, X_test, y_test=None, max_samples=2000):
        """
        Run complete SHAP analysis: compute values, generate all plots,
        return report text for inclusion in FUHRLANDER_MODEL_REPORT.md.

        Args:
            X_test: Test feature matrix, shape (N, 27)
            y_test: Optional true labels for context
            max_samples: Max samples for SHAP (default 2000)

        Returns:
            str: Markdown-formatted report section
        """
        print("\n" + "=" * 90)
        print("SHAP EXPLAINABILITY ANALYSIS")
        print("=" * 90)

        # Compute SHAP values
        self.compute_shap_values(X_test, max_samples=max_samples)

        # Generate all plots
        saved_files = self.plot_all(y=y_test)

        # Get global importance table
        importance_df = self.get_global_importance()

        # Explain a few representative samples
        print("\n  Explaining representative samples...")
        # Pick one healthy, one pre-fault, one fault (if labels available)
        sample_explanations = []
        if y_test is not None:
            y_arr = np.asarray(y_test)
            for label, label_name in [(0, 'Healthy'), (1, 'Pre-Fault'), (2, 'Fault')]:
                indices = np.where(y_arr == label)[0]
                if len(indices) > 0:
                    idx = indices[len(indices) // 2]  # Middle sample
                    result = self.explain_single_sample(
                        X_test[idx:idx+1] if isinstance(X_test, np.ndarray)
                        else X_test.iloc[idx:idx+1],
                        sample_id=f"{label_name}_{idx}",
                        save_plot=True,
                        verbose=True
                    )
                    sample_explanations.append((label_name, idx, result))
        else:
            # No labels - just explain first, middle, last
            for pos, name in [(0, 'first'), (len(X_test)//2, 'middle'),
                              (len(X_test)-1, 'last')]:
                result = self.explain_single_sample(
                    X_test[pos:pos+1] if isinstance(X_test, np.ndarray)
                    else X_test.iloc[pos:pos+1],
                    sample_id=f"{name}_{pos}",
                    save_plot=True,
                    verbose=True
                )
                sample_explanations.append((name, pos, result))

        # Build report text
        report = self._build_report_section(
            importance_df, sample_explanations, saved_files
        )

        # Save standalone SHAP report
        report_path = os.path.join(self.output_dir, 'SHAP_ANALYSIS_REPORT.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n  Saved: SHAP_ANALYSIS_REPORT.md")

        return report

    def _build_report_section(self, importance_df, sample_explanations,
                              saved_files):
        """Build markdown report section for SHAP analysis."""
        # Global importance table
        imp_table = "| Rank | Feature | SHAP Importance (Binary) |"
        if 'multi_shap_Fault' in importance_df.columns:
            imp_table += " SHAP Importance (Fault Class) |"
        imp_table += "\n|---|---|---|"
        if 'multi_shap_Fault' in importance_df.columns:
            imp_table += "---|"
        imp_table += "\n"

        for i, row in importance_df.head(20).iterrows():
            imp_table += f"| {i+1} | `{row['feature']}` | {row['binary_shap_importance']:.4f} |"
            if 'multi_shap_Fault' in importance_df.columns:
                imp_table += f" {row['multi_shap_Fault']:.4f} |"
            imp_table += "\n"

        # MDI vs SHAP comparison
        mdi_imp = self.binary_clf.feature_importances_
        shap_imp = importance_df.set_index('feature')['binary_shap_importance']

        mdi_top5 = np.argsort(mdi_imp)[::-1][:5]
        shap_top5 = importance_df.head(5)['feature'].tolist()

        comparison = "| Rank | MDI (Gini) Top Feature | SHAP Top Feature |\n"
        comparison += "|---|---|---|\n"
        for i in range(5):
            mdi_feat = self.feature_names[mdi_top5[i]]
            shap_feat = shap_top5[i]
            comparison += f"| {i+1} | `{mdi_feat}` | `{shap_feat}` |\n"

        # Sample explanations
        sample_text = ""
        for label_name, idx, result in sample_explanations:
            sample_text += f"\n**{label_name} sample (index {idx}):**\n"
            sample_text += f"- Binary: P(anomalous) = {result['binary_probability'][1]:.3f}\n"
            sample_text += f"- Multi-class: P = [H:{result['multi_probability'][0]:.3f}, "
            sample_text += f"PF:{result['multi_probability'][1]:.3f}, "
            sample_text += f"F:{result['multi_probability'][2]:.3f}]\n"
            sample_text += f"- Top features pushing toward fault:\n"
            for feat, sv, fv in result['top_features_pushing_fault'][:3]:
                sample_text += f"  - `{feat}`: SHAP={sv:+.4f} (value={fv:.2f})\n"
            sample_text += f"- Top features pushing toward healthy:\n"
            for feat, sv, fv in result['top_features_pushing_healthy'][:3]:
                sample_text += f"  - `{feat}`: SHAP={sv:+.4f} (value={fv:.2f})\n"

        files_list = "\n".join(f"- `{os.path.basename(f)}`" for f in saved_files)

        return f"""# SHAP Explainability Analysis Report
## Fuhrlander FL2500 Predictive Maintenance

### Method

This analysis uses **SHAP (SHapley Additive exPlanations)** to explain model
predictions. Unlike the MDI (Mean Decrease in Impurity / Gini importance) used
in the base model, SHAP provides:

1. **Consistent feature attribution** - based on game-theoretic Shapley values
2. **Direction of effect** - shows whether a feature pushes prediction toward
   fault or healthy
3. **Per-sample explanations** - not just global ranking, but why each
   individual prediction was made
4. **Interaction detection** - captures how features work together

SHAP TreeExplainer is used for both:
- **Binary GradientBoosting** (Healthy vs Anomalous)
- **Multi-class RandomForest** (Healthy / Pre-Fault / Fault)

### Global Feature Importance (SHAP)

{imp_table}

### MDI (Gini) vs SHAP Comparison

{comparison}

**Key differences between MDI and SHAP:**
- MDI measures how much each feature reduces impurity across all tree splits
  (computed during training). It is biased toward high-cardinality features.
- SHAP measures the marginal contribution of each feature to each individual
  prediction (computed post-hoc). It provides both magnitude AND direction.
- Where MDI and SHAP agree on a feature's importance, there is high confidence
  that the feature is genuinely predictive.
- Where they disagree, SHAP is generally more reliable because it accounts for
  feature correlations and interaction effects.

### Per-Sample Explanations
{sample_text}

### Generated Plots

{files_list}

### Usage in Agentic / Demo Setup

```python
from shap_explainer import SHAPExplainer

# Initialize with trained models
explainer = SHAPExplainer(
    binary_clf=binary_clf,
    multi_clf=multi_clf,
    feature_names=ALL_FEATURES,
    output_dir='RUL/'
)

# Explain a single new SCADA observation (for real-time demo)
result = explainer.explain_single_sample(
    X_new,           # shape (1, 27) - one hourly observation
    sample_id="demo_turbine_83_hour_5000",
    save_plot=True,
    verbose=True
)

# Access structured results
print(result['explanation_text'])        # Human-readable
print(result['binary_probability'])      # [p_healthy, p_anomalous]
print(result['multi_probability'])       # [p_healthy, p_prefault, p_fault]
print(result['top_features_pushing_fault'])   # [(feat, shap, val), ...]
print(result['top_features_pushing_healthy']) # [(feat, shap, val), ...]

# Full batch analysis on test set
report_md = explainer.run_full_analysis(X_test, y_test, max_samples=2000)
```

---
*Generated by SHAP Explainability Module*
*SHAP method: TreeExplainer (exact Shapley values for tree ensembles)*
"""


# =====================================================================
# Convenience function for calling from main pipeline
# =====================================================================

def run_shap_analysis(binary_clf, multi_clf, feature_names,
                      X_test, y_test=None, output_dir=None,
                      max_samples=2000):
    """
    Convenience function to run full SHAP analysis.
    Called from wind_turbine_pm_fuhrlander.py main().

    Args:
        binary_clf: Trained GradientBoostingClassifier
        multi_clf: Trained RandomForestClassifier
        feature_names: List of 27 feature names
        X_test: Test data, shape (N, 27)
        y_test: Optional test labels
        output_dir: Where to save plots
        max_samples: Max samples for SHAP

    Returns:
        SHAPExplainer instance (for further use in agentic setup)
    """
    explainer = SHAPExplainer(
        binary_clf=binary_clf,
        multi_clf=multi_clf,
        feature_names=feature_names,
        output_dir=output_dir or os.path.dirname(__file__)
    )
    explainer.run_full_analysis(X_test, y_test, max_samples=max_samples)
    return explainer
