"""
Bias Detection and Fairness Metrics

This module implements various fairness metrics and bias detection techniques
for machine learning models across different demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


@dataclass
class FairnessReport:
    """Container for fairness analysis results."""
    metric_name: str
    overall_value: float
    group_values: Dict[str, float]
    is_fair: bool
    threshold: float
    details: Dict[str, Any]


class FairnessMetrics:
    """
    Calculate various fairness metrics for machine learning models.

    Supports metrics including:
    - Demographic Parity (Statistical Parity)
    - Equal Opportunity
    - Equalized Odds
    - Predictive Parity
    - Calibration
    - Disparate Impact
    """

    @staticmethod
    def demographic_parity(y_pred: np.ndarray, sensitive_feature: np.ndarray) -> Dict[str, float]:
        """
        Calculate demographic parity (statistical parity).

        Measures if positive prediction rate is equal across groups.
        Fair if P(Ŷ=1|A=a) = P(Ŷ=1|A=b) for all groups a, b

        Args:
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute (e.g., race, gender)

        Returns:
            Dictionary with positive rate for each group
        """
        groups = np.unique(sensitive_feature)
        positive_rates = {}

        for group in groups:
            group_mask = sensitive_feature == group
            positive_rate = np.mean(y_pred[group_mask])
            positive_rates[str(group)] = positive_rate

        return positive_rates

    @staticmethod
    def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray,
                         sensitive_feature: np.ndarray) -> Dict[str, float]:
        """
        Calculate equal opportunity (equal TPR across groups).

        Measures if true positive rate is equal across groups.
        Fair if P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b) for all groups a, b

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute

        Returns:
            Dictionary with TPR for each group
        """
        groups = np.unique(sensitive_feature)
        tpr_scores = {}

        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # Calculate TPR (True Positive Rate)
            positive_mask = y_true_group == 1
            if np.sum(positive_mask) > 0:
                tpr = np.mean(y_pred_group[positive_mask])
            else:
                tpr = 0.0

            tpr_scores[str(group)] = tpr

        return tpr_scores

    @staticmethod
    def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray,
                       sensitive_feature: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate equalized odds (equal TPR and FPR across groups).

        Fair if both TPR and FPR are equal across groups:
        - P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b)  (Equal TPR)
        - P(Ŷ=1|Y=0,A=a) = P(Ŷ=1|Y=0,A=b)  (Equal FPR)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute

        Returns:
            Dictionary with TPR and FPR for each group
        """
        groups = np.unique(sensitive_feature)
        results = {}

        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # Calculate TPR (True Positive Rate)
            positive_mask = y_true_group == 1
            if np.sum(positive_mask) > 0:
                tpr = np.mean(y_pred_group[positive_mask])
            else:
                tpr = 0.0

            # Calculate FPR (False Positive Rate)
            negative_mask = y_true_group == 0
            if np.sum(negative_mask) > 0:
                fpr = np.mean(y_pred_group[negative_mask])
            else:
                fpr = 0.0

            results[str(group)] = {'tpr': tpr, 'fpr': fpr}

        return results

    @staticmethod
    def predictive_parity(y_true: np.ndarray, y_pred: np.ndarray,
                          sensitive_feature: np.ndarray) -> Dict[str, float]:
        """
        Calculate predictive parity (equal PPV across groups).

        Measures if positive predictive value is equal across groups.
        Fair if P(Y=1|Ŷ=1,A=a) = P(Y=1|Ŷ=1,A=b) for all groups a, b

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute

        Returns:
            Dictionary with PPV (precision) for each group
        """
        groups = np.unique(sensitive_feature)
        ppv_scores = {}

        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            # Calculate PPV (Positive Predictive Value / Precision)
            predicted_positive = y_pred_group == 1
            if np.sum(predicted_positive) > 0:
                ppv = np.mean(y_true_group[predicted_positive])
            else:
                ppv = 0.0

            ppv_scores[str(group)] = ppv

        return ppv_scores

    @staticmethod
    def disparate_impact(y_pred: np.ndarray, sensitive_feature: np.ndarray,
                        privileged_group: Any) -> float:
        """
        Calculate disparate impact ratio.

        Ratio of positive prediction rates between unprivileged and privileged groups.
        A ratio < 0.8 or > 1.25 indicates potential bias (80% rule).

        Args:
            y_pred: Predicted labels
            sensitive_feature: Sensitive attribute
            privileged_group: Value indicating the privileged group

        Returns:
            Disparate impact ratio
        """
        privileged_mask = sensitive_feature == privileged_group
        unprivileged_mask = ~privileged_mask

        privileged_positive_rate = np.mean(y_pred[privileged_mask])
        unprivileged_positive_rate = np.mean(y_pred[unprivileged_mask])

        if privileged_positive_rate == 0:
            return 0.0

        return unprivileged_positive_rate / privileged_positive_rate

    @staticmethod
    def calibration_by_group(y_true: np.ndarray, y_proba: np.ndarray,
                            sensitive_feature: np.ndarray, n_bins: int = 10) -> Dict[str, Dict]:
        """
        Calculate calibration metrics by group.

        Measures how well predicted probabilities match actual outcomes for each group.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            sensitive_feature: Sensitive attribute
            n_bins: Number of bins for calibration

        Returns:
            Dictionary with calibration data for each group
        """
        groups = np.unique(sensitive_feature)
        calibration_data = {}

        for group in groups:
            group_mask = sensitive_feature == group
            y_true_group = y_true[group_mask]
            y_proba_group = y_proba[group_mask]

            # Create probability bins
            bins = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(y_proba_group, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)

            mean_predicted_proba = []
            mean_actual_proba = []
            counts = []

            for i in range(n_bins):
                bin_mask = bin_indices == i
                if np.sum(bin_mask) > 0:
                    mean_predicted_proba.append(np.mean(y_proba_group[bin_mask]))
                    mean_actual_proba.append(np.mean(y_true_group[bin_mask]))
                    counts.append(np.sum(bin_mask))

            calibration_data[str(group)] = {
                'mean_predicted': mean_predicted_proba,
                'mean_actual': mean_actual_proba,
                'counts': counts
            }

        return calibration_data


class BiasDetector:
    """
    Detect and analyze bias in machine learning models.

    This class provides comprehensive bias detection across multiple fairness metrics
    and generates detailed reports with visualizations.
    """

    def __init__(self, sensitive_features: List[str],
                 fairness_threshold: float = 0.1):
        """
        Initialize the BiasDetector.

        Args:
            sensitive_features: List of column names representing sensitive attributes
            fairness_threshold: Maximum allowed difference between groups (default: 0.1)
        """
        self.sensitive_features = sensitive_features
        self.fairness_threshold = fairness_threshold
        self.reports: List[FairnessReport] = []

    def analyze_bias(self, y_true: np.ndarray, y_pred: np.ndarray,
                    sensitive_data: pd.DataFrame, y_proba: Optional[np.ndarray] = None,
                    privileged_groups: Optional[Dict[str, Any]] = None) -> List[FairnessReport]:
        """
        Perform comprehensive bias analysis across all sensitive features.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_data: DataFrame containing sensitive attributes
            y_proba: Predicted probabilities (optional, for calibration)
            privileged_groups: Dictionary mapping feature names to privileged group values

        Returns:
            List of FairnessReport objects
        """
        self.reports = []

        for feature in self.sensitive_features:
            if feature not in sensitive_data.columns:
                continue

            sensitive_feature = sensitive_data[feature].values

            # Demographic Parity
            dp_scores = FairnessMetrics.demographic_parity(y_pred, sensitive_feature)
            dp_diff = max(dp_scores.values()) - min(dp_scores.values())
            self.reports.append(FairnessReport(
                metric_name=f"Demographic Parity ({feature})",
                overall_value=dp_diff,
                group_values=dp_scores,
                is_fair=dp_diff <= self.fairness_threshold,
                threshold=self.fairness_threshold,
                details={'metric_type': 'demographic_parity'}
            ))

            # Equal Opportunity
            eo_scores = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive_feature)
            eo_diff = max(eo_scores.values()) - min(eo_scores.values())
            self.reports.append(FairnessReport(
                metric_name=f"Equal Opportunity ({feature})",
                overall_value=eo_diff,
                group_values=eo_scores,
                is_fair=eo_diff <= self.fairness_threshold,
                threshold=self.fairness_threshold,
                details={'metric_type': 'equal_opportunity'}
            ))

            # Equalized Odds
            eq_odds = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive_feature)
            tpr_values = [v['tpr'] for v in eq_odds.values()]
            fpr_values = [v['fpr'] for v in eq_odds.values()]
            tpr_diff = max(tpr_values) - min(tpr_values)
            fpr_diff = max(fpr_values) - min(fpr_values)
            eq_odds_diff = max(tpr_diff, fpr_diff)
            self.reports.append(FairnessReport(
                metric_name=f"Equalized Odds ({feature})",
                overall_value=eq_odds_diff,
                group_values=eq_odds,
                is_fair=eq_odds_diff <= self.fairness_threshold,
                threshold=self.fairness_threshold,
                details={'metric_type': 'equalized_odds', 'tpr_diff': tpr_diff, 'fpr_diff': fpr_diff}
            ))

            # Predictive Parity
            pp_scores = FairnessMetrics.predictive_parity(y_true, y_pred, sensitive_feature)
            pp_diff = max(pp_scores.values()) - min(pp_scores.values())
            self.reports.append(FairnessReport(
                metric_name=f"Predictive Parity ({feature})",
                overall_value=pp_diff,
                group_values=pp_scores,
                is_fair=pp_diff <= self.fairness_threshold,
                threshold=self.fairness_threshold,
                details={'metric_type': 'predictive_parity'}
            ))

            # Disparate Impact (if privileged group specified)
            if privileged_groups and feature in privileged_groups:
                di_ratio = FairnessMetrics.disparate_impact(
                    y_pred, sensitive_feature, privileged_groups[feature]
                )
                # Fair if ratio is between 0.8 and 1.25 (80% rule)
                is_fair = 0.8 <= di_ratio <= 1.25
                self.reports.append(FairnessReport(
                    metric_name=f"Disparate Impact ({feature})",
                    overall_value=di_ratio,
                    group_values={'ratio': di_ratio},
                    is_fair=is_fair,
                    threshold=0.8,  # Lower bound of 80% rule
                    details={'metric_type': 'disparate_impact', 'upper_bound': 1.25}
                ))

            # Calibration (if probabilities provided)
            if y_proba is not None:
                calib_data = FairnessMetrics.calibration_by_group(
                    y_true, y_proba, sensitive_feature
                )
                # Calculate calibration error for each group
                calib_errors = {}
                for group, data in calib_data.items():
                    if len(data['mean_predicted']) > 0:
                        error = np.mean(np.abs(
                            np.array(data['mean_predicted']) - np.array(data['mean_actual'])
                        ))
                        calib_errors[group] = error

                if calib_errors:
                    calib_diff = max(calib_errors.values()) - min(calib_errors.values())
                    self.reports.append(FairnessReport(
                        metric_name=f"Calibration ({feature})",
                        overall_value=calib_diff,
                        group_values=calib_errors,
                        is_fair=calib_diff <= self.fairness_threshold,
                        threshold=self.fairness_threshold,
                        details={'metric_type': 'calibration', 'calibration_data': calib_data}
                    ))

        return self.reports

    def get_bias_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all bias metrics.

        Returns:
            DataFrame with bias analysis results
        """
        if not self.reports:
            return pd.DataFrame()

        summary_data = []
        for report in self.reports:
            summary_data.append({
                'Metric': report.metric_name,
                'Value': report.overall_value,
                'Threshold': report.threshold,
                'Is Fair': report.is_fair,
                'Status': '✓ Fair' if report.is_fair else '✗ Biased'
            })

        return pd.DataFrame(summary_data)

    def plot_fairness_metrics(self, save_path: Optional[str] = None):
        """
        Visualize fairness metrics across all sensitive features.

        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.reports:
            print("No bias analysis results available. Run analyze_bias() first.")
            return

        # Group reports by feature
        feature_reports = {}
        for report in self.reports:
            # Extract feature name from metric name
            feature = report.metric_name.split('(')[1].rstrip(')')
            if feature not in feature_reports:
                feature_reports[feature] = []
            feature_reports[feature].append(report)

        n_features = len(feature_reports)
        fig, axes = plt.subplots(n_features, 1, figsize=(12, 4 * n_features))
        if n_features == 1:
            axes = [axes]

        for idx, (feature, reports) in enumerate(feature_reports.items()):
            ax = axes[idx]

            metric_names = []
            metric_values = []
            colors = []

            for report in reports:
                metric_type = report.details.get('metric_type', 'unknown')
                metric_names.append(metric_type.replace('_', ' ').title())
                metric_values.append(report.overall_value)
                colors.append('green' if report.is_fair else 'red')

            bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
            ax.axvline(x=self.fairness_threshold, color='orange',
                      linestyle='--', label=f'Threshold ({self.fairness_threshold})')
            ax.set_xlabel('Bias Metric Value')
            ax.set_title(f'Fairness Metrics for: {feature}')
            ax.legend()
            ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_group_comparison(self, metric_type: str = 'demographic_parity',
                            save_path: Optional[str] = None):
        """
        Plot comparison of metric values across groups for a specific metric type.

        Args:
            metric_type: Type of metric to plot
            save_path: Path to save the plot (optional)
        """
        # Filter reports by metric type
        relevant_reports = [r for r in self.reports
                          if r.details.get('metric_type') == metric_type]

        if not relevant_reports:
            print(f"No reports found for metric type: {metric_type}")
            return

        n_reports = len(relevant_reports)
        fig, axes = plt.subplots(1, n_reports, figsize=(6 * n_reports, 5))
        if n_reports == 1:
            axes = [axes]

        for idx, report in enumerate(relevant_reports):
            ax = axes[idx]

            groups = list(report.group_values.keys())
            values = list(report.group_values.values())

            # Handle nested dictionaries (like equalized_odds)
            if isinstance(values[0], dict):
                # Plot TPR and FPR separately
                tpr_values = [v['tpr'] for v in values]
                fpr_values = [v['fpr'] for v in values]

                x = np.arange(len(groups))
                width = 0.35

                ax.bar(x - width/2, tpr_values, width, label='TPR', alpha=0.8)
                ax.bar(x + width/2, fpr_values, width, label='FPR', alpha=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(groups)
                ax.legend()
            else:
                colors = ['green' if report.is_fair else 'red' for _ in groups]
                ax.bar(groups, values, color=colors, alpha=0.7)

            ax.set_ylabel('Metric Value')
            ax.set_title(report.metric_name)
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def generate_report(self, output_path: str, format: str = 'html'):
        """
        Generate a comprehensive bias report.

        Args:
            output_path: Path to save the report
            format: Report format ('html', 'markdown', or 'text')
        """
        if not self.reports:
            print("No bias analysis results available. Run analyze_bias() first.")
            return

        summary_df = self.get_bias_summary()

        if format == 'html':
            html_content = f"""
            <html>
            <head>
                <title>Fairness and Bias Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .fair {{ color: green; font-weight: bold; }}
                    .biased {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Fairness and Bias Analysis Report</h1>
                <p>Generated bias analysis across {len(self.sensitive_features)} sensitive feature(s)</p>

                <h2>Summary</h2>
                {summary_df.to_html(index=False, escape=False)}

                <h2>Detailed Results</h2>
            """

            for report in self.reports:
                status_class = 'fair' if report.is_fair else 'biased'
                html_content += f"""
                <h3>{report.metric_name}</h3>
                <p>Overall Value: <strong>{report.overall_value:.4f}</strong></p>
                <p>Threshold: {report.threshold}</p>
                <p class="{status_class}">{report.details.get('metric_type', 'N/A').replace('_', ' ').title()}: {'Fair' if report.is_fair else 'Biased'}</p>
                <p>Group Values:</p>
                <ul>
                """
                for group, value in report.group_values.items():
                    html_content += f"<li>{group}: {value}</li>"
                html_content += "</ul>"

            html_content += """
            </body>
            </html>
            """

            with open(output_path, 'w') as f:
                f.write(html_content)

        elif format == 'markdown':
            md_content = f"# Fairness and Bias Analysis Report\n\n"
            md_content += f"Generated bias analysis across {len(self.sensitive_features)} sensitive feature(s)\n\n"
            md_content += "## Summary\n\n"
            md_content += summary_df.to_markdown(index=False) + "\n\n"
            md_content += "## Detailed Results\n\n"

            for report in self.reports:
                md_content += f"### {report.metric_name}\n\n"
                md_content += f"- **Overall Value**: {report.overall_value:.4f}\n"
                md_content += f"- **Threshold**: {report.threshold}\n"
                md_content += f"- **Status**: {'✓ Fair' if report.is_fair else '✗ Biased'}\n"
                md_content += f"- **Group Values**:\n"
                for group, value in report.group_values.items():
                    md_content += f"  - {group}: {value}\n"
                md_content += "\n"

            with open(output_path, 'w') as f:
                f.write(md_content)

        else:  # text format
            with open(output_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FAIRNESS AND BIAS ANALYSIS REPORT\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Sensitive Features Analyzed: {', '.join(self.sensitive_features)}\n")
                f.write(f"Fairness Threshold: {self.fairness_threshold}\n\n")
                f.write(summary_df.to_string(index=False))
                f.write("\n\n")

                for report in self.reports:
                    f.write("-" * 80 + "\n")
                    f.write(f"{report.metric_name}\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Overall Value: {report.overall_value:.4f}\n")
                    f.write(f"Threshold: {report.threshold}\n")
                    f.write(f"Status: {'Fair' if report.is_fair else 'Biased'}\n")
                    f.write(f"Group Values:\n")
                    for group, value in report.group_values.items():
                        f.write(f"  {group}: {value}\n")
                    f.write("\n")

        print(f"Report generated: {output_path}")
