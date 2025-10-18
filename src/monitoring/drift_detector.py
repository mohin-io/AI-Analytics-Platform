"""
Drift Detection Module

Implements various drift detection techniques including:
- Data drift (distribution changes in features)
- Concept drift (changes in relationship between features and target)
- Prediction drift (changes in model outputs)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.metrics import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


@dataclass
class DriftReport:
    """Container for drift detection results."""
    feature_name: str
    drift_score: float
    p_value: float
    is_drift: bool
    threshold: float
    drift_type: str
    details: Dict[str, Any]


class DriftDetector:
    """
    Detect various types of drift in data and predictions.

    Supports multiple statistical tests:
    - Kolmogorov-Smirnov test (continuous features)
    - Chi-square test (categorical features)
    - Population Stability Index (PSI)
    - Jensen-Shannon divergence
    """

    def __init__(self, threshold: float = 0.05, psi_threshold: float = 0.1):
        """
        Initialize drift detector.

        Args:
            threshold: P-value threshold for statistical tests
            psi_threshold: Threshold for PSI (0.1 = small shift, 0.2 = significant shift)
        """
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Any] = {}

    def fit(self, reference_data: pd.DataFrame):
        """
        Fit the drift detector on reference data (baseline).

        Args:
            reference_data: Reference dataset to compare against
        """
        self.reference_data = reference_data.copy()

        # Calculate statistics for each feature
        for col in reference_data.columns:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                self.reference_stats[col] = {
                    'type': 'numeric',
                    'mean': reference_data[col].mean(),
                    'std': reference_data[col].std(),
                    'min': reference_data[col].min(),
                    'max': reference_data[col].max(),
                    'quantiles': reference_data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                self.reference_stats[col] = {
                    'type': 'categorical',
                    'value_counts': reference_data[col].value_counts().to_dict(),
                    'unique_values': reference_data[col].nunique()
                }

    def detect_feature_drift(self, current_data: pd.DataFrame,
                            features: Optional[List[str]] = None) -> List[DriftReport]:
        """
        Detect drift in individual features.

        Args:
            current_data: Current dataset to check for drift
            features: List of features to check (None = all features)

        Returns:
            List of DriftReport objects
        """
        if self.reference_data is None:
            raise ValueError("Must fit detector on reference data first")

        if features is None:
            features = self.reference_data.columns.tolist()

        reports = []

        for feature in features:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue

            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                continue

            if pd.api.types.is_numeric_dtype(ref_values):
                # Use Kolmogorov-Smirnov test for numeric features
                statistic, p_value = ks_2samp(ref_values, curr_values)
                is_drift = p_value < self.threshold

                # Also calculate PSI
                psi_score = self._calculate_psi(ref_values, curr_values)

                reports.append(DriftReport(
                    feature_name=feature,
                    drift_score=statistic,
                    p_value=p_value,
                    is_drift=is_drift or (psi_score > self.psi_threshold),
                    threshold=self.threshold,
                    drift_type='KS-Test',
                    details={
                        'psi': psi_score,
                        'ref_mean': ref_values.mean(),
                        'curr_mean': curr_values.mean(),
                        'ref_std': ref_values.std(),
                        'curr_std': curr_values.std()
                    }
                ))

            else:
                # Use Chi-square test for categorical features
                chi2, p_value = self._chi_square_test(ref_values, curr_values)
                is_drift = p_value < self.threshold

                # Also calculate PSI for categorical
                psi_score = self._calculate_psi_categorical(ref_values, curr_values)

                reports.append(DriftReport(
                    feature_name=feature,
                    drift_score=chi2,
                    p_value=p_value,
                    is_drift=is_drift or (psi_score > self.psi_threshold),
                    threshold=self.threshold,
                    drift_type='Chi-Square',
                    details={
                        'psi': psi_score,
                        'ref_categories': ref_values.nunique(),
                        'curr_categories': curr_values.nunique()
                    }
                ))

        return reports

    def detect_prediction_drift(self, ref_predictions: np.ndarray,
                               curr_predictions: np.ndarray,
                               task_type: str = 'classification') -> DriftReport:
        """
        Detect drift in model predictions.

        Args:
            ref_predictions: Reference predictions
            curr_predictions: Current predictions
            task_type: 'classification' or 'regression'

        Returns:
            DriftReport object
        """
        if task_type == 'classification':
            # Chi-square test for classification
            chi2, p_value = self._chi_square_test(ref_predictions, curr_predictions)
            psi_score = self._calculate_psi_categorical(ref_predictions, curr_predictions)

            return DriftReport(
                feature_name='predictions',
                drift_score=chi2,
                p_value=p_value,
                is_drift=p_value < self.threshold or psi_score > self.psi_threshold,
                threshold=self.threshold,
                drift_type='Prediction-Drift-Classification',
                details={
                    'psi': psi_score,
                    'ref_distribution': pd.Series(ref_predictions).value_counts().to_dict(),
                    'curr_distribution': pd.Series(curr_predictions).value_counts().to_dict()
                }
            )
        else:
            # KS test for regression
            statistic, p_value = ks_2samp(ref_predictions, curr_predictions)
            psi_score = self._calculate_psi(ref_predictions, curr_predictions)

            return DriftReport(
                feature_name='predictions',
                drift_score=statistic,
                p_value=p_value,
                is_drift=p_value < self.threshold or psi_score > self.psi_threshold,
                threshold=self.threshold,
                drift_type='Prediction-Drift-Regression',
                details={
                    'psi': psi_score,
                    'ref_mean': np.mean(ref_predictions),
                    'curr_mean': np.mean(curr_predictions),
                    'ref_std': np.std(ref_predictions),
                    'curr_std': np.std(curr_predictions)
                }
            )

    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray,
                       n_bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for numeric features.

        PSI measures the change in distribution between two samples.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Small change
        PSI >= 0.2: Significant change

        Args:
            reference: Reference data
            current: Current data
            n_bins: Number of bins for discretization

        Returns:
            PSI score
        """
        # Create bins based on reference data
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)  # Remove duplicates

        if len(bins) <= 1:
            return 0.0

        # Bin both datasets
        ref_binned = np.digitize(reference, bins[:-1])
        curr_binned = np.digitize(current, bins[:-1])

        # Calculate proportions
        ref_counts = np.bincount(ref_binned, minlength=len(bins))
        curr_counts = np.bincount(curr_binned, minlength=len(bins))

        ref_props = ref_counts / len(reference)
        curr_props = curr_counts / len(current)

        # Calculate PSI
        psi = 0
        for i in range(len(ref_props)):
            if ref_props[i] > 0 and curr_props[i] > 0:
                psi += (curr_props[i] - ref_props[i]) * np.log(curr_props[i] / ref_props[i])
            elif curr_props[i] > 0:
                # Current has values in bin where reference doesn't
                psi += curr_props[i] * np.log(curr_props[i] / 0.0001)

        return psi

    def _calculate_psi_categorical(self, reference: pd.Series,
                                   current: pd.Series) -> float:
        """
        Calculate PSI for categorical features.

        Args:
            reference: Reference data
            current: Current data

        Returns:
            PSI score
        """
        ref_counts = reference.value_counts(normalize=True)
        curr_counts = current.value_counts(normalize=True)

        # Get all categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)

        psi = 0
        for category in all_categories:
            ref_prop = ref_counts.get(category, 0.0001)  # Small value for missing categories
            curr_prop = curr_counts.get(category, 0.0001)

            psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)

        return psi

    def _chi_square_test(self, reference: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Perform chi-square test for categorical features.

        Args:
            reference: Reference data
            current: Current data

        Returns:
            Tuple of (chi-square statistic, p-value)
        """
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()

        # Get all categories
        all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))

        # Create contingency table
        observed = []
        expected = []

        for category in all_categories:
            curr_count = curr_counts.get(category, 0)
            ref_count = ref_counts.get(category, 0)

            observed.append(curr_count)
            expected.append(ref_count)

        # Normalize expected to match observed sum
        expected = np.array(expected)
        if expected.sum() > 0:
            expected = expected * (sum(observed) / expected.sum())

        # Perform chi-square test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2, p_value = stats.chisquare(observed, expected)

        return chi2, p_value


class DataDriftAnalyzer:
    """
    Comprehensive data drift analysis with visualization.
    """

    def __init__(self, drift_detector: DriftDetector):
        """
        Initialize analyzer.

        Args:
            drift_detector: Fitted DriftDetector instance
        """
        self.drift_detector = drift_detector
        self.drift_reports: List[DriftReport] = []

    def analyze(self, current_data: pd.DataFrame,
                features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze drift across all features.

        Args:
            current_data: Current dataset
            features: Features to analyze (None = all)

        Returns:
            DataFrame with drift analysis results
        """
        self.drift_reports = self.drift_detector.detect_feature_drift(current_data, features)

        # Create summary DataFrame
        summary_data = []
        for report in self.drift_reports:
            summary_data.append({
                'Feature': report.feature_name,
                'Drift Score': report.drift_score,
                'P-Value': report.p_value,
                'PSI': report.details.get('psi', 0),
                'Drift Detected': report.is_drift,
                'Test Type': report.drift_type
            })

        return pd.DataFrame(summary_data)

    def plot_drift_summary(self, save_path: Optional[str] = None):
        """
        Plot summary of drift detection results.

        Args:
            save_path: Path to save plot (optional)
        """
        if not self.drift_reports:
            print("No drift reports available. Run analyze() first.")
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Drift scores
        features = [r.feature_name for r in self.drift_reports]
        scores = [r.drift_score for r in self.drift_reports]
        colors = ['red' if r.is_drift else 'green' for r in self.drift_reports]

        axes[0].barh(features, scores, color=colors, alpha=0.7)
        axes[0].set_xlabel('Drift Score')
        axes[0].set_title('Drift Detection Scores by Feature')
        axes[0].grid(axis='x', alpha=0.3)

        # Plot 2: PSI scores
        psi_scores = [r.details.get('psi', 0) for r in self.drift_reports]
        axes[1].barh(features, psi_scores, color=colors, alpha=0.7)
        axes[1].axvline(x=self.drift_detector.psi_threshold, color='orange',
                       linestyle='--', label=f'PSI Threshold ({self.drift_detector.psi_threshold})')
        axes[1].set_xlabel('PSI Score')
        axes[1].set_title('Population Stability Index by Feature')
        axes[1].legend()
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_feature_distributions(self, current_data: pd.DataFrame,
                                   feature: str, save_path: Optional[str] = None):
        """
        Plot distribution comparison for a specific feature.

        Args:
            current_data: Current dataset
            feature: Feature name to plot
            save_path: Path to save plot (optional)
        """
        if self.drift_detector.reference_data is None:
            print("No reference data available.")
            return

        if feature not in self.drift_detector.reference_data.columns:
            print(f"Feature '{feature}' not found in reference data.")
            return

        ref_values = self.drift_detector.reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()

        fig, ax = plt.subplots(figsize=(10, 6))

        if pd.api.types.is_numeric_dtype(ref_values):
            # Histogram for numeric features
            ax.hist(ref_values, bins=30, alpha=0.5, label='Reference', density=True)
            ax.hist(curr_values, bins=30, alpha=0.5, label='Current', density=True)
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
        else:
            # Bar chart for categorical features
            ref_counts = ref_values.value_counts(normalize=True)
            curr_counts = curr_values.value_counts(normalize=True)

            all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
            x = np.arange(len(all_categories))
            width = 0.35

            ref_props = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_props = [curr_counts.get(cat, 0) for cat in all_categories]

            ax.bar(x - width/2, ref_props, width, label='Reference', alpha=0.7)
            ax.bar(x + width/2, curr_props, width, label='Current', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(all_categories, rotation=45, ha='right')
            ax.set_ylabel('Proportion')

        ax.set_title(f'Distribution Comparison: {feature}')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def generate_report(self, output_path: str, format: str = 'html'):
        """
        Generate comprehensive drift report.

        Args:
            output_path: Path to save report
            format: Report format ('html', 'markdown', 'text')
        """
        if not self.drift_reports:
            print("No drift reports available. Run analyze() first.")
            return

        summary_df = self.analyze(pd.DataFrame())  # Use cached reports

        if format == 'html':
            html_content = f"""
            <html>
            <head>
                <title>Data Drift Detection Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .drift {{ color: red; font-weight: bold; }}
                    .no-drift {{ color: green; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Data Drift Detection Report</h1>
                <p>Analyzed {len(self.drift_reports)} feature(s) for drift</p>
                <p>Drift detected in {sum(1 for r in self.drift_reports if r.is_drift)} feature(s)</p>

                <h2>Summary</h2>
                {summary_df.to_html(index=False, escape=False)}

                <h2>Detailed Results</h2>
            """

            for report in self.drift_reports:
                status_class = 'no-drift' if not report.is_drift else 'drift'
                html_content += f"""
                <h3>{report.feature_name}</h3>
                <p>Test Type: {report.drift_type}</p>
                <p>Drift Score: <strong>{report.drift_score:.4f}</strong></p>
                <p>P-Value: {report.p_value:.4f}</p>
                <p>PSI: {report.details.get('psi', 0):.4f}</p>
                <p class="{status_class}">{'Drift Detected' if report.is_drift else 'No Drift'}</p>
                """

            html_content += """
            </body>
            </html>
            """

            with open(output_path, 'w') as f:
                f.write(html_content)

        print(f"Drift report generated: {output_path}")
