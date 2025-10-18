"""
Model Performance Monitoring

Track model performance over time and detect degradation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class PerformanceSnapshot:
    """Snapshot of model performance at a point in time."""
    timestamp: str
    metrics: Dict[str, float]
    data_size: int
    model_version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PerformanceTracker:
    """
    Track model performance metrics over time.
    """

    def __init__(self, task_type: str = 'classification'):
        """
        Initialize performance tracker.

        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.snapshots: List[PerformanceSnapshot] = []
        self.baseline_metrics: Optional[Dict[str, float]] = None

    def add_snapshot(self, y_true: np.ndarray, y_pred: np.ndarray,
                    model_version: Optional[str] = None,
                    y_proba: Optional[np.ndarray] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Add a performance snapshot.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_version: Model version identifier
            y_proba: Predicted probabilities (for classification)
            metadata: Additional metadata
        """
        # Calculate metrics based on task type
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            metrics = {
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'r2': r2_score(y_true, y_pred)
            }

        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            metrics=metrics,
            data_size=len(y_true),
            model_version=model_version,
            metadata=metadata
        )

        self.snapshots.append(snapshot)

        # Set baseline on first snapshot
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics.copy()

    def set_baseline(self, metrics: Dict[str, float]):
        """
        Manually set baseline metrics.

        Args:
            metrics: Baseline metric values
        """
        self.baseline_metrics = metrics.copy()

    def get_history(self) -> pd.DataFrame:
        """
        Get performance history as DataFrame.

        Returns:
            DataFrame with performance history
        """
        if not self.snapshots:
            return pd.DataFrame()

        records = []
        for snapshot in self.snapshots:
            record = {
                'timestamp': snapshot.timestamp,
                'model_version': snapshot.model_version,
                'data_size': snapshot.data_size,
                **snapshot.metrics
            }
            records.append(record)

        return pd.DataFrame(records)

    def detect_degradation(self, threshold: float = 0.05,
                          metric: Optional[str] = None) -> Dict[str, bool]:
        """
        Detect performance degradation compared to baseline.

        Args:
            threshold: Acceptable degradation threshold (e.g., 0.05 = 5%)
            metric: Specific metric to check (None = all metrics)

        Returns:
            Dictionary indicating degradation per metric
        """
        if not self.snapshots or self.baseline_metrics is None:
            return {}

        latest_snapshot = self.snapshots[-1]
        latest_metrics = latest_snapshot.metrics

        degradation = {}

        metrics_to_check = [metric] if metric else latest_metrics.keys()

        for metric_name in metrics_to_check:
            if metric_name not in self.baseline_metrics:
                continue

            baseline_value = self.baseline_metrics[metric_name]
            current_value = latest_metrics[metric_name]

            # For metrics where higher is better (accuracy, f1, r2, etc.)
            if metric_name in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
                degraded = (baseline_value - current_value) > threshold
            else:  # For metrics where lower is better (mae, mse, rmse)
                degraded = (current_value - baseline_value) > threshold

            degradation[metric_name] = degraded

        return degradation

    def calculate_trend(self, metric: str, window: int = 5) -> str:
        """
        Calculate trend for a specific metric.

        Args:
            metric: Metric name
            window: Number of recent snapshots to consider

        Returns:
            Trend description ('improving', 'stable', 'degrading')
        """
        if len(self.snapshots) < window:
            return 'insufficient_data'

        recent_snapshots = self.snapshots[-window:]
        values = [s.metrics.get(metric, 0) for s in recent_snapshots]

        # Calculate linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        # Determine trend based on metric type
        if metric in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
            # Higher is better
            if slope > 0.01:
                return 'improving'
            elif slope < -0.01:
                return 'degrading'
            else:
                return 'stable'
        else:
            # Lower is better (mae, mse, rmse)
            if slope < -0.01:
                return 'improving'
            elif slope > 0.01:
                return 'degrading'
            else:
                return 'stable'

    def plot_metrics_over_time(self, metrics: Optional[List[str]] = None,
                               save_path: Optional[str] = None):
        """
        Plot metric values over time.

        Args:
            metrics: List of metrics to plot (None = all)
            save_path: Path to save plot (optional)
        """
        if not self.snapshots:
            print("No snapshots available.")
            return

        df = self.get_history()

        if metrics is None:
            metrics = [col for col in df.columns
                      if col not in ['timestamp', 'model_version', 'data_size']]

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            # Convert timestamp to datetime for plotting
            timestamps = pd.to_datetime(df['timestamp'])
            values = df[metric]

            ax.plot(timestamps, values, marker='o', linestyle='-', linewidth=2, markersize=6)

            # Add baseline line if available
            if self.baseline_metrics and metric in self.baseline_metrics:
                ax.axhline(y=self.baseline_metrics[metric], color='green',
                          linestyle='--', label='Baseline', alpha=0.7)

            # Add threshold lines for degradation detection
            if self.baseline_metrics and metric in self.baseline_metrics:
                baseline = self.baseline_metrics[metric]
                if metric in ['accuracy', 'precision', 'recall', 'f1', 'r2']:
                    ax.axhline(y=baseline * 0.95, color='orange',
                              linestyle=':', label='Warning (-5%)', alpha=0.7)
                    ax.axhline(y=baseline * 0.90, color='red',
                              linestyle=':', label='Critical (-10%)', alpha=0.7)
                else:
                    ax.axhline(y=baseline * 1.05, color='orange',
                              linestyle=':', label='Warning (+5%)', alpha=0.7)
                    ax.axhline(y=baseline * 1.10, color='red',
                              linestyle=':', label='Critical (+10%)', alpha=0.7)

            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Over Time')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def export_history(self, output_path: str, format: str = 'csv'):
        """
        Export performance history.

        Args:
            output_path: Path to save file
            format: Export format ('csv', 'json', 'excel')
        """
        df = self.get_history()

        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Performance history exported to: {output_path}")


class ModelMonitor:
    """
    Comprehensive model monitoring system.

    Combines performance tracking with drift detection and alerting.
    """

    def __init__(self, task_type: str = 'classification',
                 performance_threshold: float = 0.05,
                 drift_threshold: float = 0.05):
        """
        Initialize model monitor.

        Args:
            task_type: 'classification' or 'regression'
            performance_threshold: Threshold for performance degradation
            drift_threshold: Threshold for drift detection
        """
        self.task_type = task_type
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold

        self.performance_tracker = PerformanceTracker(task_type)
        self.alerts: List[Dict[str, Any]] = []

    def log_prediction_batch(self, y_true: np.ndarray, y_pred: np.ndarray,
                            X: Optional[pd.DataFrame] = None,
                            y_proba: Optional[np.ndarray] = None,
                            model_version: Optional[str] = None):
        """
        Log a batch of predictions for monitoring.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            X: Feature data (for drift detection)
            y_proba: Predicted probabilities
            model_version: Model version identifier
        """
        # Track performance
        self.performance_tracker.add_snapshot(
            y_true, y_pred,
            model_version=model_version,
            y_proba=y_proba
        )

        # Check for performance degradation
        degradation = self.performance_tracker.detect_degradation(
            threshold=self.performance_threshold
        )

        # Generate alerts for degraded metrics
        for metric, is_degraded in degradation.items():
            if is_degraded:
                self._create_alert(
                    alert_type='performance_degradation',
                    severity='warning',
                    message=f"{metric} has degraded beyond threshold",
                    details={
                        'metric': metric,
                        'threshold': self.performance_threshold
                    }
                )

    def _create_alert(self, alert_type: str, severity: str,
                     message: str, details: Optional[Dict[str, Any]] = None):
        """
        Create an alert.

        Args:
            alert_type: Type of alert
            severity: Alert severity ('info', 'warning', 'critical')
            message: Alert message
            details: Additional details
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {}
        }

        self.alerts.append(alert)

    def get_alerts(self, severity: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get alerts.

        Args:
            severity: Filter by severity (None = all)
            limit: Maximum number of alerts to return

        Returns:
            List of alerts
        """
        filtered_alerts = self.alerts

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a['severity'] == severity]

        if limit:
            filtered_alerts = filtered_alerts[-limit:]

        return filtered_alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive monitoring summary.

        Returns:
            Dictionary with monitoring summary
        """
        history_df = self.performance_tracker.get_history()

        summary = {
            'total_snapshots': len(self.performance_tracker.snapshots),
            'total_alerts': len(self.alerts),
            'alert_breakdown': {
                'info': len([a for a in self.alerts if a['severity'] == 'info']),
                'warning': len([a for a in self.alerts if a['severity'] == 'warning']),
                'critical': len([a for a in self.alerts if a['severity'] == 'critical'])
            },
            'latest_performance': None,
            'baseline_performance': self.performance_tracker.baseline_metrics
        }

        if len(self.performance_tracker.snapshots) > 0:
            latest = self.performance_tracker.snapshots[-1]
            summary['latest_performance'] = latest.metrics

        # Calculate trends
        if len(self.performance_tracker.snapshots) >= 5:
            trends = {}
            for metric in summary['latest_performance'].keys():
                trends[metric] = self.performance_tracker.calculate_trend(metric)
            summary['trends'] = trends

        return summary

    def generate_monitoring_report(self, output_path: str, format: str = 'html'):
        """
        Generate comprehensive monitoring report.

        Args:
            output_path: Path to save report
            format: Report format ('html', 'json')
        """
        summary = self.get_monitoring_summary()

        if format == 'html':
            html_content = f"""
            <html>
            <head>
                <title>Model Monitoring Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    h2 {{ color: #666; margin-top: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    .info {{ color: blue; }}
                    .warning {{ color: orange; font-weight: bold; }}
                    .critical {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <h1>Model Monitoring Report</h1>
                <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

                <h2>Summary</h2>
                <p>Total Snapshots: {summary['total_snapshots']}</p>
                <p>Total Alerts: {summary['total_alerts']}</p>
                <ul>
                    <li class="info">Info: {summary['alert_breakdown']['info']}</li>
                    <li class="warning">Warning: {summary['alert_breakdown']['warning']}</li>
                    <li class="critical">Critical: {summary['alert_breakdown']['critical']}</li>
                </ul>

                <h2>Current Performance</h2>
            """

            if summary['latest_performance']:
                html_content += "<table><tr><th>Metric</th><th>Value</th>"
                if 'trends' in summary:
                    html_content += "<th>Trend</th>"
                html_content += "</tr>"

                for metric, value in summary['latest_performance'].items():
                    html_content += f"<tr><td>{metric.upper()}</td><td>{value:.4f}</td>"
                    if 'trends' in summary:
                        trend = summary['trends'].get(metric, 'unknown')
                        html_content += f"<td>{trend}</td>"
                    html_content += "</tr>"

                html_content += "</table>"

            # Recent alerts
            if self.alerts:
                html_content += "<h2>Recent Alerts</h2><table>"
                html_content += "<tr><th>Timestamp</th><th>Type</th><th>Severity</th><th>Message</th></tr>"

                for alert in self.alerts[-10:]:  # Last 10 alerts
                    severity_class = alert['severity']
                    html_content += f"""
                    <tr>
                        <td>{alert['timestamp']}</td>
                        <td>{alert['type']}</td>
                        <td class="{severity_class}">{alert['severity'].upper()}</td>
                        <td>{alert['message']}</td>
                    </tr>
                    """

                html_content += "</table>"

            html_content += """
            </body>
            </html>
            """

            with open(output_path, 'w') as f:
                f.write(html_content)

        elif format == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'alerts': self.alerts,
                    'performance_history': self.performance_tracker.get_history().to_dict(orient='records')
                }, f, indent=2)

        print(f"Monitoring report generated: {output_path}")
