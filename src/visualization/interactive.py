"""
Interactive Visualization Dashboard

Create interactive plots and dashboards using Plotly.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class InteractiveDashboard:
    """
    Interactive visualization dashboard using Plotly.

    Creates interactive plots for model analysis and comparison.
    """

    def __init__(self):
        """Initialize interactive dashboard."""
        self.figures: List[go.Figure] = []

    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             metrics: List[str],
                             title: str = "Model Comparison") -> go.Figure:
        """
        Create interactive model comparison chart.

        Args:
            comparison_df: DataFrame with model metrics
            metrics: Metrics to plot
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        for metric in metrics:
            if metric in comparison_df.columns:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df.index,
                    y=comparison_df[metric],
                    text=comparison_df[metric].round(3),
                    textposition='auto'
                ))

        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            hovermode='x unified',
            template='plotly_white'
        )

        self.figures.append(fig)
        return fig

    def plot_learning_curves(self, train_scores: List[float],
                            val_scores: List[float],
                            title: str = "Learning Curves") -> go.Figure:
        """
        Plot interactive learning curves.

        Args:
            train_scores: Training scores
            val_scores: Validation scores
            title: Plot title

        Returns:
            Plotly figure
        """
        epochs = list(range(1, len(train_scores) + 1))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=epochs, y=train_scores,
            mode='lines+markers',
            name='Training',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=epochs, y=val_scores,
            mode='lines+markers',
            name='Validation',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Epoch",
            yaxis_title="Score",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def plot_confusion_matrix_interactive(self, cm: np.ndarray,
                                         labels: Optional[List[str]] = None,
                                         title: str = "Confusion Matrix") -> go.Figure:
        """
        Create interactive confusion matrix.

        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title

        Returns:
            Plotly figure
        """
        if labels is None:
            labels = [f"Class {i}" for i in range(cm.shape[0])]

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont=dict(size=12),
            hovertemplate='True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            template='plotly_white'
        )

        return fig

    def plot_feature_importance_interactive(self, feature_names: List[str],
                                           importance_values: np.ndarray,
                                           top_n: int = 20,
                                           title: str = "Feature Importance") -> go.Figure:
        """
        Interactive feature importance plot.

        Args:
            feature_names: Feature names
            importance_values: Importance values
            top_n: Number of top features to show
            title: Plot title

        Returns:
            Plotly figure
        """
        # Sort and select top features
        indices = np.argsort(importance_values)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importances = importance_values[indices]

        fig = go.Figure(go.Bar(
            x=top_importances,
            y=top_features,
            orientation='h',
            marker=dict(
                color=top_importances,
                colorscale='Viridis',
                showscale=True
            ),
            text=top_importances.round(3),
            textposition='auto'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, top_n * 25),
            template='plotly_white'
        )

        return fig

    def plot_roc_curves_interactive(self, roc_data: Dict[str, tuple],
                                   title: str = "ROC Curves") -> go.Figure:
        """
        Plot interactive ROC curves for multiple models.

        Args:
            roc_data: Dict mapping model names to (fpr, tpr, auc) tuples
            title: Plot title

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray')
        ))

        # Add ROC curves
        for model_name, (fpr, tpr, auc) in roc_data.items():
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC={auc:.3f})',
                line=dict(width=2)
            ))

        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def create_3d_scatter(self, X: np.ndarray, y: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         title: str = "3D Feature Space") -> go.Figure:
        """
        Create 3D scatter plot of features.

        Args:
            X: Feature matrix (uses first 3 features or PCA)
            y: Labels
            feature_names: Feature names
            title: Plot title

        Returns:
            Plotly figure
        """
        if X.shape[1] > 3:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            X_3d = pca.fit_transform(X)
            feature_names = ['PC1', 'PC2', 'PC3']
        else:
            X_3d = X[:, :3]
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(3)]

        fig = go.Figure(data=[go.Scatter3d(
            x=X_3d[:, 0],
            y=X_3d[:, 1],
            z=X_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=y,
                colorscale='Viridis',
                showscale=True,
                opacity=0.8
            ),
            text=[f'Label: {label}' for label in y],
            hovertemplate='%{text}<extra></extra>'
        )])

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=feature_names[0],
                yaxis_title=feature_names[1],
                zaxis_title=feature_names[2]
            ),
            template='plotly_white'
        )

        return fig

    def save_dashboard(self, filepath: str, format: str = 'html'):
        """
        Save dashboard to file.

        Args:
            filepath: Output file path
            format: Output format ('html' or 'png')
        """
        if format == 'html' and self.figures:
            # Combine all figures into HTML
            with open(filepath, 'w') as f:
                f.write('<html><head><title>ML Dashboard</title></head><body>')
                for i, fig in enumerate(self.figures):
                    f.write(f'<h2>Figure {i+1}</h2>')
                    f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
                f.write('</body></html>')
