"""
Knowledge Distillation

Transfer knowledge from a large teacher model to a smaller student model.
"""

import numpy as np
from typing import Any, Optional
from sklearn.base import clone, BaseEstimator


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Trains a smaller student model to mimic a larger teacher model's
    predictions, often achieving similar performance with fewer parameters.
    """

    def __init__(self, teacher_model: Any, student_model: Any,
                 temperature: float = 3.0,
                 alpha: float = 0.5):
        """
        Initialize knowledge distillation.

        Args:
            teacher_model: Large, accurate teacher model
            student_model: Smaller student model to train
            temperature: Temperature for softening probabilities
            alpha: Weight for distillation loss vs hard label loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha

    def distill(self, X: np.ndarray, y: np.ndarray,
               X_val: Optional[np.ndarray] = None,
               y_val: Optional[np.ndarray] = None) -> BaseEstimator:
        """
        Perform knowledge distillation.

        Args:
            X: Training features
            y: Training labels (hard labels)
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Trained student model
        """
        # Get teacher predictions (soft labels)
        if hasattr(self.teacher_model, 'predict_proba'):
            teacher_probs = self.teacher_model.predict_proba(X)
            # Apply temperature
            teacher_soft_labels = self._apply_temperature(teacher_probs)
        else:
            # If teacher doesn't have predict_proba, use hard labels
            teacher_soft_labels = self.teacher_model.predict(X)
            self.alpha = 0.0  # Only use hard labels

        # Train student model
        # For sklearn models, we'll use a weighted approach
        # In practice, you'd implement custom loss function

        # Option 1: Train on soft labels only (simplified)
        if self.alpha == 1.0:
            # Pure distillation
            if hasattr(self.student_model, 'predict_proba'):
                # For probabilistic models, we can't directly train on probabilities
                # So we'll train on hard labels from teacher
                teacher_hard_labels = np.argmax(teacher_soft_labels, axis=1)
                self.student_model.fit(X, teacher_hard_labels)
            else:
                self.student_model.fit(X, teacher_soft_labels)

        # Option 2: Train on hard labels only
        elif self.alpha == 0.0:
            self.student_model.fit(X, y)

        # Option 3: Mixed training (approximate)
        else:
            # First train on hard labels
            self.student_model.fit(X, y)

            # Then fine-tune on soft labels
            if hasattr(self.student_model, 'partial_fit'):
                teacher_hard_labels = np.argmax(teacher_soft_labels, axis=1)
                self.student_model.partial_fit(X, teacher_hard_labels)

        return self.student_model

    def _apply_temperature(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to soften probability distribution.

        Args:
            probabilities: Class probabilities

        Returns:
            Softened probabilities
        """
        # Apply temperature: p_i = exp(logit_i / T) / sum(exp(logit_j / T))
        # For sklearn probs, convert back to logits, apply temperature, then softmax

        # Avoid log(0)
        probabilities = np.clip(probabilities, 1e-10, 1 - 1e-10)

        # Convert to logits
        logits = np.log(probabilities)

        # Apply temperature
        scaled_logits = logits / self.temperature

        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        soft_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return soft_probs

    def evaluate_compression(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate compression effectiveness.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        # Teacher performance
        teacher_pred = self.teacher_model.predict(X_test)
        teacher_acc = np.mean(teacher_pred == y_test)

        # Student performance
        student_pred = self.student_model.predict(X_test)
        student_acc = np.mean(student_pred == y_test)

        # Agreement between teacher and student
        agreement = np.mean(teacher_pred == student_pred)

        # Model sizes (approximate for sklearn models)
        teacher_size = self._estimate_model_size(self.teacher_model)
        student_size = self._estimate_model_size(self.student_model)

        compression_ratio = teacher_size / student_size if student_size > 0 else 1.0

        return {
            'teacher_accuracy': teacher_acc,
            'student_accuracy': student_acc,
            'accuracy_retention': student_acc / teacher_acc if teacher_acc > 0 else 0.0,
            'teacher_student_agreement': agreement,
            'teacher_size_bytes': teacher_size,
            'student_size_bytes': student_size,
            'compression_ratio': compression_ratio
        }

    def _estimate_model_size(self, model: Any) -> int:
        """
        Estimate model size in bytes.

        Args:
            model: Model to estimate

        Returns:
            Estimated size in bytes
        """
        size = 0

        # Count parameters
        if hasattr(model, 'coef_'):
            size += model.coef_.nbytes

        if hasattr(model, 'intercept_'):
            size += model.intercept_.nbytes

        # For ensemble models
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                size += self._estimate_model_size(estimator)

        return size


class ProgressiveDistillation:
    """
    Progressive distillation using intermediate teacher models.

    Gradually reduces model size through multiple distillation steps.
    """

    def __init__(self, teacher_model: Any,
                 intermediate_sizes: list,
                 final_student: Any,
                 temperature: float = 3.0):
        """
        Initialize progressive distillation.

        Args:
            teacher_model: Original large teacher
            intermediate_sizes: List of intermediate model sizes
            final_student: Final small student model
            temperature: Distillation temperature
        """
        self.teacher_model = teacher_model
        self.intermediate_sizes = intermediate_sizes
        self.final_student = final_student
        self.temperature = temperature
        self.intermediate_models = []

    def distill_progressively(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Perform progressive distillation.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Final distilled student model
        """
        current_teacher = self.teacher_model

        # Distill to intermediate models
        for i, size in enumerate(self.intermediate_sizes):
            print(f"Distilling to intermediate model {i+1} (size: {size})")

            # Create intermediate student
            # (In practice, you'd create a model of specified size)
            intermediate_student = clone(current_teacher)

            # Distill
            distiller = KnowledgeDistillation(
                current_teacher,
                intermediate_student,
                temperature=self.temperature
            )
            intermediate_student = distiller.distill(X, y)

            self.intermediate_models.append(intermediate_student)
            current_teacher = intermediate_student

        # Final distillation to target student
        print("Distilling to final student model")
        final_distiller = KnowledgeDistillation(
            current_teacher,
            self.final_student,
            temperature=self.temperature
        )
        final_student = final_distiller.distill(X, y)

        return final_student


class SelfDistillation:
    """
    Self-distillation where model learns from its own predictions.

    Useful for regularization and improving generalization.
    """

    def __init__(self, model: Any, temperature: float = 3.0, n_iterations: int = 3):
        """
        Initialize self-distillation.

        Args:
            model: Model to self-distill
            temperature: Temperature for soft labels
            n_iterations: Number of self-distillation iterations
        """
        self.model = model
        self.temperature = temperature
        self.n_iterations = n_iterations

    def self_distill(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Perform self-distillation.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Self-distilled model
        """
        # Initial training
        self.model.fit(X, y)

        # Self-distillation iterations
        for iteration in range(self.n_iterations):
            print(f"Self-distillation iteration {iteration + 1}/{self.n_iterations}")

            # Use model's own predictions as soft labels
            if hasattr(self.model, 'predict_proba'):
                soft_labels = self.model.predict_proba(X)

                # Convert to hard labels (argmax)
                pseudo_labels = np.argmax(soft_labels, axis=1)

                # Retrain on pseudo labels
                if hasattr(self.model, 'partial_fit'):
                    self.model.partial_fit(X, pseudo_labels)
                else:
                    # Clone and retrain
                    new_model = clone(self.model)
                    new_model.fit(X, pseudo_labels)
                    self.model = new_model

        return self.model
