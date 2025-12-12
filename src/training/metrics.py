"""
Medical Metrics Calculator
Comprehensive metrics for medical imaging classification
Specialized for chest X-ray pneumonia detection
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class MedicalMetrics:
    """
    Comprehensive medical metrics calculator
    Focuses on metrics important for medical diagnosis
    """
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ['NORMAL', 'PNEUMONIA']
        self.num_classes = len(self.class_names)
    
    def calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive medical metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (for AUC, etc.)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name.lower()}'] = precision_per_class[i]
            metrics[f'recall_{class_name.lower()}'] = recall_per_class[i]
            metrics[f'f1_{class_name.lower()}'] = f1_per_class[i]
        
        # Medical-specific metrics (assuming binary classification)
        if self.num_classes == 2:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Sensitivity (True Positive Rate) - ability to correctly identify pneumonia
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # Specificity (True Negative Rate) - ability to correctly identify normal
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # Positive Predictive Value (Precision for positive class)
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Negative Predictive Value
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            
            # False Positive Rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # False Negative Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            # Diagnostic metrics
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
        
        # AUC metrics (if probabilities provided)
        if y_prob is not None:
            try:
                if self.num_classes == 2:
                    # Binary classification AUC
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc_pr'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    # Multi-class AUC
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
                    metrics['auc_pr'] = average_precision_score(y_true, y_prob, average='weighted')
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        else:
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        return metrics
    
    def calculate_confidence_metrics(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculate confidence-based metrics
        Important for medical applications where confidence matters
        """
        
        if y_prob.ndim == 1:
            confidences = y_prob
            y_pred = (y_prob > confidence_threshold).astype(int)
        else:
            confidences = np.max(y_prob, axis=1)
            y_pred = np.argmax(y_prob, axis=1)
        
        metrics = {}
        
        # Overall confidence statistics
        metrics['mean_confidence'] = float(np.mean(confidences))
        metrics['std_confidence'] = float(np.std(confidences))
        metrics['min_confidence'] = float(np.min(confidences))
        metrics['max_confidence'] = float(np.max(confidences))
        
        # Confidence by correctness
        correct_mask = (y_pred == y_true)
        incorrect_mask = ~correct_mask
        
        if np.any(correct_mask):
            metrics['mean_confidence_correct'] = float(np.mean(confidences[correct_mask]))
        else:
            metrics['mean_confidence_correct'] = 0.0
        
        if np.any(incorrect_mask):
            metrics['mean_confidence_incorrect'] = float(np.mean(confidences[incorrect_mask]))
        else:
            metrics['mean_confidence_incorrect'] = 0.0
        
        # High confidence predictions
        high_conf_mask = confidences > 0.8
        if np.any(high_conf_mask):
            metrics['high_confidence_accuracy'] = float(np.mean(correct_mask[high_conf_mask]))
            metrics['high_confidence_count'] = int(np.sum(high_conf_mask))
        else:
            metrics['high_confidence_accuracy'] = 0.0
            metrics['high_confidence_count'] = 0
        
        # Low confidence predictions
        low_conf_mask = confidences < 0.6
        if np.any(low_conf_mask):
            metrics['low_confidence_accuracy'] = float(np.mean(correct_mask[low_conf_mask]))
            metrics['low_confidence_count'] = int(np.sum(low_conf_mask))
        else:
            metrics['low_confidence_accuracy'] = 0.0
            metrics['low_confidence_count'] = 0
        
        return metrics
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        save_path: str = None,
        normalize: bool = True
    ) -> plt.Figure:
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm_norm = cm
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        # Add counts to normalized matrix
        if normalize:
            for i in range(len(self.class_names)):
                for j in range(len(self.class_names)):
                    ax.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                           ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        save_path: str = None
    ) -> plt.Figure:
        """Plot ROC curve for binary classification"""
        
        if self.num_classes != 2:
            raise ValueError("ROC curve plotting only supported for binary classification")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
        auc_score = roc_auc_score(y_true, y_prob[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray, 
        save_path: str = None
    ) -> plt.Figure:
        """Plot Precision-Recall curve for binary classification"""
        
        if self.num_classes != 2:
            raise ValueError("PR curve plotting only supported for binary classification")
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
        ap_score = average_precision_score(y_true, y_prob[:, 1])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot PR curve
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR Curve (AP = {ap_score:.3f})')
        
        # Plot baseline
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='navy', linestyle='--', lw=2,
                   label=f'Baseline (AP = {baseline:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PR curve saved to {save_path}")
        
        return fig
    
    def generate_medical_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate a medical-focused evaluation report
        """
        
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        report = []
        report.append("MEDICAL MODEL EVALUATION REPORT")
        report.append("=" * 50)
        report.append("")
        
        # Overall Performance
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy']*100:.1f}%)")
        report.append(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        report.append("")
        
        # Medical Metrics (for binary classification)
        if self.num_classes == 2:
            report.append("DIAGNOSTIC PERFORMANCE:")
            report.append(f"  Sensitivity (Recall): {metrics['sensitivity']:.3f} - Ability to detect pneumonia")
            report.append(f"  Specificity: {metrics['specificity']:.3f} - Ability to identify normal cases")
            report.append(f"  Positive Predictive Value: {metrics['ppv']:.3f} - Accuracy of pneumonia predictions")
            report.append(f"  Negative Predictive Value: {metrics['npv']:.3f} - Accuracy of normal predictions")
            report.append("")
            
            report.append("ERROR ANALYSIS:")
            report.append(f"  False Positives: {metrics['false_positives']} (Normal cases predicted as Pneumonia)")
            report.append(f"  False Negatives: {metrics['false_negatives']} (Pneumonia cases predicted as Normal)")
            report.append(f"  False Positive Rate: {metrics['fpr']:.3f}")
            report.append(f"  False Negative Rate: {metrics['fnr']:.3f}")
            report.append("")
        
        # Per-class performance
        report.append("PER-CLASS PERFORMANCE:")
        for class_name in self.class_names:
            class_lower = class_name.lower()
            report.append(f"  {class_name}:")
            report.append(f"    Precision: {metrics[f'precision_{class_lower}']:.3f}")
            report.append(f"    Recall: {metrics[f'recall_{class_lower}']:.3f}")
            report.append(f"    F1-Score: {metrics[f'f1_{class_lower}']:.3f}")
        report.append("")
        
        # Confidence metrics (if available)
        if y_prob is not None:
            conf_metrics = self.calculate_confidence_metrics(y_true, y_prob)
            report.append("CONFIDENCE ANALYSIS:")
            report.append(f"  Mean Confidence: {conf_metrics['mean_confidence']:.3f}")
            report.append(f"  High Confidence Predictions (>0.8): {conf_metrics['high_confidence_count']}")
            report.append(f"  High Confidence Accuracy: {conf_metrics['high_confidence_accuracy']:.3f}")
            report.append(f"  Low Confidence Predictions (<0.6): {conf_metrics['low_confidence_count']}")
            report.append(f"  Low Confidence Accuracy: {conf_metrics['low_confidence_accuracy']:.3f}")
            report.append("")
        
        # Clinical interpretation
        report.append("CLINICAL INTERPRETATION:")
        if self.num_classes == 2 and metrics['sensitivity'] >= 0.85:
            report.append("  ✓ Good sensitivity - Model effectively detects most pneumonia cases")
        elif self.num_classes == 2 and metrics['sensitivity'] < 0.75:
            report.append("  ⚠ Low sensitivity - Model may miss pneumonia cases (high clinical risk)")
        
        if self.num_classes == 2 and metrics['specificity'] >= 0.85:
            report.append("  ✓ Good specificity - Model avoids false alarms")
        elif self.num_classes == 2 and metrics['specificity'] < 0.75:
            report.append("  ⚠ Low specificity - Model may over-diagnose pneumonia")
        
        if metrics['accuracy'] >= 0.90:
            report.append("  ✓ Excellent overall accuracy for medical imaging")
        elif metrics['accuracy'] >= 0.85:
            report.append("  ✓ Good overall accuracy for medical imaging")
        else:
            report.append("  ⚠ Accuracy below recommended threshold for medical applications")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test medical metrics
    print("Testing Medical Metrics...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate realistic medical data
    y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% pneumonia
    
    # Simulate model predictions with some realistic performance
    y_prob = np.random.rand(n_samples, 2)
    # Make predictions somewhat correlated with true labels
    for i in range(n_samples):
        if y_true[i] == 1:  # Pneumonia
            y_prob[i, 1] += 0.3  # Boost pneumonia probability
        else:  # Normal
            y_prob[i, 0] += 0.3  # Boost normal probability
    
    # Normalize probabilities
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Test metrics calculation
    metrics_calc = MedicalMetrics(['NORMAL', 'PNEUMONIA'])
    
    # Calculate metrics
    metrics = metrics_calc.calculate_metrics(y_true, y_pred, y_prob)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Sensitivity: {metrics['sensitivity']:.3f}")
    print(f"Specificity: {metrics['specificity']:.3f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    
    # Test confidence metrics
    conf_metrics = metrics_calc.calculate_confidence_metrics(y_true, y_prob)
    print(f"Mean Confidence: {conf_metrics['mean_confidence']:.3f}")
    
    # Test medical report
    report = metrics_calc.generate_medical_report(y_true, y_pred, y_prob)
    print("\nMedical Report Preview:")
    print(report[:500] + "...")
    
    print("\n✅ Medical metrics working correctly!")