"""Bounded selected-model explanations and attribution stability utilities."""
from .shap_explainer import explain_prediction, save_explanation, sign_agreement, top_k_agreement

__all__ = ['explain_prediction', 'save_explanation', 'sign_agreement', 'top_k_agreement']
