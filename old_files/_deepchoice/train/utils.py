"""Utility functions for training."""
import torch


def diff_in_norm(model1: dict, model2: dict) -> float:
    """Calculates the change of norm of parameters from one model to another,
    In particular, let theta1 and theta2 be parameters of models,
    this method compute ||theta1 - theta2||.
    This method can be used to assess the convergence state during training.
    """
    total_norm_diff = 0.0
    for key in model1.keys():
        theta1, theta2 = model1[key], model2[key]
        total_norm_diff += (theta1 - theta2).data.norm(2)
    total_norm_diff = total_norm_diff ** 0.5
    return total_norm_diff
