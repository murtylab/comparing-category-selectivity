from scipy import stats as scipy_stats
import numpy as np
import torch

from ..utils import responsecomparison_utils as rcutils

# Function to fit and evaluate an encoding model
def fit_eval_encoder(mapper, x_train, y_train, x_test, y_test, metric="pearsonr"):
    """
    Fit an encoding model to training data and evaluate it on both training and test data.
    """
    encoder = fit_encoder(mapper, x_train, y_train)
    # Evaluate the model on training and test data
    train_metrics = eval_encoder(encoder, x_train, y_train, metric=metric)
    test_metrics = eval_encoder(encoder, x_test, y_test, metric=metric)
    return encoder, train_metrics, test_metrics

# Function to fit an encoding model
def fit_encoder(mapper, x_train, y_train):
    """
    Fit an encoding model to training data.
    """
    encoder = mapper.fit(x_train, y_train)
    return encoder

# Get performance of a trained model on a dataset
def eval_encoder(encoder, x_test, y_test, metric="pearsonr"):
    """
    Evaluate a trained encoding model on a test dataset.
    """
    # Check if the encoder is fitted
    if not hasattr(encoder, "predict"):
        raise ValueError("Encoder is not fitted. Please fit the encoder first.")
    # Error handling for the metrics
    assert metric in ["pearsonr", "rdmr", "all"], \
        "Invalid metric specified. Use 'pearsonr' or 'rdmr' or 'all'."
    if isinstance(encoder, str):
        raise ValueError("Encoder should be a trained model, not a string.")
    if isinstance(metric, str):
        metric = [metric.lower()]
    if "all" in metric:
        metric = ["pearsonr", "rdmr"]
    # Predict the test data
    y_pred_test = encoder.predict(x_test)
    # Initialize the dictionary to store the results
    test_metrics = {}
    # Calculate the metrics
    if "pearsonr" in metric:
        test_metrics['pearsonr'], _ = scipy_stats.pearsonr(y_test, y_pred_test)
    if "rdmr" in metric:
        test_metrics['rdmr'] = rcutils.rdmr(
            torch.tensor(y_test), torch.tensor(y_pred_test)
        )
    # Error handling for the metrics
    if len(metric) == 1:
        test_metrics = test_metrics[metric[0]]
    # Return the metrics
    return test_metrics
