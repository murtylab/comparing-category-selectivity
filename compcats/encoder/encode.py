from scipy import stats as scipy_stats
import numpy as np
import sklearn
import sklearn.linear_model
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

def get_alpha_model(model, alpha, X_trainval, Y_trainval, kf):
    encoding_model = sklearn.base.clone(model)
    logs = {}
    betas = None
    intercepts = []
    train_corrs, val_corrs = [], []
    train_indices, val_indices = [], []
    for split_idx, (train_index, val_index) in enumerate(kf.split(X_trainval)):
        train_indices.append(train_index)
        val_indices.append(val_index)
        encoding_model.alpha = alpha
        X_train, Y_train = X_trainval[train_index], Y_trainval[train_index]
        X_val, Y_val = X_trainval[val_index], Y_trainval[val_index]
        encoding_model, train_corr, val_corr = fit_eval_encoder(encoding_model, X_train, Y_train, X_val, Y_val)
        train_corrs.append(train_corr)
        val_corrs.append(val_corr)
        if betas is None:
            betas = encoding_model.coef_
        else:
            betas = np.vstack((betas, encoding_model.coef_))
        intercepts.append(encoding_model.intercept_)
    logs['coefs'] = betas
    logs['intercept'] = intercepts
    logs['train_corr'] = train_corrs
    logs['val_corr'] = val_corrs
    logs['train_indices'] = train_indices
    logs['val_indices'] = val_indices
    logs['num_folds'] = kf.n_splits
    mean_betas = np.mean(betas, axis=0)
    mean_intercept = np.mean(intercepts)
    return mean_betas, mean_intercept, np.mean(train_corrs), np.mean(val_corrs), logs


def get_optimal_alpha(model, X_trainval, Y_trainval, kf):
    # Clone mapping model
    mapping = sklearn.base.clone(model)
    # Alpha values to search for
    alpha_values = [0.001, 0.01, 0.1, 1, 10, 100]
    # init
    alpha_val_corr = []
    alpha_models = {}
    search_logs = {}
    # Iterate over alpha values
    for idx, alpha in enumerate(alpha_values):
        # Get params & performance for a particular alpha
        beta_, intercept_, train_corr_, val_corr_, search_logs[alpha] = get_alpha_model(mapping, alpha, X_trainval, Y_trainval, kf)
        # Store
        alpha_models[alpha] = {
            'coef': beta_,
            'intercept': intercept_,
            'train_corr': train_corr_,
            'val_corr': val_corr_
        }
        alpha_val_corr.append(val_corr_)
    # Get alpha with best validation correlation
    best_corr_idx = np.nanargmax(alpha_val_corr) # Ignore NaNs
    optimal_alpha = alpha_values[best_corr_idx]
    return optimal_alpha, alpha_models[optimal_alpha], search_logs


# Function to calculate subject-wise beta coefficients on training data
def get_optimal_model(model, X, Y, seed, alpha='search', unit_subsets=None):
    """
    Get beta coefficients for a subject
    alpha = 'search' or alpha value
    """
    # Clone the mapping model
    mapping = sklearn.base.clone(model)
    # Error handling
    if isinstance(mapping, sklearn.linear_model.LinearRegression):
        alpha = 0
    # Training Validation data
    X_trainval, Y_trainval = X, Y
    if unit_subsets:
        X_trainval = X_trainval[:, unit_subsets]
    # Train val split indices
    kf = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    # Find alpha
    if alpha == 'search':
        optimal_alpha, em_details, search_logs = get_optimal_alpha(mapping, X_trainval, Y_trainval, kf)
        em_details['alpha_search_logs'] = search_logs
    elif alpha is None:
        raise ValueError('alpha is not provided!')
    else:
        # Get the model if alpha is unknown or irrelevant
        optimal_alpha = alpha
        beta_, intercept_, train_corr_, val_corr_, logs = get_alpha_model(mapping, optimal_alpha, X_trainval, Y_trainval, kf)
        em_details = {
            'coef': beta_,
            'intercept': intercept_,
            'train_corr': train_corr_,
            'val_corr': val_corr_
        }
        em_details['alpha_search_logs'] = {f'{optimal_alpha}': logs}
    # Store optimal alpha
    em_details['alpha'] = optimal_alpha
    return em_details # Keys: coef, intercept, train_corr, val_corr, alpha, alpha_search_logs
