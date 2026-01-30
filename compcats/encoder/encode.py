from typing import List, Optional, Union
from scipy import stats as scipy_stats
import numpy as np
import sklearn.linear_model
from sklearn.model_selection import KFold
import torch

from ..constants import DEFAULT_SEED
from ..utils import responsecomparison_utils as rcutils

DEFAULT_ALPHA_SEARCH_VALUES = [0.001, 0.01, 0.1, 1, 10, 100]

# Function to fit and evaluate an encoding model
def fit_eval_mapper(
    mapper: sklearn.base.BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    metric: str = "population_pearsonr"
):
    """
    Fit an encoding model to training data and evaluate it on both training and test data.
    Args:
        mapper(sklearn.base.BaseEstimator): Mapping model (e.g., Ridge regression)
        x_train(np.ndarray): Training data (num_samples x num_input_features)
        y_train(np.ndarray): Training responses (num_samples x num_output_features)
        x_test(np.ndarray): Test data (num_samples x num_input_features)
        y_test(np.ndarray): Test responses (num_samples x num_output_features)
        metric(str): Metric to evaluate the model. 
            Options are 'voxelwise_pearsonr', 'population_pearsonr',
            'population_rdmr', or 'all'.
    Returns:
        encoder(sklearn.base.BaseEstimator): Trained encoding model
        train_metrics(dict/float): Training metrics
        test_metrics(dict/float): Test metrics
    """
    encoder = fit_mapper(mapper, x_train, y_train)
    # Evaluate the model on training and test data
    train_metrics = eval_encoder(encoder, x_train, y_train, metric=metric)
    test_metrics = eval_encoder(encoder, x_test, y_test, metric=metric)
    return encoder, train_metrics, test_metrics

# Function to fit an encoding model
def fit_mapper(
    mapper: sklearn.base.BaseEstimator,
    x_train: np.ndarray,
    y_train: np.ndarray,
):
    """
    Fit an encoding model to training data.
    Args:
        mapper(sklearn.base.BaseEstimator): Mapping model (e.g., Ridge regression)
        x_train(np.ndarray): Training data (num_samples x num_input_features)
        y_train(np.ndarray): Training responses (num_samples x num_output_features)
    Returns:
        encoder(sklearn.base.BaseEstimator): Trained encoding model
    """
    encoder = mapper.fit(x_train, y_train)
    return encoder

# Get performance of a trained model on a dataset
def eval_encoder(encoder, x_test, y_test, metric="population_pearsonr"):
    """
    Evaluate a trained encoding model on a test dataset.
    """
    # Check if the encoder is fitted
    if not hasattr(encoder, "predict"):
        raise ValueError("Encoder is not fitted. Please fit the encoder first.")
    # Error handling for the metrics
    allowed_metrics = [
        'voxelwise_pearsonr',
        'population_pearsonr',
        'population_rdmr',
        'population_all',
        'all'
    ] # TODO: Implement: population all
    assert metric in allowed_metrics, \
        f"Invalid metric specified. Allowed={allowed_metrics}"
    if isinstance(encoder, str):
        raise ValueError("Encoder should be a trained model, not a string.")
    if isinstance(metric, str):
        metric = [metric.lower()]
    if "all" in metric:
        metric = [
            'voxelwise_pearsonr',
            'population_pearsonr',
            'population_rdmr',
        ]
    # Predict the test data
    y_pred_test = encoder.predict(x_test)
    # Initialize the dictionary to store the results
    test_metrics = {}
    # Calculate the metrics
    if "population_pearsonr" in metric:
        test_metrics['population_pearsonr'], _ = scipy_stats.pearsonr(y_test, y_pred_test)
    if "population_rdmr" in metric:
        test_metrics['population_rdmr'] = rcutils.rdmr(
            torch.tensor(y_test), torch.tensor(y_pred_test)
        )
    if "voxelwise_pearsonr" in metric:
        test_metrics['voxelwise_pearsonr'] = []
        n_vox = y_test.shape[1]
        r = np.full(n_vox, np.nan)
        for v in range(n_vox):
            yt, yp = y_test[:, v], y_pred_test[:, v]
            # Skip dead / constant voxels
            if np.std(yt) < 1e-8 or np.std(yp) < 1e-8:
                continue
            r[v], _ = scipy_stats.pearsonr(yt, yp)
    # Error handling for the metrics
    if len(metric) == 1:
        test_metrics = test_metrics[metric[0]]
    # Return the metrics
    return test_metrics

# Function to get model details for a particular alpha using k-fold cross-validation
# mapper, alpha, x_trainval, y_trainval, kf
def get_alpha_model(
    mapper: sklearn.base.BaseEstimator,
    alpha: float,
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    kf: KFold
):
    """
    Get a trained encoding model for a particular alpha using k-fold cross-validation.
    Args:
        mapper(sklearn.base.BaseEstimator): Mapping model (e.g., Ridge regression)
        alpha(float): Regularization strength
        x_trainval(np.ndarray): Training + validation data (num_samples x num_input_features)
        y_trainval(np.ndarray): Training + validation responses (num_samples x num_output_features)
        kf(KFold): KFold cross-validation object
    Returns:
        mean_betas(np.ndarray): Mean beta coefficients across folds
        mean_intercept(float): Mean intercept across folds
        mean_train_corr(float): Mean training correlation across folds
        mean_val_corr(float): Mean validation correlation across folds
        logs(dict): Dictionary containing logs for each fold
            - 'coefs': Beta coefficients for each fold
            - 'intercept': Intercept for each fold
            - 'train_corr': Training correlation for each fold
            - 'val_corr': Validation correlation for each fold
            - 'train_indices': Training indices for each fold
            - 'val_indices': Validation indices for each fold
            - 'num_folds': Number of folds
    """
    encoding_model = sklearn.base.clone(mapper)
    logs = {}
    betas = None
    intercepts = []
    train_corrs, val_corrs = [], []
    train_indices, val_indices = [], []
    for _, (train_index, val_index) in enumerate(kf.split(x_trainval)):
        train_indices.append(train_index)
        val_indices.append(val_index)
        encoding_model.alpha = alpha
        x_train, y_train = x_trainval[train_index], y_trainval[train_index]
        x_val, y_val = x_trainval[val_index], y_trainval[val_index]
        encoding_model, train_corr, val_corr = fit_eval_mapper(
            encoding_model, x_train, y_train, x_val, y_val
        )
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

# Function to get optimal alpha using k-fold cross-validation
def get_optimal_alpha(
    mapper: sklearn.base.BaseEstimator,
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    kf: KFold,
    alpha_values: List[float]
):
    """
    Get optimal alpha value by searching for best alpha using k-fold cross-validation.
    Args:
        model(sklearn.base.BaseEstimator): Mapping model (e.g., Ridge regression)
        x_trainval(np.ndarray): Training + validation data (num_samples x num_input_features)
        y_trainval(np.ndarray): Training + validation responses (num_samples x num_output_features)
        kf(KFold): KFold cross-validation object
        alpha_values(list of float): List of alpha values to search
    Returns:
        optimal_alpha(float): Optimal alpha value
        best_model_details(dict): Dictionary containing details of the best model:
            - 'coef': Beta coefficients of the model
            - 'intercept': Intercept of the model
            - 'train_corr': Training correlation
            - 'val_corr': Validation correlation
        search_logs(dict): Dictionary containing logs for each alpha value
    """
    # Clone mapping model
    mapping = sklearn.base.clone(mapper)
    # init
    alpha_val_corr = []
    alpha_models = {}
    search_logs = {}
    # Iterate over alpha values
    for _, alpha in enumerate(alpha_values):
        # Get params & performance for a particular alpha
        beta_, intercept_, train_corr_, val_corr_, search_logs[alpha] = get_alpha_model(
            mapping, alpha, x_trainval, y_trainval, kf
        )
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
def get_optimal_model(
    mapper: sklearn.base.BaseEstimator,
    x_trainval: np.ndarray,
    y_trainval: np.ndarray,
    seed: Optional[int] = DEFAULT_SEED,
    alpha: Optional[Union[str, float, List[float]]]= 'search',
    unit_subsets: Optional[List[int]] = None,
):
    """
    Get optimal encoding model by searching for best alpha using k-fold cross-validation, or 
    using a provided alpha value.
    Args:
        mapper(sklearn.base.BaseEstimator): Mapping model (e.g., Ridge regression)
        x_trainval(np.ndarray): Training + validation data (num_samples x num_input_features)
        y_trainval(np.ndarray): Training + validation responses (num_samples x num_output_features)
        seed(int, optional): Random seed for reproducibility. Default is DEFAULT_SEED.
        alpha(str/float/list, optional): 'search' to search for optimal alpha
            using cross-validation, or a float value or list of float values to use as alpha(s).
            Default is 'search'.
        unit_subsets(list of int, optional): List of unit indices to consider.
            If None, all units are considered. Default is None.
    Returns:
        em_details(dict): Dictionary containing details of the encoding model:
            - 'coef': Beta coefficients of the model
            - 'intercept': Intercept of the model
            - 'train_corr': Training correlation
            - 'val_corr': Validation correlation
            - 'alpha': Optimal alpha value used
            - 'alpha_search_logs': Logs for each alpha value if search was performed
    """
    # Error handling
    assert x_trainval.shape[0] == y_trainval.shape[0], \
        "Number of samples in x_trainval and y_trainval must be the same."
    assert x_trainval.ndim == 2, \
        "x_trainval must be a 2D array (num_samples x num_input_features)."
    assert y_trainval.ndim == 2, \
        "y_trainval must be a 2D array (num_samples x num_output_features)."
    # Determine if we need to search for alpha
    search_for_alpha = False
    if isinstance(alpha, str):
        if alpha.lower() == 'search':
            search_for_alpha = True
            alpha = DEFAULT_ALPHA_SEARCH_VALUES
        else:
            raise ValueError('alpha should be "search" or a float value or list of float values')
    elif isinstance(alpha, list):
        if len(alpha) > 1:
            search_for_alpha = True
        elif len(alpha) == 1:
            alpha = alpha[0]
        else:
            raise ValueError('alpha list is empty')
    elif isinstance(alpha, (float, int)):
        pass
    else:
        raise ValueError('alpha should be "search" or a float value or list of float values')
    # If linear regression, no regularization
    if isinstance(mapper, sklearn.linear_model.LinearRegression):
        alpha = 0
    # Clone the mapping model
    mapping = sklearn.base.clone(mapper)
    # If unit subsets are provided, select only those units
    if unit_subsets:
        x_trainval = x_trainval[:, unit_subsets]
    # Train val split indices
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    # Find alpha
    if search_for_alpha:
        assert isinstance(alpha, list), 'alpha should be a list of float values for searching'
        optimal_alpha, em_details, search_logs = get_optimal_alpha(
            mapping, x_trainval, y_trainval, kf, alpha_values=alpha
        )
        em_details['alpha_search_logs'] = search_logs
    else:
        assert isinstance(alpha, float), 'alpha should be a float value'
        # Get the model if alpha is unknown or irrelevant
        optimal_alpha = alpha
        beta_, intercept_, train_corr_, val_corr_, logs = get_alpha_model(
            mapping, optimal_alpha, x_trainval, y_trainval, kf
        )
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
