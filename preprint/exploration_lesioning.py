import os
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.markers as mmarkers
import scipy
from sklearn.linear_model import Ridge, Lasso

from compcats.localizer import localizer
from compcats.datasets.fmri import nsd
from compcats.utils import model_utils, misc_utils, responsecomparison_utils as rc_utils
from compcats.models import resnet50_openaiclip, alexnet_imagenet1kv1

import main_analysis as ma
import paths

# Matplotlib settings
import matplotlib as mpl
from matplotlib import rcParams
rcParams.update({'figure.autolayout': False})
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# Font
font_path = '/home/alishd/.fonts/Avenir.ttc'
avenir_font = fm.FontProperties(fname=font_path)
mpl.rcParams['font.family'] = avenir_font.get_name()
ttc = fm.get_font(font_path)
print("Font name from TTC:", ttc.family_name)
print("Style:", ttc.style_name)

RESULTS_DIR = 'results'
MODELS_DIR = 'models'

NSD_DIR = paths.NSD_DIR
GRILLSPECTOR_LOCALIZER_DIR = paths.GRILLSPECTOR_LOCALIZER_DIR

splithalf_csv = pd.read_csv(f'{ma.SUBJECTSUBJECT_CORRELATIONS_CSV_PATH}')

# Util function

def get_em_performances(
        X_train, Y_train, X_test, Y_test, mapper, alpha, froi_mask, splithalf_csv,
        n_bootstrap = 10
):
    """
    Train an encoding model and return the performance.
    """
    results = {
        # Weights
        'allunits_weights_coef': None,
        'allunits_weights_intercept': None,
        # Splithalf correlations
        'splithalf_pearsonr': None,
        'splithalf_rdmr': None,
        # Baseline
        'allunits_raw_pearsonr': None,
        'allunits_raw_rdmr': None,
        'allunits_normed_pearsonr': None,
        'allunits_normed_rdmr': None,
        # Set fROI weights to zero
        'froi0weights_raw_pearsonr': None,
        'froi0weights_raw_rdmr': None,
        'froi0weights_normed_pearsonr': None,
        'froi0weights_normed_rdmr': None,
        # Randomly zero out weights except fROI
        'random0weights_raw_pearsonrs': [],
        'random0weights_raw_rdmrs': [],
        'random0weights_normed_pearsonrs': [],
        'random0weights_normed_rdmrs': [],
        # # Shuffle fROI features
        # 'froiperm_raw_pearsonr': None,
        # 'froiperm_raw_rdmr': None,
        # 'froiperm_normed_pearsonr': None,
        # 'froiperm_normed_rdmr': None,
        # # Randomly shuffle features except fROI
        # 'randomperm_raw_pearsonrs': [],
        # 'randomperm_raw_rdmrs': [],
        # 'randomperm_normed_pearsonrs': [],
        # 'randomperm_normed_rdmrs': [],
        # # Lesion fROI features and retrain the model
        # 'froiremove_raw_pearsonr': None,
        # 'froiremove_raw_rdmr': None,
        # 'froiremove_normed_pearsonr': None,
        # 'froiremove_normed_rdmr': None,
        # # Randomly lesion features except fROI and retrain the model
        # 'randomremove_raw_pearsonrs': [],
        # 'randomremove_raw_rdmrs': [],
        # 'randomremove_normed_pearsonrs': [],
        # 'randomremove_normed_rdmrs': [],
    }
    # Initialize the mapping model based on the mapper type
    if mapper == 'ridge':
        mapping_model = Ridge(alpha=alpha)
    elif mapper == 'l1positive':
        mapping_model = Lasso(alpha=alpha, positive=True)
    else:
        raise ValueError(f"Unknown mapper: {mapper}")
    
    # Get the split-half correlations for the subjects
    # Splithalf correlations
    splithalf_info = splithalf_csv[
        (splithalf_csv['fmri_froi'] == 'FFA') &
        (splithalf_csv['fmri_tthreshold'] == 7) &
        (splithalf_csv['fmri_dataset'] == 'nsd515') &
        (splithalf_csv['subject1'] == 1)
    ].iloc[0]
    splithalf_pearsonr = splithalf_info['subject1_splithalf_pearsonr'].item()
    splithalf_rdmr = splithalf_info['subject1_splithalf_rdmr'].item()

    # Train an encoding model on all the units
    x_train = X_train.reshape(X_train.shape[0], -1)
    y_train = Y_train
    assert y_train.ndim == 2, "Y_train should be 2D, sample x voxels"
    mapping_model.fit(x_train, y_train)
    # Check the encoding model performance
    x_test = X_test.reshape(X_test.shape[0], -1)
    y_pred = mapping_model.predict(x_test)
    y_test = Y_test
    # Convert to torch tensors and move to GPU for compatibility with rc_utils
    y_pred = torch.Tensor(y_pred).to('cuda' if torch.cuda.is_available() else 'cpu')
    y_test = torch.Tensor(y_test).to('cuda' if torch.cuda.is_available() else 'cpu')
    # Calculate Pearson correlation and RDM correlation
    allunits_raw_pearsonr = rc_utils.pearsonr(y_pred.mean(axis=-1), y_test.mean(axis=-1))
    allunits_raw_rdmr = rc_utils.rdmr(y_pred, y_test)
    # Normalize with split-half correlations
    allunits_normed_pearsonr = (allunits_raw_pearsonr / np.sqrt(splithalf_pearsonr)).item()
    allunits_normed_rdmr = (allunits_raw_rdmr / np.sqrt(splithalf_rdmr)).item()
    # Store the results
    results['allunits_weights_coef'] = mapping_model.coef_
    results['allunits_weights_intercept'] = mapping_model.intercept_
    results['splithalf_pearsonr'] = splithalf_pearsonr
    results['splithalf_rdmr'] = splithalf_rdmr
    results['allunits_raw_pearsonr'] = allunits_raw_pearsonr
    results['allunits_raw_rdmr'] = allunits_raw_rdmr
    results['allunits_normed_pearsonr'] = allunits_normed_pearsonr
    results['allunits_normed_rdmr'] = allunits_normed_rdmr

    # Set fROI weights to zero
    froi_mask_flat = froi_mask.reshape(-1)
    model_copy_coef = mapping_model.coef_.copy()
    model_copy_coef[:, froi_mask_flat] = 0
    # Make a new model with the modified weights
    model_copy = None
    if mapper == 'ridge':
        model_copy = Ridge(alpha=alpha)
    elif mapper == 'l1positive':
        model_copy = Lasso(alpha=alpha, positive=True)
    model_copy.coef_ = model_copy_coef
    model_copy.intercept_ = mapping_model.intercept_
    # Predict with the modified model
    y_pred_froiwt0 = model_copy.predict(x_test)
    # Calculate Pearson correlation and RDM correlation
    y_pred_froiwt0 = torch.Tensor(y_pred_froiwt0).to('cuda' if torch.cuda.is_available() else 'cpu')
    y_test = torch.Tensor(y_test).to('cuda' if torch.cuda.is_available() else 'cpu')
    froi0weights_raw_pearsonr = rc_utils.pearsonr(y_pred_froiwt0.mean(axis=-1), y_test.mean(axis=-1))
    froi0weights_raw_rdmr = rc_utils.rdmr(y_pred_froiwt0, y_test)
    # Normalize with split-half correlations
    froi0weights_normed_pearsonr = (froi0weights_raw_pearsonr / np.sqrt(splithalf_pearsonr)).item()
    froi0weights_normed_rdmr = (froi0weights_raw_rdmr / np.sqrt(splithalf_rdmr)).item()
    # Store the results
    results['froi0weights_raw_pearsonr'] = froi0weights_raw_pearsonr
    results['froi0weights_raw_rdmr'] = froi0weights_raw_rdmr
    results['froi0weights_normed_pearsonr'] = froi0weights_normed_pearsonr
    results['froi0weights_normed_rdmr'] = froi0weights_normed_rdmr

    # Control: Randomly zero out weights except fROI
    n_random = froi_mask_flat.sum().item()
    random_indices_log = []
    for _ in range(n_bootstrap):
        # Get indices where froi_mask_flat is False
        false_indices = torch.where(froi_mask_flat == False)[0]
        # Randomly select n_random indices from the false indices
        random_indices = torch.randperm(len(false_indices))[:n_random]
        random_indices = false_indices[random_indices]
        random_indices_log.append(random_indices)
        # Create a copy of the model and set the selected indices to zero
        model_copy_coef = mapping_model.coef_.copy()
        model_copy_coef[:, random_indices] = 0
        # Make a new model with the modified weights
        model_copy = None
        if mapper == 'ridge':
            model_copy = Ridge(alpha=alpha)
        elif mapper == 'l1positive':
            model_copy = Lasso(alpha=alpha, positive=True)
        model_copy.coef_ = model_copy_coef
        model_copy.intercept_ = mapping_model.intercept_
        # Predict with the modified model
        y_pred_randomwt0 = model_copy.predict(x_test)
        # Calculate Pearson correlation and RDM correlation
        y_pred_randomwt0 = torch.Tensor(y_pred_randomwt0).to('cuda' if torch.cuda.is_available() else 'cpu')
        y_test = torch.Tensor(y_test).to('cuda' if torch.cuda.is_available() else 'cpu')
        random0weights_raw_pearsonr = rc_utils.pearsonr(y_pred_randomwt0.mean(axis=-1), y_test.mean(axis=-1))
        random0weights_raw_rdmr = rc_utils.rdmr(y_pred_randomwt0, y_test)
        # Normalize with split-half correlations
        random0weights_normed_pearsonr = (random0weights_raw_pearsonr / np.sqrt(splithalf_pearsonr)).item()
        random0weights_normed_rdmr = (random0weights_raw_rdmr / np.sqrt(splithalf_rdmr)).item()
        # Store the results
        results['random0weights_raw_pearsonrs'].append(random0weights_raw_pearsonr)
        results['random0weights_raw_rdmrs'].append(random0weights_raw_rdmr)
        results['random0weights_normed_pearsonrs'].append(random0weights_normed_pearsonr)
        results['random0weights_normed_rdmrs'].append(random0weights_normed_rdmr)
    # Log the random indices for reproducibility
    results['random0weights_indices_log'] = np.array(random_indices_log)
    return results


# Results
frois = ['FFA', 'PPA', 'EBA']
tthreshold = 7
subjects = [1, 2, 5, 7]
model_localizer = 'grillspector'
model_localizer_dir = GRILLSPECTOR_LOCALIZER_DIR
mapper = 'ridge'
mapper_alpha = 0.1

all_models = ma.get_all_models()

results_dir = f'{RESULTS_DIR}/exploration/lesioning'
os.makedirs(results_dir, exist_ok=True)
csv_path = os.path.join(results_dir, 'lesioning_results.csv')
if os.path.exists(csv_path):
    results_csv = pd.read_csv(csv_path)
    results_csv_list = results_csv.to_dict('records')
else:
    results_csv_list = []

total_iters = len(frois) * len(subjects) * len(all_models)
with tqdm(total=total_iters) as pbar:
    for froi in frois:
        X_train_all_models = {}
        X_test_all_models = {}
        for model_name, model in all_models.items():
            layer = ma.get_froi_models_layers(froi, froi)[model_name]
            # Get model full layer activations
            X_train_all_models[model_name] = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer,
                'nsd', subset=485
            )
            X_test_all_models[model_name] = model_utils.get_dataset_activations(
                MODELS_DIR, NSD_DIR, model, layer,
                'nsd', subset=515
            )
        for subject in subjects:
            # Get fROI data
            Y_train, _ = nsd.get_betas(
                NSD_DIR, subject, froi, tthreshold, 485, True
            )
            Y_test, _ = nsd.get_betas(
                NSD_DIR, subject, froi, tthreshold, 515, True
            )
            for model_name, model in all_models.items():
                layer = ma.get_froi_models_layers(froi, froi)[model_name]
                row_params = {
                    'model': model_name,
                    'layer': layer,
                    'model_froi': froi,
                    'fmri_froi': froi,
                    'model_tthreshold': tthreshold,
                    'fmri_tthreshold': tthreshold,
                    'model_localizer': model_localizer,
                    'subject': subject,
                    'mapper': mapper,
                    'mapper_alpha': mapper_alpha,
                }
                if misc_utils.check_row_present(row_params, results_csv_list):
                    pbar.update(1)
                    continue
                results_dict = row_params.copy()
                # Get model full layer activations
                # Get model X_train & X_test Convert tensors to numpy arrays and move to CPU
                X_train = X_train_all_models[model_name].cpu().numpy()
                X_test = X_test_all_models[model_name].cpu().numpy()
                # Get model's fROI units
                froi_mask = localizer.get_froi_mask(
                    model_localizer_dir, model_localizer, MODELS_DIR,
                    model, layer, froi, tthreshold
                )
                # Get encoding model performances
                model_results = get_em_performances(
                    X_train, Y_train, X_test, Y_test,
                    mapper, mapper_alpha, froi_mask, splithalf_csv
                )
                # Store results in a dataframe
                results_dict.update({
                    'allunits_raw_pearsonr': model_results['allunits_raw_pearsonr'],
                    'allunits_raw_rdmr': model_results['allunits_raw_rdmr'],
                    'allunits_normed_pearsonr': model_results['allunits_normed_pearsonr'],
                    'allunits_normed_rdmr': model_results['allunits_normed_rdmr'],
                    'froi0weights_raw_pearsonr': model_results['froi0weights_raw_pearsonr'],
                    'froi0weights_raw_rdmr': model_results['froi0weights_raw_rdmr'],
                    'froi0weights_normed_pearsonr': model_results['froi0weights_normed_pearsonr'],
                    'froi0weights_normed_rdmr': model_results['froi0weights_normed_rdmr'],
                    'random0weights_raw_pearsonrs': model_results['random0weights_raw_pearsonrs'],
                    'random0weights_raw_rdmrs': model_results['random0weights_raw_rdmrs'],
                    'random0weights_normed_pearsonrs': model_results['random0weights_normed_pearsonrs'],
                    'random0weights_normed_rdmrs': model_results['random0weights_normed_rdmrs'],
                })
                results_csv_list.append(results_dict)
                # Save intermediate results to CSV
                results_df = pd.DataFrame(results_csv_list)
                results_df.to_csv(csv_path, index=False)
                pbar.update(1)
