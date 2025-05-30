import os
import numpy as np
import pandas as pd
import torch
import sklearn

from compcats.datasets.fmri import nsd
from compcats.utils import model_utils, misc_utils, responsecomparison_utils as rc_utils
from compcats.localizer import  localizer

import main_analysis as ma
import paths

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ImportWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=Warning)  # catch-all for any base warnings
warnings.filterwarnings("ignore")

splithalf_csv = pd.read_csv(f'{ma.SUBJECTSUBJECT_CORRELATIONS_CSV_PATH}')

# Storing results
version = 7
print(f'Running lesion ems ridge version {version}')
RIDGE_LESION_EM_CSV_PATH = f'{ma.RESULTS_DIR}/nsd515_lesion_ems_ridge_v{version}.csv'
L1POS_LESION_EM_CSV_PATH = f'{ma.RESULTS_DIR}/nsd515_lesion_ems_l1pos_v{version}.csv'
csv_path = RIDGE_LESION_EM_CSV_PATH
if os.path.exists(csv_path):
    results_csv = pd.read_csv(csv_path)
    results_csv_list = results_csv.to_dict('records')
else:
    results_csv_list = []

alpha_values = np.array([1e-2, 1e-1, 1e-0, 1e+1, 1e+2])
n_splits = 5
seed = 42
trainval_kf = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=seed)

tthreshold = 7
model_localizer = 'grillspector'

train_subset = 485
test_subset = 515
frois = ['FFA', 'PPA', 'EBA']

subjects = [1, 2, 5, 7]

def get_model_names(version_):
    """
    Get names of models used for the analysis
    """
    # 'resnet18tdannsupervisedlw10', 'resnet18tdannsimclrlw1', 'resnet50robustl2eps3', 'resnet50faceobfuscation', 'cornets', 'cornetz'
    all_models = [
        'alexnetimagenet1kv1', 'alexnetipclref01', 'alexnetbarlowtwins', 'vgg16imagenet1kv1', 'resnet18imagenet1kv1', \
        'resnet18tdannsupervisedlw10', 'resnet18tdannsimclrlw1', 'resnet50imagenet1kv2', 'resnet50openaiclip', 'resnet50simclr', \
        'resnet50robustl2eps3', 'resnet50sin', 'resnet50ecoset', 'resnet50faceobfuscation', 'resnet50toponettau30', \
        'resnet50cesslbarlowtop', 'resnet101imagenet1kv1', 'resnet101openaiclip', 'bagnet33', 'cornets', \
        'cornetz', 'vitb32imagenet1kv1', 'vitb32openaiclip', 'vitb32dreamsimopenclip', 'vitb16imagenet21k', \
        'vitb16openaiclip', 'vitb16siglip2res224', 'vitb16eva028bs8bb131k', 'vitb16synclr', 'vitb14dinov2', \
        'vitb14dinov2reg', 'vitl14openaiclip', 'vitl14kosmos2']
    models = []
    # Select models based on version_
    if version_ == 1:
        models = ['alexnetimagenet1kv1', 'alexnetipclref01', 'alexnetbarlowtwins', 'vgg16imagenet1kv1', 'resnet18imagenet1kv1']
    elif version_ == 2:
        models = ['resnet18tdannsupervisedlw10', 'resnet18tdannsimclrlw1', 'resnet50imagenet1kv2', 'resnet50openaiclip', 'resnet50simclr']
    elif version_ == 3:
        models = ['resnet50robustl2eps3', 'resnet50sin', 'resnet50ecoset', 'resnet50faceobfuscation', 'resnet50toponettau30']
    elif version_ == 4:
        models = ['resnet50cesslbarlowtop', 'resnet101imagenet1kv1', 'resnet101openaiclip', 'bagnet33', 'cornets']
    elif version_ == 5:
        models = ['cornetz', 'vitb32imagenet1kv1', 'vitb32openaiclip', 'vitb32dreamsimopenclip', 'vitb16imagenet21k']
    elif version_ == 6:
        models = ['vitb16openaiclip', 'vitb16siglip2res224', 'vitb16eva028bs8bb131k', 'vitb16synclr', 'vitb14dinov2']
    elif version_ == 7:
        models = ['vitb14dinov2reg', 'vitl14openaiclip', 'vitl14kosmos2']
    return models

for froi in frois:
    for model_name in get_model_names(version):
        model = ma.get_model(model_name)
        layer = ma.get_froi_models_layers(froi, froi)[model_name]
        for subject in subjects:
            row_params = {
                'model': model_name,
                'subject': subject,
                'layer': layer,
                'model_froi': froi,
                'fmri_froi': froi,
                'model_tthreshold': tthreshold,
                'fmri_tthreshold': tthreshold,
                'model_localizer': model_localizer,
                'mapper': 'ridge'
            }
            # Check if row is present
            if misc_utils.check_row_present(row_params, results_csv_list):
                continue
            results_dict = row_params.copy()

            Y_trainval, _ = nsd.get_betas(
                paths.NSD_DIR, subject, froi, tthreshold, subset=train_subset, voxels=True
            )
            Y_trainval = Y_trainval.cpu().numpy()
            Y_test, _ = nsd.get_betas(
                paths.NSD_DIR, subject, froi, tthreshold, subset=test_subset, voxels=True
            )

            # Splithalf correlations
            splithalf_info = splithalf_csv[
                (splithalf_csv['fmri_froi'] == froi) &
                (splithalf_csv['fmri_tthreshold'] == tthreshold) &
                (splithalf_csv['fmri_dataset'] == f'nsd{test_subset}') &
                (splithalf_csv['subject1'] == subject)
            ].iloc[0]
            splithalf_pearsonr = splithalf_info['subject1_splithalf_pearsonr'].item()
            splithalf_rdmr = splithalf_info['subject1_splithalf_rdmr'].item()
            results_dict['fmri_splithalf_pearsonr'] = splithalf_pearsonr
            results_dict['fmri_splithalf_rdmr'] = splithalf_rdmr

            X_trianval_layer = model_utils.get_dataset_activations(
                ma.MODELS_DIR, paths.NSD_DIR, model, ma.get_froi_models_layers(froi, froi)[model_name], 
                'nsd', subset=train_subset
            )
            X_test_layer = model_utils.get_dataset_activations(
                ma.MODELS_DIR, paths.NSD_DIR, model, ma.get_froi_models_layers(froi, froi)[model_name], 
                'nsd', subset=test_subset
            )
            model_mask_froi = localizer.get_froi_mask(
                ma.get_localizer_dir(model_localizer), model_localizer, ma.MODELS_DIR, model, layer, froi, tthreshold
            )
            model_mask_froi_t0 = localizer.get_froi_mask(
                ma.get_localizer_dir(model_localizer), model_localizer, ma.MODELS_DIR, model, layer, froi, 0
            )
            model_mask_froi_lesioned = ~model_mask_froi_t0

            print('\n\n')
            print(f'FROI: {froi}, Model: {model_name}, Subject: {subject}, mfROI -> fROI')
            print('\n\n')
            X_trainval = X_trianval_layer[:, model_mask_froi]
            X_test = X_test_layer[:, model_mask_froi]
            X_trainval = X_trainval.cpu().numpy()
            X_test = X_test.cpu().numpy()
            mapping = sklearn.model_selection.GridSearchCV(
                estimator=sklearn.linear_model.Ridge(random_state=seed),
                param_grid={'alpha': alpha_values},
                cv=trainval_kf,
                scoring='r2',
                n_jobs=-1,
            )
            mapping.fit(X_trainval, Y_trainval)
            y_pred = torch.FloatTensor(mapping.predict(X_test))
            # Get test performances
            pearsonr = rc_utils.pearsonr(Y_test.mean(axis=-1), y_pred.mean(axis=-1))
            rdmr = rc_utils.rdmr(Y_test, y_pred)
            # Store results
            results_dict['froi_raw_pearsonr'] = pearsonr
            results_dict['froi_normed_pearsonr'] = (pearsonr / np.sqrt(splithalf_pearsonr)).item()
            results_dict['froi_raw_rdmr'] = rdmr
            results_dict['froi_normed_rdmr'] = (rdmr / np.sqrt(splithalf_rdmr)).item()
            
            print('\n\n')
            print(f'FROI: {froi}, Model: {model_name}, Subject: {subject}, mfROI lesioned -> fROI')
            print('\n\n')
            X_trainval = X_trianval_layer[:, model_mask_froi_lesioned]
            X_test = X_test_layer[:, model_mask_froi_lesioned]
            X_trainval = X_trainval.cpu().numpy()
            X_test = X_test.cpu().numpy()
            mapping = sklearn.model_selection.GridSearchCV(
                estimator=sklearn.linear_model.Ridge(random_state=seed),
                param_grid={'alpha': alpha_values},
                cv=trainval_kf,
                scoring='r2',
                n_jobs=-1,
            )
            mapping.fit(X_trainval, Y_trainval)
            y_pred = torch.FloatTensor(mapping.predict(X_test))
            # Get test performances
            pearsonr = rc_utils.pearsonr(Y_test.mean(axis=-1), y_pred.mean(axis=-1))
            rdmr = rc_utils.rdmr(Y_test, y_pred)
            # Store results
            results_dict['lesion_raw_pearsonr'] = pearsonr
            results_dict['lesion_normed_pearsonr'] = (pearsonr / np.sqrt(splithalf_pearsonr)).item()
            results_dict['lesion_raw_rdmr'] = rdmr
            results_dict['lesion_normed_rdmr'] = (rdmr / np.sqrt(splithalf_rdmr)).item()

            # Save results
            results_csv_list.append(results_dict)
            results_csv = pd.DataFrame(results_csv_list)
            results_csv.to_csv(
                csv_path, index=False
            )
