import os
import numpy as np
import pandas as pd
import sklearn.linear_model
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
LESION_EM_CSV_PATH = f'{ma.RESULTS_DIR}/nsd515_lesion_ems_fixedalpha'
csv_path = LESION_EM_CSV_PATH
if version >= 0:
    csv_path += f'_v{version}'
csv_path += '.csv'
if os.path.exists(csv_path):
    results_csv = pd.read_csv(csv_path)
    results_csv_list = results_csv.to_dict('records')
else:
    results_csv_list = []

seed = 42
fmri_tthresholds = [11, 9, 7, 5, 3]
model_tthresholds = [11, 9, 7, 5, 3, -1] # -1 means no thresholding i.e. full layer
model_localizer = 'grillspector'
train_subset = 485
test_subset = 515
frois = ['FFA', 'PPA', 'EBA']
subjects = [1, 2, 5, 7]
# mappers = ['ridge', 'l1pos']
mappers = ['ridge']  # For now, only ridge is used
alpha_dict = {
    'ridge': 0.1,
    'l1pos': 0.01
}

# Dev
# version = 0
# frois = ['EBA']
# subjects = [1]
# model_tthresholds = [7, -1]
# mappers = ['l1pos']
# alpha = 0.01

def get_model_names(version_):
    """
    Get names of models used for the analysis
    """
    # 'resnet18tdannsupervisedlw10', 'resnet18tdannsimclrlw1', 'resnet50robustl2eps3', 'resnet50faceobfuscation', 'cornets', 'cornetz'
    all_models = [
        'alexnetimagenet1kv1', 'alexnetipclref01', 'alexnetbarlowtwins', 'vgg16imagenet1kv1', 'resnet18imagenet1kv1', \
        'resnet18tdannsupervisedlw10', 'resnet18tdannsimclrlw1', 'resnet50imagenet1kv2', 'resnet50openaiclip', 'resnet50simclr', \
        'resnet50robustl2eps3', 'resnet50sin', 'resnet50ecoset', 'resnet50faceobfuscation', 'resnet50toponettau30', \
        'resnet50cesslbarlowtop', 'resnet50blurstrong', 'resnet101imagenet1kv1', 'resnet101openaiclip', 'bagnet33', 'cornets', \
        'cornetz', 'convnexttinyimagenet1kv1', 'vitb32imagenet1kv1', 'vitb32openaiclip', 'vitb32dreamsimopenclip', 'vitb16imagenet21k', \
        'vitb16openaiclip', 'vitb16siglip2res224', 'vitb16eva028bs8bb131k', 'vitb16synclr', 'vitb14dinov2', \
        'vitb14dinov2reg', 'vitl14openaiclip', 'vitl14kosmos2']
    models = []
    # Select models based on version
    if version == -1:  # All models
        models = all_models
    if version_ == 0: # Dev
        models = ['resnet18imagenet1kv1']
    elif version_ == 1:
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
    elif version == 7:
        models = ['vitb14dinov2reg', 'vitl14openaiclip', 'vitl14kosmos2', 'resnet50blurstrong', 'convnexttinyimagenet1kv1']
    return models

for froi in frois:
    for fmri_tthreshold in fmri_tthresholds:
        for subject in subjects:
            # Get subject data
            nsd_dir = paths.NSD_DIR
            Y_trainval, _ = nsd.get_betas(
                nsd_dir, subject, froi, fmri_tthreshold, subset=train_subset, voxels=True
            )
            Y_trainval = Y_trainval.cpu().numpy()
            Y_test, _ = nsd.get_betas(
                nsd_dir, subject, froi, fmri_tthreshold, subset=test_subset, voxels=True
            )
            # Splithalf correlations
            splithalf_info = splithalf_csv[
                (splithalf_csv['fmri_froi'] == froi) &
                (splithalf_csv['fmri_tthreshold'] == fmri_tthreshold) &
                (splithalf_csv['fmri_dataset'] == f'nsd{test_subset}') &
                (splithalf_csv['subject1'] == subject)
            ].iloc[0]
            splithalf_pearsonr = splithalf_info['subject1_splithalf_pearsonr'].item()
            splithalf_rdmr = splithalf_info['subject1_splithalf_rdmr'].item()
            for model_name in get_model_names(version):
                # Get model and layer
                model = ma.get_model(model_name)
                layer = ma.get_froi_models_layers(froi, froi)[model_name]
                # Get activations for the model and layer
                X_trianval_layer = model_utils.get_dataset_activations(
                    ma.MODELS_DIR, paths.NSD_DIR, model, ma.get_froi_models_layers(froi, froi)[model_name],
                    'nsd', subset=train_subset
                )
                X_test_layer = model_utils.get_dataset_activations(
                    ma.MODELS_DIR, paths.NSD_DIR, model, ma.get_froi_models_layers(froi, froi)[model_name],
                    'nsd', subset=test_subset
                )
                # Number of units in the layer
                n_units_layer = X_trianval_layer[0].numel()
                # Loop through model tthresholds
                for model_tthreshold in model_tthresholds:
                    # Get model mask for the froi
                    n_units_lesioned = 0
                    if model_tthreshold > 0:
                        model_mask_froi = localizer.get_froi_mask(
                            ma.get_localizer_dir(model_localizer), model_localizer, ma.MODELS_DIR,\
                            model, layer, froi, model_tthreshold
                        )
                        model_mask_froi_lesioned = ~model_mask_froi
                        X_trainval = X_trianval_layer[:, model_mask_froi_lesioned]
                        X_test = X_test_layer[:, model_mask_froi_lesioned]
                        n_units_lesioned = model_mask_froi.sum().item()
                    else:
                        X_trainval = X_trianval_layer.reshape(X_trianval_layer.shape[0], -1)
                        X_test = X_test_layer.reshape(X_test_layer.shape[0], -1)
                    X_trainval = X_trainval.cpu().numpy()
                    X_test = X_test.cpu().numpy()
                    for mapper in mappers:
                        alpha = alpha_dict[mapper]
                        row_params = {
                            'model': model_name,
                            'subject': subject,
                            'layer': layer,
                            'model_froi': froi,
                            'fmri_froi': froi,
                            'model_tthreshold': model_tthreshold,
                            'fmri_tthreshold': fmri_tthreshold,
                            'model_localizer': model_localizer,
                            'mapper': mapper,
                            'alpha': alpha,
                            'train_subset': train_subset,
                            'test_subset': test_subset,
                            'seed': seed,
                            'fmri_splithalf_pearsonr': splithalf_pearsonr,
                            'fmri_splithalf_rdmr': splithalf_rdmr,
                            'n_units_layer': n_units_layer,
                            'n_units_lesioned': n_units_lesioned,
                            'percent_units_lesioned': (n_units_lesioned / n_units_layer) * 100
                        }
                        # Check if row is present
                        if misc_utils.check_row_present(row_params, results_csv_list):
                            continue
                        results_dict = row_params.copy()

                        print('\n')
                        print(f'FROI: {froi}, Model: {model_name}, Subject: {subject}, tthreshold: {model_tthreshold}, mapper: {mapper}')
                        print('\n')
                        if mapper == 'l1pos':
                            # Use L1 positive mapping
                            mapping = sklearn.linear_model.Lasso(
                                alpha=alpha, positive=True, random_state=seed
                            )
                        elif mapper == 'ridge':
                            # Use Ridge regression mapping
                            mapping = sklearn.linear_model.Ridge(alpha=alpha, random_state=seed)
                        mapping.fit(X_trainval, Y_trainval)
                        y_pred = torch.FloatTensor(mapping.predict(X_test))
                        # Get test performances
                        pearsonr = rc_utils.pearsonr(Y_test.mean(axis=-1), y_pred.mean(axis=-1))
                        rdmr = rc_utils.rdmr(Y_test, y_pred)
                        # Store results
                        results_dict['raw_pearsonr'] = pearsonr
                        results_dict['normed_pearsonr'] = (pearsonr / np.sqrt(splithalf_pearsonr)).item()
                        results_dict['raw_rdmr'] = rdmr
                        results_dict['normed_rdmr'] = (rdmr / np.sqrt(splithalf_rdmr)).item()

                        # Save results
                        results_csv_list.append(results_dict)
                        results_csv = pd.DataFrame(results_csv_list)
                        results_csv.to_csv(
                            csv_path, index=False
                        )
