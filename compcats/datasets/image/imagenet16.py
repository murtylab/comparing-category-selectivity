"""
ImageNet16 dataset for the ImageNet classification task.
This dataset is a subset of the ImageNet dataset, containing 16 categories but equal number
    of images as imagenet1k.
https://github.com/rgeirhos/generalisation-humans-DNNs/blob/master/16-class-ImageNet/MSCOCO_to_ImageNet_category_mapping.txt
https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
https://github.com/fh295/semanticCNN/blob/master/imagenet_labels/mapping.txt

List of all sysids: https://files.fast.ai/data/classids.txt
Some of sysids mentioned in imagenet 16 are not present in the imagenet1k dataset.
"""
from importlib import resources
import os
import json
import torch

def get_imagenet_1k_16_map():
    """
    Get the ImageNet 1k to 16 mapping.
    """
    map_tensor_filename = 'imagenet_1k_16_map.pt'
    with resources.path('compcats.datasets.image.utils', map_tensor_filename) as path:
        if not path.exists():
            process_imagenet_labels()
        imagenet_1k_16_tensor = torch.load(path)
    return imagenet_1k_16_tensor

def get_imagenet16_label_weights(normalized=True, imagenet_train_path=None):
    """
    Get the ImageNet16 label weights.
    Args:
        normalized: If True, return the normalized weights.
        imagenet_train_path: Path to the ImageNet training dataset for counting samples per folder
    """
    if normalized:
        weights_tensor_pt_name = 'imagenet16_label_weights_normalized.pt'
    else:
        weights_tensor_pt_name = 'imagenet16_label_weights.pt'
    with resources.path('compcats.datasets.image.utils', weights_tensor_pt_name) as path:
        if not path.exists():
            process_imagenet16_label_weights(imagenet_train_path)
        folder_weights_tensor = torch.load(path)
    # Check if the tensor is of the correct shape
    if folder_weights_tensor.shape != (17,):
        raise ValueError(f"Expected tensor shape (17,), but got {folder_weights_tensor.shape}")
    return folder_weights_tensor

# ImageNet16 class mappings
IMAGENET_16_1K_STR_MAPPINGS_DICT = {
    'knife': ['n03041632'],
    'keyboard': ['n03085013', 'n04505470'],
    'elephant': ['n02504013', 'n02504458'],
    'bicycle': ['n02835271', 'n03792782'],
    'airplane': ['n02690373', 'n03955296', 'n13861050', 'n13941806'],
    'clock': ['n02708093', 'n03196217', 'n04548280'],
    'oven': ['n03259401', 'n04111414', 'n04111531'],
    'chair': ['n02791124', 'n03376595', 'n04099969', 'n00605023', 'n04429376'],
    'bear': ['n02132136', 'n02133161', 'n02134084', 'n02134418'],
    'boat': ['n02951358', 'n03344393', 'n03662601', 'n04273569', 'n04612373', 'n04612504'],
    'cat': ['n02122878', 'n02123045', 'n02123159',
        'n02126465', 'n02123394', 'n02123597',
        'n02124075', 'n02125311'],
    'bottle': ['n02823428', 'n03937543', 'n03983396',
        'n04557648', 'n04560804', 'n04579145',
        'n04591713'],
    'truck': ['n03345487', 'n03417042', 'n03770679',
        'n03796401', 'n00319176', 'n01016201',
        'n03930630', 'n03930777', 'n05061003',
        'n06547832', 'n10432053', 'n03977966',
        'n04461696', 'n04467665'],
    'car': ['n02814533', 'n03100240', 'n03100346', 'n13419325', 'n04285008'],
    'bird': ['n01321123', 'n01514859', 'n01792640',
        'n07646067', 'n01530575', 'n01531178', 'n01532829',
        'n01534433', 'n01537544', 'n01558993', 'n01562265',
        'n01560419', 'n01582220', 'n10281276', 'n01592084',
        'n01601694', 'n01614925', 'n01616318', 'n01622779',
        'n01795545', 'n01796340', 'n01797886', 'n01798484',
        'n01817953', 'n01818515', 'n01819313', 'n01820546',
        'n01824575', 'n01828970', 'n01829413', 'n01833805',
        'n01843065', 'n01843383', 'n01855032', 'n01855672',
        'n07646821', 'n01860187', 'n02002556', 'n02002724',
        'n02006656', 'n02007558', 'n02009229', 'n02009912',
        'n02011460', 'n02013706', 'n02017213', 'n02018207',
        'n02018795', 'n02025239', 'n02027492', 'n02028035',
        'n02033041', 'n02037110', 'n02051845', 'n02056570'],
    'dog': ['n02085782', 'n02085936', 'n02086079',
        'n02086240', 'n02086646', 'n02086910', 'n02087046',
        'n02087394', 'n02088094', 'n02088238', 'n02088364',
        'n02088466', 'n02088632', 'n02089078', 'n02089867',
        'n02089973', 'n02090379', 'n02090622', 'n02090721',
        'n02091032', 'n02091134', 'n02091244', 'n02091467',
        'n02091635', 'n02091831', 'n02092002', 'n02092339',
        'n02093256', 'n02093428', 'n02093647', 'n02093754',
        'n02093859', 'n02093991', 'n02094114', 'n02094258',
        'n02094433', 'n02095314', 'n02095570', 'n02095889',
        'n02096051', 'n02096294', 'n02096437', 'n02096585',
        'n02097047', 'n02097130', 'n02097209', 'n02097298',
        'n02097474', 'n02097658', 'n02098105', 'n02098286',
        'n02099267', 'n02099429', 'n02099601', 'n02099712',
        'n02099849', 'n02100236', 'n02100583', 'n02100735',
        'n02100877', 'n02101006', 'n02101388', 'n02101556',
        'n02102040', 'n02102177', 'n02102318', 'n02102480',
        'n02102973', 'n02104029', 'n02104365', 'n02105056',
        'n02105162', 'n02105251', 'n02105505', 'n02105641',
        'n02105855', 'n02106030', 'n02106166', 'n02106382',
        'n02106550', 'n02106662', 'n02107142', 'n02107312',
        'n02107574', 'n02107683', 'n02107908', 'n02108000',
        'n02108422', 'n02108551', 'n02108915', 'n02109047',
        'n02109525', 'n02109961', 'n02110063', 'n02110185',
        'n02110627', 'n02110806', 'n02110958', 'n02111129',
        'n02111277', 'n08825211', 'n02111500', 'n02112018',
        'n02112350', 'n02112706', 'n02113023', 'n02113624',
        'n02113712', 'n02113799', 'n02113978'],
}

IMAGENET_FOLDERNAME_16CAT_MAP = {}
for imagenet16_str, imagenet1k_strs in IMAGENET_16_1K_STR_MAPPINGS_DICT.items():
    for imagenet1k_str in imagenet1k_strs:
        IMAGENET_FOLDERNAME_16CAT_MAP[imagenet1k_str] = imagenet16_str

def process_imagenet16_label_weights(imagenet_train_path):
    """
    Process the ImageNet16 label weights.
    """
    # Error handling
    assert imagenet_train_path is not None, "pass ImageNet train path manually"

    # Create a dictionary to store folder name: number of samples
    imagenet1k_folder_counts = {}

    # Traverse through the ImageNet train directory
    for folder_name in os.listdir(imagenet_train_path):
        folder_path = os.path.join(imagenet_train_path, folder_name)
        if os.path.isdir(folder_path):
            # Count only JPEG files in the folder
            jpeg_count = len([f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpeg', '.jpg'))])
            imagenet1k_folder_counts[folder_name] = jpeg_count

    # Sort the dictionary by keys
    imagenet1k_folder_counts = dict(sorted(imagenet1k_folder_counts.items()))

    # Save the counts to a JSON file
    with open('./utils/imagenet1k_label_samples.json', 'w', encoding='utf-8') as f:
        json.dump(imagenet1k_folder_counts, f, indent=4)
    
    # Load the map from imagenet1k folders to imagenet16 categories
    with open('./utils/imagenet1k_16_categories.json', 'r', encoding='utf-8') as f:
        imagenet1k_16_categories = json.load(f)
    
    imagenet16_category_counts = {}
    for imagene1k_folder, imagenet16_category in imagenet1k_16_categories.items():
        if imagenet16_category not in imagenet16_category_counts:
            imagenet16_category_counts[imagenet16_category] = 0
        imagenet16_category_counts[imagenet16_category] += imagenet1k_folder_counts[imagene1k_folder]
    
    # Save the counts to a JSON file
    with open('./utils/imagenet16_label_samples.json', 'w', encoding='utf-8') as f:
        json.dump(imagenet16_category_counts, f, indent=4)
    
    # Load category to idx mapping
    with open('./utils/imagenet16_labels_idx.json', 'r', encoding='utf-8') as f:
        imagenet16_labels_to_idx = json.load(f)
    
    imagenet16_idx_counts = {}
    for imagenet16_catgeory, count in imagenet16_category_counts.items():
        imagenet16_idx = imagenet16_labels_to_idx[imagenet16_catgeory]
        imagenet16_idx_counts[imagenet16_idx] = count

    # Convert to tensor
    imagenet16_idx_counts_tensor = torch.zeros((17,), dtype=torch.float32)
    for imagenet16_idx, count in imagenet16_idx_counts.items():
        imagenet16_idx_counts_tensor[imagenet16_idx] = 1/count
    # Normalize the tensor
    imagenet16_idx_counts_tensor_normalized = imagenet16_idx_counts_tensor / imagenet16_idx_counts_tensor.sum()

    # Save the tensors
    torch.save(imagenet16_idx_counts_tensor, './utils/imagenet16_label_weights.pt')
    torch.save(imagenet16_idx_counts_tensor_normalized, './utils/imagenet16_label_weights_normalized.pt')

def process_imagenet_labels():
    """
    Convert the ImageNet labels to a dictionary mapping the labels to their corresponding indices.
    """
    # Load the raw ImageNet labels
    imagenet1k_foldername_to_str = {}
    with open('./utils/imagenet_label_maps_raw.txt', 'r', encoding='utf-8') as f:
        for line in f:
            foldername, _, str_ = line.strip().split(' ')
            imagenet1k_foldername_to_str[foldername] = str_
    # Sort the dictionary by keys
    imagenet1k_foldername_to_str = dict(sorted(imagenet1k_foldername_to_str.items()))
    imagenet1k_foldernames = list(imagenet1k_foldername_to_str.keys())
    assert len(imagenet1k_foldername_to_str) == 1000, "There should be 1000 labels in ImageNet1k"
    # Map foldername to 0-based index
    imagenet1k_foldername_to_idx = {
        foldername: idx for idx, foldername in enumerate(imagenet1k_foldername_to_str.keys())
    }
    with open('./utils/imagenet1k_labels_idx.json', 'w', encoding='utf-8') as f:
        json.dump(imagenet1k_foldername_to_idx, f, indent=4)
    # Imagenet 16 labels
    imagenet16_labels = sorted(list(IMAGENET_16_1K_STR_MAPPINGS_DICT.keys()))
    imagenet16_idx_to_labels = dict(enumerate(imagenet16_labels))
    imagenet16_labels_to_idx = {label: idx for idx, label in imagenet16_idx_to_labels.items()}
    # Save the labels
    imagenet16_labels_to_idx_copy = imagenet16_labels_to_idx.copy()
    imagenet16_labels_to_idx_copy['other'] = 16
    with open('./utils/imagenet16_labels_idx.json', 'w', encoding='utf-8') as f:
        json.dump(imagenet16_labels_to_idx_copy, f, indent=4)
    # Imagenet 1k to 16 categories
    imagenet_1k_16_categories = {}
    for imagenet1k_str in list(imagenet1k_foldername_to_str.keys()):
        imagenet_1k_16_categories[imagenet1k_str] = 'other'
        for imagenet16_str, imagenet1k_strs in IMAGENET_16_1K_STR_MAPPINGS_DICT.items():
            if imagenet1k_str in imagenet1k_strs:
                imagenet_1k_16_categories[imagenet1k_str] = imagenet16_str
                break
    # Save the categories
    with open('./utils/imagenet1k_16_categories.json', 'w', encoding='utf-8') as f:
        json.dump(imagenet_1k_16_categories, f, indent=4)
    # Imagenet 16 idx to 1k idx
    imagenet_16_idx_to_1k_idx = {}
    for imagenet16_idx, imagenet16_str in imagenet16_idx_to_labels.items():
        imagenet_16_idx_to_1k_idx[imagenet16_idx] = []
        for imagenet1k_foldername in IMAGENET_16_1K_STR_MAPPINGS_DICT[imagenet16_str]:
            if imagenet1k_foldername in imagenet1k_foldernames:
                imagenet_16_idx_to_1k_idx[imagenet16_idx].append(
                    imagenet1k_foldername_to_idx[imagenet1k_foldername]
                )
            else:
                imagenet_16_cat = IMAGENET_FOLDERNAME_16CAT_MAP[imagenet1k_foldername]
                with open('./utils/imagenet16_missing_folders.log', 'w', encoding='utf-8') as log_file:
                    log_file.write(
                        f"Foldername {imagenet1k_foldername} corresponding to {imagenet_16_cat} from Imagenet 16 not found in ImageNet1k folders\n"
                    )
    # Convert to tensor
    imagenet_1k_16_tensor = torch.full((1000,), 16, dtype=torch.int64)
    for imagenet16_idx, imagenet1k_indices in imagenet_16_idx_to_1k_idx.items():
        for imagenet1k_index in imagenet1k_indices:
            imagenet_1k_16_tensor[imagenet1k_index] = imagenet16_idx
    # Save the tensor
    torch.save(imagenet_1k_16_tensor, './utils/imagenet_1k_16_map.pt')
