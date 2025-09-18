"""
Code to load the AVA dataset
References:
- https://www.kaggle.com/datasets/nicolacarrassi/ava-aesthetic-visual-assessment/data
- HF link (alternate source): https://huggingface.co/datasets/Iceclear/AVA
- https://github.com/imfing/ava_downloader/tree/master
- TMM17 Vanishing point annotations: http://infolab.stanford.edu/~wangz/project/imsearch/Aesthetics/TMM17/
- https://arxiv.org/abs/1608.04267

Other potential datasets for vanishing points:
- https://github.com/zhou13/neurvps
- https://github.com/zihan-z/vpdet_tmm17
- https://arxiv.org/abs/2412.20666
"""
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class AVADataset(Dataset):
    """
    Dataset for AVA database
    """
    def __init__(
        self, root_dir, transform, landscape_only=True,
        vanishing_point_images=None
    ):
        self.name = 'ava'
        self.root_dir = root_dir
        self.transform = transform
        self.landscape_only = landscape_only
        self.vanishing_point_images = vanishing_point_images
        if self.landscape_only is False:
            assert self.vanishing_point_images is None, \
                "Cannot set vanishing_point_images if landscape_only is False"
        # Get tags
        self.tags_path = f'{self.root_dir}/AVA_Files/tags.txt'
        self.tag_labels = {}
        with open(self.tags_path, 'r', encoding='utf-8') as f:
            tag_lines = f.readlines()
        for line in tag_lines:
            parts = line.split()
            tag_id = int(parts[0])
            tag_name = ' '.join(parts[1:]).strip()
            self.tag_labels[tag_id] = tag_name
        # Sort tags by tag_id
        self.tag_labels = dict(sorted(self.tag_labels.items()))
        # Get annotations
        self.annotations_path = f'{self.root_dir}/AVA_Files/AVA.txt'
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            annotation_lines = f.readlines()
        self.all_annotations_df = []
        for line in annotation_lines:
            parts = line.split()
            assert len(parts) == 15, f"Line does not have 15 parts: {line}"
            image_id = int(parts[1])
            image_tag1 = int(parts[-3])
            image_tag1_label = self.tag_labels.get(image_tag1, 'unknown')
            image_tag2 = int(parts[-2])
            image_tag2_label = self.tag_labels.get(image_tag2, 'unknown')
            image_info = {
                'image_id': image_id,
                'tag1_id': image_tag1,
                'tag1_label': image_tag1_label,
                'tag2_id': image_tag2,
                'tag2_label': image_tag2_label
            }
            self.all_annotations_df.append(image_info)
        self.all_annotations_df = pd.DataFrame(self.all_annotations_df)

        # Subset annotations based on landscape_only
        if self.landscape_only:
            self.name += '_landscape'
            self.annotations_df = self.all_annotations_df.copy()
            self.annotations_df = self.annotations_df[
                (self.annotations_df['tag1_label'] == 'Landscape') |
                (self.annotations_df['tag2_label'] == 'Landscape')
            ]
            # Filter for vanishing point images if specified
            if self.vanishing_point_images is not None:
                self.missing_ids = [953841, 398594, 953897]
                assert isinstance(self.vanishing_point_images, bool), \
                    "vanishing_point_images should be a boolean or None"
                self.vp_annotations_path = f'{self.root_dir}/vp-labels'
                self.vp_image_ids = [
                    int(fname.split('.')[0]) for fname in os.listdir(
                        f'{self.vp_annotations_path}/AVA_landscape'
                    )
                ]
                if self.vanishing_point_images is True:
                    self.name += '_vp'
                    self.annotations_df = self.annotations_df[
                        self.annotations_df['image_id'].isin(self.vp_image_ids)
                    ]
                elif self.vanishing_point_images is False:
                    self.name += '_novp'
                    self.annotations_df = self.annotations_df[
                        ~self.annotations_df['image_id'].isin(self.vp_image_ids)
                    ]
                    self.annotations_df = self.annotations_df[
                        ~self.annotations_df['image_id'].isin(self.missing_ids)
                    ]
        else:
            self.annotations_df = self.all_annotations_df.copy()
        # Get image paths
        self.images_dir = f'{self.root_dir}/images'

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx):
        img_info = self.annotations_df.iloc[idx]
        img_id = img_info['image_id']
        img_path = f'{self.images_dir}/{img_id}.jpg'
        # Example - {root_dir}/953349.jpg
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def get_stimuli_dataset(
    root_dir, transform, landscape_only=True,
    vanishing_point_images=None
):
    """
    Get AVA dataset
    Args:
        root_dir(str): Root directory
        transform(torchvision.transforms): Image transformations
    """
    dataset = AVADataset(
        root_dir, transform=transform, landscape_only=landscape_only,
        vanishing_point_images=vanishing_point_images
    )
    return dataset
