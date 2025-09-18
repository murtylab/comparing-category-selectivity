# COCO Utils
from pycocotools.coco import COCO

def get_image_labels(image_id, annotations_obj, captions_obj):
    """
    Get the labels for an image given its ID from the COCO dataset.
    
    Args:
        image_id (int): The ID of the image.
        
    Returns:
        list: A list of labels (category names) associated with the image.
    """
    # Get all annotations for the given image
    annotation_ids = annotations_obj.getAnnIds(imgIds=image_id)
    annotations = annotations_obj.loadAnns(annotation_ids)
    # Get category IDs from annotations
    category_ids = {annotation['category_id'] for annotation in annotations}
    # Get category names from category IDs
    category_names = [annotations_obj.loadCats(cat_id)[0]['name'] for cat_id in category_ids]

    # Get captions for the image
    caption_ids = captions_obj.getAnnIds(imgIds=image_id)
    captions = captions_obj.loadAnns(caption_ids)
    caption_texts = [caption['caption'] for caption in captions]

    return category_names, caption_texts

def add_image_info_to_df(df, annotations_path):
    """
    Add labels to the DataFrame given the COCO object.
    
    Args:
        df (pd.DataFrame): The DataFrame to which to add the labels.
        
    Returns:
        pd.DataFrame: The DataFrame with the labels added.
    """
    # Paths and COCO objects
    coco_train_annotations = COCO(f'{annotations_path}/instances_train2017.json')
    coco_train_captions = COCO(f'{annotations_path}/captions_train2017.json')
    coco_val_annotations = COCO(f'{annotations_path}/instances_val2017.json')
    coco_val_captions = COCO(f'{annotations_path}/captions_val2017.json')

    # Create a copy of the dataframe to avoid the SettingWithCopyWarning
    df_copy = df.copy()

    # Create a function to get the appropriate COCO object based on the split
    def get_annotations_obj(split):
        if split == 'train2017':
            return coco_train_annotations
        else:
            return coco_val_annotations
    def get_captions_obj(split):
        if split == 'train2017':
            return coco_train_captions
        else:
            return coco_val_captions

    # Apply the function to get labels for each row using the appropriate COCO object
    df_copy.loc[:, 'labels'] = df_copy.apply(
        lambda row: get_image_labels(
            row['cocoId'], 
            annotations_obj=get_annotations_obj(row['cocoSplit']), 
            captions_obj=get_captions_obj(row['cocoSplit'])
        )[0],
        axis=1
    )
    df_copy.loc[:, 'captions'] = df_copy.apply(
        lambda row: get_image_labels(
            row['cocoId'], 
            annotations_obj=get_annotations_obj(row['cocoSplit']), 
            captions_obj=get_captions_obj(row['cocoSplit'])
        )[1], 
        axis=1
    )
    return df_copy
