from components.modules import *
import pandas as pd
import os

"""
The `train_preprocessing` function coordinates the preparation of training data for a machine learning model.
The process starts by combining and storing information using the `merge_train_data` function, followed by loading
the merged labels from a CSV file. Subsequently, it produces masks, bounding boxes, and annotations, and then proceeds 
to convert DICOM pictures to PNG format and reorganise them. The function operates by renaming pictures, masks, and annotations 
to provide uniformity in file name. Subsequently, it compares the train pictures with their respective masks and categorises the
data into training and validation sets, utilising the preset split ratio. Every individual stage comprises regular reports on the
current state of affairs to monitor advancement and verify the fulfilment of tasks.
"""
def train_preprocessing(scale_factor,split_ratio, train_series_meta, train_df, train_merged, segmentation_dir, mask_dir, annotation_dir, 
                        train_images_dir, train_images_png_dir, final_train_images_png_dir, final_mask_dir, 
                        final_annotation_dir, matched_train_images_dir, dataset_dir, matched_output_path, unmatched_output_path):
    # Call the function to merge and save the DataFrame
    merge_train_data(train_series_meta, train_df, train_merged)
    # Load label CSV into DataFrame
    label_df = pd.read_csv(train_merged)
    # Define combined labels mapping (example)
    combined_labels = {
    'bowel_healthy': 0,
    'bowel_injury': 1,
    'liver_healthy': 2,
    'liver_low': 3,
    'liver_high': 4,
    'spleen_healthy': 5,
    'spleen_low': 6,
    'spleen_high': 7,
    'kidney_healthy': 8,
    'kidney_low': 9,
    'kidney_high': 10
     }
    # Generate masks, bounding boxes, and annotations
    print("\nGenerating bounding boxes and annotations.......................................................\n")
    matched_series_ids = make_masks_bounding_box_and_annotations(scale_factor, segmentation_dir, train_images_dir, mask_dir, annotation_dir, label_df, 
                                                                 combined_labels,matched_output_path, unmatched_output_path)
    print("\nBounding boxes and annotations generated........................................................\n")
    # Convert .dcm images to PNG and restructure
    print("\nSelecting matched series Id from train images...................................................\n")
    # Call your function here
    matched_series_id_segmentation_train_images(train_images_dir, matched_series_ids, train_images_png_dir)
    print("\nSelecting matched series Id from train images have been successfully completed..................\n")
    # Rename images
    print("\nRenaming images...............................................................\n")
    renaming_images(train_images_png_dir, final_train_images_png_dir)
    print("\nImages renamed................................................................\n")
    # Rename masks
    print("\nRenaming masks................................................................\n")
    renaming_masks(mask_dir, final_mask_dir)
    print("\nMasks renamed.................................................................\n")
    # Rename annotations
    print("\nRenaming annotations...........................................................\n")
    rename_annotations(annotation_dir, final_annotation_dir)
    print("\nAnnotations renamed............................................................\n")
    # Consider only matched train images with masks
    print("\nConsidering matched train images with masks....................................\n")
    consider_matched_train_images_with_masks(final_mask_dir, final_train_images_png_dir, matched_train_images_dir)
    print("\nMatched train images with masks considered.....................................\n")
    # Divide data into train and valid sets
    print("\nDividing data into train and valid sets........................................\n")
    data_split_train_valid(matched_train_images_dir, final_annotation_dir, dataset_dir, split_ratio)
    print("\nData divided into train and valid sets.........................................\n")

