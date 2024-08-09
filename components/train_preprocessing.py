from components.modules import *
import pandas as pd
import os

# Define the main preprocessing function
def train_preprocessing(scale_factor,split_ratio, train_series_meta, train_df, train_merged, segmentation_dir, mask_dir, annotation_dir, 
                        train_images_dir, train_images_png_dir, final_train_images_png_dir, final_mask_dir, 
                        final_annotation_dir, matched_train_images_dir, dataset_dir, matched_output_path, unmatched_output_path):
    

    # update_column_names (data_path,updated_data_path)

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

    # List all files in the directory
    # nii_files = ['21057.nii','51033.nii','397.nii','10494.nii']
    # nii_files = ['397.nii']
    # nii_files = [f for f in os.listdir(segmentation_dir) if f.endswith('.nii')]
    # # Consider the first 20 files
    # nii_files = nii_files[:10]

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






# # -----------------  preprocessing train.csv ---------------------------------------------------------------------------------

# train_series_meta = '../Inputs/train_series_meta.csv'
# train_df = '../Inputs/train.csv'
# train_merged = '../Outputs/Preprocessing/merged_dataframe.csv'

# # Call the function to merge and save the DataFrame
# merge_train_data(train_series_meta, train_df, train_merged)



# # ---------------------------- Converting .dcm train images to annoatations and masks -----------------------------------------------

# # Example usage
# segmentation_dir = '../Inputs/segmentations'
# mask_dir = '../Outputs/Preprocessing/masks'
# annotation_dir = '../Outputs/Preprocessing/annotations'
# # csv_file = '../Outputs/Preprocessing/merged_dataframe.csv'

# # Load label CSV into DataFrame
# label_df = pd.read_csv(train_merged)

# # Define combined labels mapping (example)
# combined_labels = {
#     'bowel_healthy': 0,
#     'bowel_injury': 1,
#     'liver_healthy': 2,
#     'liver_low': 3,
#     'liver_high': 4,
#     'spleen_healthy': 5,
#     'spleen_low': 6,
#     'spleen_high': 7,
#     'kidney_healthy': 8,
#     'kidney_low': 9,
#     'kidney_high': 10
# }



# # List all files in the directory
# nii_files = ['21057.nii','51033.nii','397.nii','10494.nii']

# # Call the function
# make_masks_bounding_box_and_annotations(nii_files, segmentation_dir, mask_dir, annotation_dir, label_df, combined_labels)



# # -------- converting .dcm (images) to png - converting instead of white color to grayscale image also using 512 image size ---------

# train_images_dir = '../Inputs/train_images/'
# train_images_png_dir = '../Outputs/Preprocessing/train_images_png/'
# convert_dcm_and_restructure_image(train_images_dir, train_images_png_dir)


# # ------------------------ Renaming images -------------------------------------------------------------------


# # output_dir = '../Outputs/Preprocessing/train_images_png/'
# final_train_images_png_dir = '../Outputs/Preprocessing/final_train_images_png/'

# renaming_images(train_images_png_dir, final_train_images_png_dir)


# # ------------------------------------------- Renaming masks -----------------------------------------------------------------


# # src_directory = '../Outputs/Preprocessing/masks'
# final_mask_dir = '../Outputs/Preprocessing/final_masks'

# renaming_masks(mask_dir, final_mask_dir)


# # ------------------------------------  Renaming images - train_annotations -------------------------------------------------

# # annotation_dir = '../Outputs/Preprocessing/annotations'
# final_annotation_dir = '../Outputs/Preprocessing/final_annotations'

# rename_annotations(annotation_dir, final_annotation_dir)



# # -------------------------- Consedring only matched train images with the masked images only --------------------------------------

# # folder_1 = '../Outputs/Preprocessing/final_masks'
# # folder_2 = '../Outputs/Preprocessing/final_train_images_png/'
# matched_train_images_dir = '../Outputs/Preprocessing/matched_train_images/'  

# consider_matched_train_images_with_masks(final_mask_dir, final_train_images_png_dir, matched_train_images_dir)



# # --------------------------------- Dividing data into train and valid -----------------------------------------------------

# # source_img_dir =  '../Outputs/Preprocessing/matched_train_images/'  
# # source_label_dir =  '../Outputs/Preprocessing/final_annotations'
# dataset_dir = '../Outputs/Preprocessing/datasets/data'

# data_split_train_valid(matched_train_images_dir, final_annotation_dir, dataset_dir)

# print("\nPreprocessing on training data completed")
