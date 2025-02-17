import os
import numpy as np
import pandas as pd
import cv2
import nibabel as nib
import pydicom
import shutil
import random
import pandas as pd
import shutil
import json

""" 
preprocessing train.csv
Merges two CSV files with patient and series data based on the 'patient_id' column, adds a new column 'combined_id' by concatenating 'patient_id' 
and'series_id', then saves the merged DataFrame to a new CSV file.  
"""

def merge_train_data(train_series_meta_path, train_df_path, output_csv_path):
    train_series_meta = pd.read_csv(train_series_meta_path)
    train_df = pd.read_csv(train_df_path)
    merged_df = pd.merge(train_series_meta, train_df, on='patient_id')
    merged_df['combined_id'] = merged_df['patient_id'].astype(str) + '_' + merged_df['series_id'].astype(str)
    merged_df.to_csv(output_csv_path, index=False)
    print(f"\nMerged DataFrame saved to {output_csv_path}")

""" 
Matches series IDs from segmentation files and training image directories, saving both matched and unmatched series IDs to JSON files. 
"""

def match_series_ids(segmentations_dir, train_images_dir, matched_output_path, unmatched_output_path):
    segmentation_files = [f for f in os.listdir(segmentations_dir) if f.endswith('.nii')]
    segmentation_series_ids = [os.path.splitext(f)[0] for f in segmentation_files]
    train_images_series_ids = []
    for patient_id in os.listdir(train_images_dir):
        patient_path = os.path.join(train_images_dir, patient_id)
        if os.path.isdir(patient_path):
            for series_id in os.listdir(patient_path):
                series_path = os.path.join(patient_path, series_id)
                if os.path.isdir(series_path):
                    train_images_series_ids.append(series_id)

    segmentation_series_set = set(segmentation_series_ids)
    train_images_series_set = set(train_images_series_ids)
    matched_series_ids = segmentation_series_set.intersection(train_images_series_set)
    unmatched_series_ids = segmentation_series_set.difference(train_images_series_set)
    matched_count = len(matched_series_ids)
    unmatched_count = len(unmatched_series_ids)
    matched_series_ids_nii = {f"{series_id}.nii" for series_id in matched_series_ids}
    unmatched_series_ids_nii = {f"{series_id}.nii" for series_id in unmatched_series_ids}
    print(f"\nTotal series IDs in segmentations: {len(segmentation_series_ids)}")
    print(f"\nTotal series IDs in train_images: {len(train_images_series_ids)}")
    print(f"\nMatched series IDs: {matched_count}")
    print(f"\nUnmatched series IDs: {unmatched_count}")
    with open(matched_output_path, 'w') as matched_file:
        json.dump(list(matched_series_ids_nii), matched_file, indent=4)
    with open(unmatched_output_path, 'w') as unmatched_file:
        json.dump(list(unmatched_series_ids_nii), unmatched_file, indent=4)
    return matched_series_ids_nii

"""
-------------------------- Converting .png train images to annoatations and masks -----------------------------------------------
This function analyses medical images by producing masks, bounding boxes, and annotations for organs using segmented NIfTI (.nii) files. 
The process begins by comparing the series IDs in the segmentation and training picture folders. 
Once a match is found, the relevant NIfTI files are loaded. The method performs image scaling and resizing operations on the image slices. 
It also generates masks for certain organs based on given greyscale values. Additionally, it identifies contours to construct bounding boxes around the organs. Subsequently, it associates these rectangular regions with specific organ labels using information from a supplied DataFrame and stores the annotations in YOLO format. The masks and annotations that have been processed are stored to designated folders. Additionally, the function provides a list of series IDs that have been matched.

"""

def make_masks_bounding_box_and_annotations(scale_factor, segmentation_dir,train_images_dir, mask_dir, annotation_dir, label_df, 
                                            combined_labels,matched_output_path, unmatched_output_path):   
    # Define the grayscale colors and corresponding organs
    grayscale_colors = [(51, 51, 51), (255, 255, 255), (102, 102, 102), (204, 204, 204), (153, 153, 153)]
    organs = ['bowel', 'liver', 'spleen', 'right kidney', 'left kidney']
    # Ensure the output directories exist
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)
    matched_series_ids = match_series_ids(segmentation_dir, train_images_dir,matched_output_path, unmatched_output_path)
    matched_series_ids = list(matched_series_ids)
    print("matched_series_ids",matched_series_ids)
    for nii_file in matched_series_ids:
        nii_path = os.path.join(segmentation_dir, nii_file)
        if not os.path.exists(nii_path):
            print(f"File {nii_file} not found in the directory. Skipping...")
            continue
        print(f"\n.............Processing..........:   {nii_file}")
        # Load the NIfTI image
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata()
        print(f"Shape of Image: {data.shape}")
        images = []
        base_filename = os.path.splitext(os.path.basename(nii_path))[0]
        # Find the matching row in label_df based on series_id
        series_id = str(base_filename)  # Convert base_filename to string
        filtered_df = label_df[label_df['series_id'].astype(str) == series_id]
        if filtered_df.empty:
            print(f"\nNo matching entry found in label_df for series_id {series_id}. Skipping processing.")
            continue
        # Get the first row (assuming only one entry per series_id)
        label_row = filtered_df.iloc[0]
        # Create a combined ID for naming based on series_id and additional information
        combined_id = f"{label_row['combined_id']}"  # Modify this as per your data
        for i in range(data.shape[2]):
            img = data[:, :, i] * scale_factor
            if img.max() > 0:  # Check if the maximum pixel value in the image is greater than 0
                img = np.rot90(img)  # Rotate the image
                # Resize the image 
                resized_img = cv2.resize(img, (512, 512))
                # Initialize a blank mask image
                mask_img = np.zeros_like(resized_img, dtype=np.uint8)
                bboxes = []
                for idx, grayscale in enumerate(grayscale_colors):
                    mask = resized_img == grayscale[0]
                    mask_img[mask] = grayscale[0]
                    # Find contours and create bounding boxes
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        bboxes.append((organs[idx], x, y, w, h))  # Store organ name instead of index
                # Save the resized image as a PNG file with combined ID and slice number
                png_filename = os.path.join(mask_dir, f'{combined_id}_{i:04d}.png')
                # print(f"Saving PNG mask: {png_filename}")
                cv2.imwrite(png_filename, resized_img)
                # Save the bounding boxes in YOLO format with combined ID and slice number
                annotation_filename = os.path.join(annotation_dir, f'{combined_id}_{i:04d}.txt')
                # print(f"Saving annotation: {annotation_filename}")
                with open(annotation_filename, 'w') as f:
                    for bbox in bboxes:
                        organ, x, y, w, h = bbox
                        # Map CSV labels to defined organs
                        if organ == 'right kidney' or organ == 'left kidney':
                            label = 'kidney'
                        else:
                            label = organ
                        # Define label based on CSV information
                        label_info = {
                            'liver': ('healthy' if label_row['liver_healthy'] == 1 else
                                      'low' if label_row['liver_low'] == 1 else
                                      'high' if label_row['liver_high'] == 1 else None),
                            'spleen': ('healthy' if label_row['spleen_healthy'] == 1 else
                                       'low' if label_row['spleen_low'] == 1 else
                                       'high' if label_row['spleen_high'] == 1 else None),
                            'bowel': ('healthy' if label_row['bowel_healthy'] == 1 else
                                      'injury' if label_row['bowel_injury'] == 1 else None),
                            'kidney': ('healthy' if label_row['kidney_healthy'] == 1 else
                                       'low' if label_row['kidney_low'] == 1 else
                                       'high' if label_row['kidney_high'] == 1 else None)
                        }
                        # Only proceed if the organ's label is not None
                        if label_info[label] is not None:
                            combined_label = f"{label}_{label_info[label]}"
                            # Assign class ID based on combined labels mapping
                            if combined_label in combined_labels:
                                cls_id = combined_labels[combined_label]
                                x_center = (x + w / 2) / resized_img.shape[1]
                                y_center = (y + h / 2) / resized_img.shape[0]
                                width = w / resized_img.shape[1]
                                height = h / resized_img.shape[0]
                                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            else:
                                print(f"Warning: Combined label '{combined_label}' not found in combined_labels. Skipping annotation...")
                        else:
                            print(f"Warning: Organ '{label}' is not present in the mask for {combined_id}. Skipping annotation...")
                images.append(resized_img)
        print(f"\nTotal slices processed: {len(images)}")
    return matched_series_ids

"""

----------- converting .dcm (images) to png - converting instead of white color to grayscale image also using 512 image size -------
This function transforms a DICOM file, which stands for Digital Imaging and Communications in Medicine, into a PNG picture in greyscale. 
The program analyses the DICOM file, retrieves and standardises the pixel data, and subsequently modifies the pixel values according to the image's photometric 
interpretation, perhaps inverting them. The pixel data is rescaled to a standardised resolution of 512x512 and stored as a PNG picture in greyscale format. 
The function creates a fresh filename by combining the patient ID, series ID, and picture ID, and then stores the image to the designated destination location. 
The function provides the updated filename along with a boolean indicator of success, or an error message if the processing is unsuccessful.

"""
def convert_dcm_to_grayscale_png(src_path, dest_path, patient_id, series_id, image_id):
    try:
        # Read DICOM file
        img = pydicom.dcmread(src_path)
        # Check Photometric Interpretation (PIP)
        pip = img.PhotometricInterpretation
        # Extract pixel data
        data = img.pixel_array
        # Normalize pixel values (0-1 range)
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        # Convert to grayscale (0-255 range)
        data = (data * 255).astype(np.uint8)
        data = cv2.resize(data, dsize=(512, 512))
        # Adjust pixel values based on Photometric Interpretation
        if (pip == 'MONOCHROME1'):
            data = 255 - data  # Invert pixel values
        # Construct filename and save as grayscale PNG
        new_filename = f"{patient_id}_{series_id}_{str(image_id).zfill(4)}.png"
        new_filepath = os.path.join(dest_path, new_filename)
        cv2.imwrite(new_filepath, data, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # Save as grayscale PNG
        return new_filename, True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return None, False
    
"""

This function iterates through a directory hierarchy that contains DICOM (.dcm) files. 
It converts each DICOM file to a greyscale PNG using the `convert_dcm_to_grayscale_png` function and then saves the resulting images to a designated output directory. 
The original structure is preserved by retrieving the patient ID and series ID from the directory hierarchy. The function manages errors by gathering 
any unsuccessful conversions and displaying a summary of the overall number of images converted and any discovered faults. An output directory is created if 
it does not already exist. This feature is beneficial for doing batch processing and arranging medical images.

"""
def convert_dcm_and_restructure_image(root_dir, output_dir):
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        errors = []
        dataset = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if not file.endswith('.dcm'):
                    continue
                parts = os.path.relpath(root, root_dir).split(os.path.sep)
                patient_id = parts[0]
                series_id = parts[1]
                image_id = os.path.splitext(file)[0]  # Extract file name without extension
                src_path = os.path.join(root, file)
                new_filename, success = convert_dcm_to_grayscale_png(src_path, output_dir, patient_id, series_id, image_id)
                if success:
                    dataset.append(new_filename)
                else:
                    errors.append(src_path)
        print(f"Total converted images: {len(dataset)}, Errors: {len(errors)}")
        if errors:
            print("Errors occurred while processing the following files:")
            for err in errors:
                print(err)
    except Exception as e:
        print(f"An error occurred: {e}")

"""
This function categorises and duplicates certain PNG pictures from a directory hierarchy that includes medical images. 
The function accepts a list of matched series IDs and iterates through each patient in the root directory. 
It then verifies if the series directory for each patient is present in the list of matched series IDs. If a successful match is detected, 
all PNG pictures belonging to that particular series will be duplicated and transferred to a designated destination directory. 
The function verifies the existence of the destination directory prior to copying files. 
"""
def matched_series_id_segmentation_train_images(root_dir, matched_series_ids, target_dir):
    matched_series_ids = [series_id.replace('.nii', '') for series_id in matched_series_ids]
    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Iterate through each patient directory
    for patient_dir in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        # Iterate through each series directory within the patient directory
        for series_id in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series_id)
            series_path = series_path.replace(os.path.sep, '/')
            if not os.path.isdir(series_path):
                continue
            # Check if the current series_id is in the matched_series_ids set
            if series_id in matched_series_ids:
                # Copy all .png images to the target directory
                for image_file in os.listdir(series_path):
                    if image_file.endswith('.png'):
                        image_path = os.path.join(series_path, image_file)
                        shutil.copy(image_path, target_dir)  # Copy image to target directory

"""

------------------------------------------ Renaming images - train_images ----------------------------------------------------
This method performs the task of renaming and organising PNG pictures from a source directory according to a certain naming convention, 
and thereafter transfers them to a designated destination directory. The process begins by scanning the source directory in order to gather all picture files. 
These files are then grouped based on the unique combinations of the first two sections of their filenames, which correspond to the folder names they are located in. 
The iamges in each group are organised and renamed in a sequential manner, with a zero-padded series number appended to the filenames. 
Subsequently, the renamed pics are duplicated to the designated directory. Upon completion of the processing, the initial source directory is removed in order to clean up. 

"""
def renaming_images(src_directory, dest_directory):
    def get_image_files(src_path):
        image_files = []
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith('.png'):
                    image_files.append(os.path.join(root, file))
        return image_files
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # Get all image files from the source path
    image_files = get_image_files(src_directory)
    # Dictionary to keep track of processed folders and their series
    folder_dict = {}
    # First pass to identify unique folder combinations and their images
    for img_file in image_files:
        # Extract filename without extension
        img_name = os.path.basename(img_file)
        img_name_no_ext = os.path.splitext(img_name)[0]
        # Split filename by underscores to get folder_1_name and folder_2_name
        parts = img_name_no_ext.split('_')
        if len(parts) < 2:
            print(f"Skipping file {img_file} as it does not follow the expected naming convention.")
            continue
        folder_1_name = parts[0]
        folder_2_name = parts[1]
        # Generate a unique key for folder_1_name and folder_2_name
        folder_key = f"{folder_1_name}_{folder_2_name}"
        # Initialize the list of images if this folder_key is encountered for the first time
        if folder_key not in folder_dict:
            folder_dict[folder_key] = []
        # Append the image file to the corresponding folder key
        folder_dict[folder_key].append(img_file)
    # Second pass to rename and copy images in sequence
    for folder_key, images in folder_dict.items():
        images.sort()  # Sort images to ensure they are processed in order
        for idx, img_file in enumerate(images):
            img_name = os.path.basename(img_file)
            img_name_no_ext = os.path.splitext(img_name)[0]
            parts = img_name_no_ext.split('_')
            if len(parts) < 3:
                continue
            folder_1_name = parts[0]
            folder_2_name = parts[1]
            # Generate new series number
            new_series_num = str(idx).zfill(4)
            # Construct new image name
            new_img_name = f"{folder_1_name}_{folder_2_name}_{new_series_num}.png"
            # Full new path with new image name
            new_img_full_path = os.path.join(dest_directory, new_img_name)
            # Copy and rename the image to the new path
            shutil.copy(img_file, new_img_full_path)
    print(f"Renaming and copying completed. Processed {len(image_files)} images.")
    #After successful processing, delete the source directory
    try:
        shutil.rmtree(src_directory)
        print(f"Source directory {src_directory} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {src_directory}: {e}")


"""
------------------------------------ Renaming images - train_masks -----------------------------------------------------------
This method rearranges and rebrands PNG mask images from a source directory, and then duplicates them to a target directory. 
The program examines the source directory for PNG files and categorises them according to the first two portions of their filenames, 
which correspond to folder names. The method thereafter handles each group by systematically renaming each image with a series number 
that includes leading zeros, guaranteeing that images within the same group are arranged and titled uniformly. 
The renamed images are duplicated to the designated location. Once the images have been processed properly, the source directory
is removed in order to clear up. This function facilitates the structure and organising of mask images, making them more manageable and convenient to utilise.

"""
def renaming_masks(src_directory, dest_directory):
    def get_image_files(src_path):
        image_files = []
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith('.png'):
                    image_files.append(os.path.join(root, file))
        return image_files
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # Get all image files from the source path
    image_files = get_image_files(src_directory)
    # Dictionary to keep track of processed folders and their series
    folder_dict = {}
    # First pass to identify unique folder combinations and their images
    for img_file in image_files:
        # Extract filename without extension
        img_name = os.path.basename(img_file)
        img_name_no_ext = os.path.splitext(img_name)[0]
        # Split filename by underscores to get folder_1_name and folder_2_name
        parts = img_name_no_ext.split('_')
        if len(parts) < 2:
            print(f"Skipping file {img_file} as it does not follow the expected naming convention.")
            continue
        folder_1_name = parts[0]
        folder_2_name = parts[1]
        # Generate a unique key for folder_1_name and folder_2_name
        folder_key = f"{folder_1_name}_{folder_2_name}"
        # Initialize the list of images if this folder_key is encountered for the first time
        if folder_key not in folder_dict:
            folder_dict[folder_key] = []
        # Append the image file to the corresponding folder key
        folder_dict[folder_key].append(img_file)
    # Second pass to rename and copy images in sequence
    for folder_key, images in folder_dict.items():
        images.sort()  # Sort images to ensure they are processed in order
        for idx, img_file in enumerate(images):
            img_name = os.path.basename(img_file)
            img_name_no_ext = os.path.splitext(img_name)[0]
            parts = img_name_no_ext.split('_')
            if len(parts) < 3:
                continue
            folder_1_name = parts[0]
            folder_2_name = parts[1]
            # Generate new series number
            new_series_num = str(idx).zfill(4)
            # Construct new image name
            new_img_name = f"{folder_1_name}_{folder_2_name}_{new_series_num}.png"
            # Full new path with new image name
            new_img_full_path = os.path.join(dest_directory, new_img_name)
            # Copy and rename the image to the new path
            shutil.copy(img_file, new_img_full_path)
    print(f"Renaming and copying completed. Processed {len(image_files)} images.")
    # After successful processing, delete the source directory
    try:
        shutil.rmtree(src_directory)
        print(f"Source directory {src_directory} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {src_directory}: {e}")

"""
--------------------------------- Renaming annotations -----------------------------------------------------------------
This function rearranges and rebrands text annotation files (.txt) from a source directory, and then duplicates them to a target directory. 
The method initially searches the source directory to locate all text files and categorises them according to the first two segments of their filenames, 
which correspond to folder names. During the second iteration, the program systematically renames each set of files by assigning them sequential numbers
 with leading zeros. This guarantees a uniform and organised naming scheme. The files that have been given new names are subsequently 
 duplicated to the designated directory. Upon successful completion of the processing, the source directory is removed in order to clean up. 
 This function is beneficial for arranging annotation files in a methodical manner to guarantee consistent naming and structure.
"""
def rename_annotations(src_directory, dest_directory):
    def get_text_files(src_path):
        text_files = []
        for root, _, files in os.walk(src_path):
            for file in files:
                if file.endswith('.txt'):
                    text_files.append(os.path.join(root, file))
        return text_files
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    # Get all text files from the source path
    text_files = get_text_files(src_directory)
    # Dictionary to keep track of processed folders and their series
    folder_dict = {}
    # First pass to identify unique folder combinations and their text files
    for txt_file in text_files:
        # Extract filename without extension
        txt_name = os.path.basename(txt_file)
        txt_name_no_ext = os.path.splitext(txt_name)[0]
        # Split filename by underscores to get folder_1_name and folder_2_name
        parts = txt_name_no_ext.split('_')
        if len(parts) < 2:
            print(f"Skipping file {txt_file} as it does not follow the expected naming convention.")
            continue
        folder_1_name = parts[0]
        folder_2_name = parts[1]
        # Generate a unique key for folder_1_name and folder_2_name
        folder_key = f"{folder_1_name}_{folder_2_name}"
        # Initialize the list of text files if this folder_key is encountered for the first time
        if folder_key not in folder_dict:
            folder_dict[folder_key] = []
        # Append the text file to the corresponding folder key
        folder_dict[folder_key].append(txt_file)
    # Second pass to rename and copy text files in sequence
    for folder_key, files in folder_dict.items():
        files.sort()  # Sort files to ensure they are processed in order
        for idx, txt_file in enumerate(files):
            txt_name = os.path.basename(txt_file)
            txt_name_no_ext = os.path.splitext(txt_name)[0]
            parts = txt_name_no_ext.split('_')
            if len(parts) < 3:
                continue
            folder_1_name = parts[0]
            folder_2_name = parts[1]
            # Generate new series number
            new_series_num = str(idx).zfill(4)
            # Construct new text file name
            new_txt_name = f"{folder_1_name}_{folder_2_name}_{new_series_num}.txt"
            # Full new path with new text file name
            new_txt_full_path = os.path.join(dest_directory, new_txt_name)
            # Copy and rename the text file to the new path
            shutil.copy(txt_file, new_txt_full_path)
    print(f"Renaming and copying completed. Processed {len(text_files)} files.")
    # After successful processing, delete the source directory
    try:
        shutil.rmtree(src_directory)
        print(f"Source directory {src_directory} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {src_directory}: {e}")

"""
------------------------------ Consedring only matched train images with the masked images only -------------------------
This function arranges pics files among three folders. The function initially retrieves the names of picture files from `folder_1` and verifies 
their presence in `folder_2`. The method will duplicate each corresponding image discovered in `folder_2` and transfer it to `folder_3`. 
A notice is produced if a pics is not present in `folder_2` that exists in `folder_1`. Upon finishing the duplication procedure, the function 
proceeds to remove both `folder_1` and `folder_2` in order to clean up. This function is beneficial for merging matching photos into a unified 
directory while eliminating the original directories to optimise the dataset.
"""
def consider_matched_train_images_with_masks(folder_1, folder_2, folder_3):
    def get_image_filenames(folder_path):
        image_files = []
        for file in os.listdir(folder_path):
            if file.endswith('.png'):  # Adjust file extension as per your images
                image_files.append(file)
        return image_files
    # Get image filenames from folder_1
    folder_1_images = get_image_filenames(folder_1)
    # Ensure folder_3 exists
    if not os.path.exists(folder_3):
        os.makedirs(folder_3)
    # Iterate over each image in folder_1 and check if it exists in folder_2
    for image_name in folder_1_images:
        src_path = os.path.join(folder_2, image_name)
        dst_path = os.path.join(folder_3, image_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
            # print(f"Copied {image_name} from {folder_2} to {folder_3}.")
        else:
            print(f"{image_name} does not exist in {folder_2}.")
    # After copying, delete folder_2
    try:
        shutil.rmtree(folder_2)
        print(f"Source directory {folder_2} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {folder_2}: {e}")
    # Delete folder_1
    try:
        shutil.rmtree(folder_1)
        print(f"Source directory {folder_1} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {folder_1}: {e}")

"""
------------------------------------Dividing data into train and valid-----------------------------------------------------
This function partitions and separates images and label data into training and validation sets according to a given split ratio. 
The function initially generates the essential folders for storing the training and validation images and labels. 
The method thereafter compiles a comprehensive inventory of all image files in the source directory, randomises their order, and 
divides them based on the specified ratio (e.g., 90% for training and 10% for validation). 
The function duplicates the images and their related label files to the appropriate training and validation folders. 
Upon completion of the copying procedure, it removes the original source folders in order to clean up. 
"""
def data_split_train_valid(source_img_dir, source_label_dir, dataset_dir, split_ratio):
    # Create directories for train and validation splits
    train_img_dir = os.path.join(dataset_dir, 'train', 'images')
    val_img_dir = os.path.join(dataset_dir, 'valid', 'images')
    train_label_dir = os.path.join(dataset_dir, 'train', 'labels')
    val_label_dir = os.path.join(dataset_dir, 'valid', 'labels')
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    # List all image files
    all_images = [f for f in os.listdir(source_img_dir) if f.endswith('.png')]
    # Shuffle and split data
    random.shuffle(all_images)
    split_index = int(split_ratio * len(all_images))  # 90% for training, 10% for validation
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]
    # Function to copy images and corresponding labels with original names
    def copy_images_and_labels(file_list, src_img_dir, src_label_dir, dst_img_dir, dst_label_dir):
        for file_name in file_list:
            base_name = os.path.splitext(file_name)[0]
            img_src = os.path.join(src_img_dir, file_name)
            label_src = os.path.join(src_label_dir, f'{base_name}.txt')
            img_dst = os.path.join(dst_img_dir, file_name)
            label_dst = os.path.join(dst_label_dir, f'{base_name}.txt')
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dst)
    # Copy training images and labels
    copy_images_and_labels(train_images, source_img_dir, source_label_dir, train_img_dir, train_label_dir)
    # Copy validation images and labels
    copy_images_and_labels(val_images, source_img_dir, source_label_dir, val_img_dir, val_label_dir)
    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")
    # Delete source directories after copying
    try:
        shutil.rmtree(source_img_dir)
        print(f"Source directory {source_img_dir} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {source_img_dir}: {e}")
    try:
        shutil.rmtree(source_label_dir)
        print(f"Source directory {source_label_dir} deleted successfully.")
    except Exception as e:
        print(f"Error deleting source directory {source_label_dir}: {e}")
    
"""
Generate test images -> .nii to .png  
This function creates two dictionaries that are associated with label encoding for datasets in the field of medical imaging. 
The `combined_labels` dictionary associates combined label names (e.g., 'bowel_healthy' and 'liver_high') with integer IDs. 
These identifiers are utilised for categorising and manipulating labels in the dataset. The `reverse_combined_labels` dictionary 
serves as a reverse mapping, allowing the conversion of integer identifiers back to their corresponding label names.
This configuration is beneficial for both encoding labels for the purpose of model training and decoding predictions for 
the purpose of interpretation. The function produces a tuple containing both dictionaries.
"""
def create_combined_labels():
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
    reverse_combined_labels = {v: k for k, v in combined_labels.items()}
    return combined_labels, reverse_combined_labels

"""
The method guarantees that each class is assigned a unique colour to facilitate separation and visualisation in applications 
such as medical picture segmentation or object recognition.
"""
def create_color_dict(reverse_combined_labels):
    colors = [
    (255, 182, 193), # Light Pink
    (144, 238, 144), # Light Green
    (173, 216, 230), # Light Blue
    (255, 255, 224), # Light Yellow
    (255, 182, 255), # Light Magenta
    (224, 255, 255), # Light Cyan
    (255, 228, 196), # Bisque
    (250, 235, 215), # Antique White
    (240, 230, 140), # Khaki
    (255, 240, 245), # Lavender Blush
    (230, 230, 250)  # Lavender
                              ]
    color_dict = {label: colors[i % len(colors)] for i, label in reverse_combined_labels.items()}
    return color_dict

"""
The function overlays bounding boxes, labels, and confidence scores on an image. 
It visualizes object detections by drawing rectangles around detected objects and annotating them with class labels and confidence scores. 
The bounding boxes and labels use colors corresponding to different object classes, making it easy to distinguish between them. 
This function is useful for visualizing the results of object detection algorithms in a clear and informative manner.
"""
def draw_bounding_boxes(img, boxes, names, color_dict):
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = names[int(cls_id)]
        score = conf
        label_text = f'{label} {score:.2f}'
        font_scale = 0.25  # Reduce the font size
        font_thickness = 1  # Reduce the font thickness
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        color = color_dict[label]  # Get the color for this class
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, thickness=cv2.FILLED)
        cv2.putText(img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

"""
This method turns a list of detection results into a pandas DataFrame, with each class being represented as an own column. 
If a class is not identified in an image, its confidence score is assigned a value of 0. 
The DataFrame structure facilitates the study of detection results, where each row corresponds to an image and the columns 
indicate confidence ratings for each class.
"""  
def transform_results(results_list, combined_labels):
    # Initialize a dictionary to store the transformed results
    transformed_results = {}
    # Iterate through the results list and populate the dictionary
    for result in results_list:
        filename = result['filename']
        patient_id = filename.split('_')[0]
        class_name = result['class']
        confidence = float(result['confidence'].detach().cpu().numpy())  # Clean the confidence value
        # If the filename is not already in the dictionary, add it with all classes set to 0
        if filename not in transformed_results:
            transformed_results[filename] = {'patient_id': patient_id, **{cls: 0.0 for cls in combined_labels.keys()}}
        # Set the confidence for the detected class
        transformed_results[filename][class_name] = confidence
    # Convert the dictionary to a DataFrame
    transformed_df = pd.DataFrame.from_dict(transformed_results, orient='index').reset_index()
    transformed_df.rename(columns={'index': 'filename'}, inplace=True)
    return transformed_df

"""
This function computes the mean confidence scores for each class per patient given a DataFrame of detection results. 
The algorithm initially retrieves the patient ID from the filenames and subsequently organises the data based on patient ID,
calculating the average confidence ratings for each class. The resultant DataFrame contains the mean scores for each class per patient, 
rounded to two decimal places for enhanced clarity.
"""
def compute_patient_averages(transformed_df):
    # Ensure 'patient_id' is a column
    transformed_df['patient_id'] = transformed_df['filename'].apply(lambda x: x.split('_')[0])
    # Select only numeric columns for averaging
    numeric_columns = transformed_df.select_dtypes(include=[np.number]).columns
    # Group by patient_id and compute the mean for each class
    patient_averages_df = transformed_df.groupby('patient_id')[numeric_columns].mean().reset_index()
    # Round the values to 2 decimal places
    patient_averages_df = patient_averages_df.round(2)
    return patient_averages_df
