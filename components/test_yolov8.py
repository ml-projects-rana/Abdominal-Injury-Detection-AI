import os
import cv2
from ultralytics import YOLO  
from components.modules import *

"""
The `test_model` function is executing the following tasks:
The YOLO object identification model is loaded using the given model weights file.
Generates essential folders for storing output photos and verifies their existence.
Performs inference on test photos found in the designated directory, utilising the YOLO model with defined confidence and IOU thresholds.
Applies bounding boxes to the images to highlight discovered items. Each class is assigned a distinct colour, and the annotated images are saved.
The system gathers probability ratings for every identified object and stores this information in a CSV file. 
Subsequently, it calculates the mean confidence ratings for each class per patient and stores this data in a separate CSV file.
The outcomes are stored in the designated folders and files.

"""
def test_model(model_path, test_image_path, output_directory, conf, iou, output_csv_path, output_excel_path):
    # Create combined labels and reverse mapping
    combined_labels, reverse_combined_labels = create_combined_labels()
    # Create color dictionary
    color_dict = create_color_dict(reverse_combined_labels)
    # Load the model
    model = YOLO(model_path)
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    # List to store the results
    results_list = []
    # Perform inference
    results = model.predict(source=test_image_path, conf=conf, iou=iou)
    # Iterate over the results and corresponding original image paths
    for result in results:
        # Get the original image path
        original_image_path = result.path
        # Extract filename from the original path
        filename = os.path.basename(original_image_path)
        # Load the original image
        img = cv2.imread(original_image_path)
        # Draw bounding boxes with reduced label size and different colors for each class
        draw_bounding_boxes(img, result.boxes.data, reverse_combined_labels, color_dict)
        # Save the probability scores to the list
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box
            class_name = reverse_combined_labels[int(class_id)]
            results_list.append({'filename': filename, 'class': class_name, 'confidence': score})
        # Define the output file path
        output_file_path = os.path.join(output_directory, filename)
        # Save the image with predictions
        cv2.imwrite(output_file_path, img)
    # Transform results and save to CSV
    transformed_df = transform_results(results_list, combined_labels)
    transformed_df.to_csv(output_csv_path, index=False)
     # Compute patient averages and save to CSV
    patient_averages_df = compute_patient_averages(transformed_df)
    patient_averages_df.to_csv(output_excel_path, index=False)

