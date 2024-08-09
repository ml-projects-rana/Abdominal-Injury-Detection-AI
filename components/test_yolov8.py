import os
import cv2
from ultralytics import YOLO  # Make sure to import YOLO from the correct library
from components.modules import *

def test_model(model_path, test_image_path, output_directory, conf, iou, output_csv_path, output_excel_path):
    """
    Load a YOLO model, perform inference on test images, draw bounding boxes, and save the results.
    Also, save the probability score for each image per class in a CSV file and compute averages.

    Args:
        model_path (str): Path to the YOLO model weights file.
        test_image_path (str): Path to the directory containing test images.
        output_directory (str): Directory where output images will be saved.
        conf (float): Confidence threshold for object detection.
        iou (float): IOU threshold for Non-Maximum Suppression (NMS).
        output_csv_path (str): Path to the output CSV file to save the probability scores.
        output_excel_path (str): Path to the output Excel file to save the average probabilities.
    """
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

# def test_model(model_path, test_image_path, output_directory, conf, iou, output_csv_path):
#     """
#     Load a YOLO model, perform inference on test images, draw bounding boxes, and save the results.
#     Also, save the probability score for each image per class in a CSV file.

#     Args:
#         model_path (str): Path to the YOLO model weights file.
#         test_image_path (str): Path to the directory containing test images.
#         output_directory (str): Directory where output images will be saved.
#         conf (float): Confidence threshold for object detection.
#         iou (float): IOU threshold for Non-Maximum Suppression (NMS).
#         output_csv_path (str): Path to the output CSV file to save the probability scores.
#     """
#     # Create combined labels and reverse mapping
#     combined_labels, reverse_combined_labels = create_combined_labels()

#     # Create color dictionary
#     color_dict = create_color_dict(reverse_combined_labels)

#     # Load the model
#     model = YOLO(model_path)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     # List to store the results
#     results_list = []

#     # Perform inference
#     results = model.predict(source=test_image_path, conf=conf, iou=iou)

#     # Iterate over the results and corresponding original image paths
#     for result in results:
#         # Get the original image path
#         original_image_path = result.path
#         # Extract filename from the original path
#         filename = os.path.basename(original_image_path)

#         # Load the original image
#         img = cv2.imread(original_image_path)

#         # Draw bounding boxes with reduced label size and different colors for each class
#         draw_bounding_boxes(img, result.boxes.data, reverse_combined_labels, color_dict)

#         # Save the probability scores to the list
#         for box in result.boxes.data:
#             x1, y1, x2, y2, score, class_id = box
#             class_name = reverse_combined_labels[int(class_id)]
#             results_list.append({'filename': filename, 'class': class_name, 'confidence': score})

#         # Define the output file path
#         output_file_path = os.path.join(output_directory, filename)

#         # Save the image with predictions
#         cv2.imwrite(output_file_path, img)

#     # Transform results and save to CSV
#     transformed_df = transform_results(results_list, combined_labels)
#     transformed_df.to_csv(output_csv_path, index=False)

# def test_model(model_path, test_image_path, output_directory, conf, iou, output_csv_path):
#     """
#     Load a YOLO model, perform inference on test images, draw bounding boxes, and save the results.
#     Also, save the probability score for each image per class in a CSV file.

#     Args:
#         model_path (str): Path to the YOLO model weights file.
#         test_image_path (str): Path to the directory containing test images.
#         output_directory (str): Directory where output images will be saved.
#         conf (float): Confidence threshold for object detection.
#         iou (float): IOU threshold for Non-Maximum Suppression (NMS).
#         output_csv_path (str): Path to the output CSV file to save the probability scores.
#     """
#     # Create combined labels and reverse mapping
#     combined_labels, reverse_combined_labels = create_combined_labels()

#     # Create color dictionary
#     color_dict = create_color_dict(reverse_combined_labels)

#     # Load the model
#     model = YOLO(model_path)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     # List to store the results
#     results_list = []

#     # Perform inference
#     results = model.predict(source=test_image_path, conf=conf, iou=iou)

#     # Iterate over the results and corresponding original image paths
#     for result in results:
#         # Get the original image path
#         original_image_path = result.path
#         # Extract filename from the original path
#         filename = os.path.basename(original_image_path)

#         # Load the original image
#         img = cv2.imread(original_image_path)

#         # Draw bounding boxes with reduced label size and different colors for each class
#         draw_bounding_boxes(img, result.boxes.data, reverse_combined_labels, color_dict)

#         # Save the probability scores to the list
#         for box in result.boxes.data:
#             x1, y1, x2, y2, score, class_id = box
#             class_name = reverse_combined_labels[int(class_id)]
#             results_list.append({'filename': filename, 'class': class_name, 'confidence': score})

#         # Define the output file path
#         output_file_path = os.path.join(output_directory, filename)

#         # Save the image with predictions
#         cv2.imwrite(output_file_path, img)

#     # Convert the results list to a DataFrame and save to a CSV file
#     results_df = pd.DataFrame(results_list)
#     results_df.to_csv(output_csv_path, index=False)



# def test_model(model_path, test_image_path, output_directory, conf, iou):
#     """
#     Load a YOLO model, perform inference on test images, draw bounding boxes, and save the results.

#     Args:
#         model_path (str): Path to the YOLO model weights file.
#         test_image_path (str): Path to the directory containing test images.
#         output_directory (str): Directory where output images will be saved.
#         conf (float): Confidence threshold for object detection.
#         iou (float): IOU threshold for Non-Maximum Suppression (NMS).
#     """
#     # Load the model
#     model = YOLO(model_path)

#     # Create the output directory if it doesn't exist
#     os.makedirs(output_directory, exist_ok=True)

#     # Perform inference
#     results = model.predict(source=test_image_path, conf=conf, iou=iou)

    
#     # Iterate over the results and corresponding original image paths
#     for result in results:
#         # Get the original image path
#         original_image_path = result.path
#         # Extract filename from the original path
#         filename = os.path.basename(original_image_path)

#         # Load the original image
#         img = cv2.imread(original_image_path)

#         # Draw bounding boxes with reduced label size
#         draw_bounding_boxes(img, result.boxes.data, result.names)

#         # Define the output file path
#         output_file_path = os.path.join(output_directory, filename)

#         # Save the image with predictions
#         cv2.imwrite(output_file_path, img)





























# from ultralytics import YOLO
# import os
# import cv2


# # Load the model
# model = YOLO('../Outputs/Preprocessing/Results/run/detect/train/weights/best.pt')

# # Path to the test images
# test_image_path = '../Outputs/Preprocessing/test_images_png/'

# # Path to save the results
# output_directory = '../Outputs/Preprocessing/predict/'
# os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# # Perform inference
# results = model.predict(source=test_image_path, conf=0.1, iou=0.1)

# # Function to draw bounding boxes with reduced label size
# def draw_bounding_boxes(img, boxes, names):
#     for box in boxes:
#         x1, y1, x2, y2, conf, cls_id = box
#         x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
#         label = names[int(cls_id)]
#         # print("label",label)
#         score = conf
#         # print("score",score)
#         label_text = f'{label} {score:.2f}'
#         font_scale = 0.25  # Reduce the font size
#         font_thickness = 1  # Reduce the font thickness
#         (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
#         cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
#         cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), thickness=cv2.FILLED)
#         cv2.putText(img, label_text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)

# # Iterate over the results and corresponding original image paths
# for result in results:
#     # Get the original image path
#     original_image_path = result.path
#     # Extract filename from the original path
#     filename = os.path.basename(original_image_path)
    
#     # Load the original image
#     img = cv2.imread(original_image_path)
    
#     # Draw bounding boxes with reduced label size
#     draw_bounding_boxes(img, result.boxes.data, result.names)
    
#     # Define the output file path
#     output_file_path = os.path.join(output_directory, filename)
    
#     # Save the image with predictions
#     cv2.imwrite(output_file_path, img)

# print(f'Results saved in {output_directory}')


# # Load the trained model
# model = YOLO('../Outputs/Preprocessing/Results/run/detect/train/weights/best.pt')  # Adjust path to your trained model

# # Path to the test images
# test_image_path = '../Outputs/Preprocessing/test_images_png/'

# # Path to save the results
# output_directory = '../Outputs/Preprocessing/predict/'
# os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist

# # Perform inference
# results = model.predict(source=test_image_path, conf=0.1, iou=0.1)

# # Iterate over the results and corresponding original image paths
# for result in results:
#     # Get the original image path
#     original_image_path = result.path
#     # Extract filename from the original path
#     filename = os.path.basename(original_image_path)
    
#     # Get the image with predictions
#     img_with_predictions = result.plot()  # This returns the image with predictions
    
#     # Define the output file path
#     output_file_path = os.path.join(output_directory, filename)
    
#     # Save the image with predictions
#     cv2.imwrite(output_file_path, img_with_predictions)

# print(f'Results saved in {output_directory}')


