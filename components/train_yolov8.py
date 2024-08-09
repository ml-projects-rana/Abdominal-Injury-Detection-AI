from ultralytics import YOLO
import os
import pandas as pd
import torch
import yaml
import torch


def yolo_model(results_dir, yaml_file, model_name,epoch,image_size,batch_size):
    try:
        # Check PyTorch version
        print(f"PyTorch Version: {torch.__version__}")

        # Check for GPU availability
        gpu_available = torch.cuda.is_available()
        print(f"Is CUDA available? {gpu_available}")
        if gpu_available:
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        # Define the path for the Results directory
        os.makedirs(results_dir, exist_ok=True)
        print(f"Directory '{results_dir}' is created or already exists.")

        # Proceed with your YOLOv8 implementation if the YAML file exists
        if os.path.exists(yaml_file):
            print(f"'{yaml_file}' exists.")

            # Load the model configuration
            model = YOLO(model_name)  # Using a pre-trained model weights file

            # Train the model using the .yaml file
            results = model.train(data=yaml_file, epochs=epoch, imgsz=image_size, batch=batch_size)
            print("Training completed.")

            # Print results to check the training output
            print(f"Training Results: {results}")

            # Verify where weights are saved
            if hasattr(results, 'save_dir'):
                save_dir = results.save_dir  # This attribute should contain the path
                print(f"Model weights should be saved in: {save_dir}")
                
                # List the contents of the save directory
                for root, dirs, files in os.walk(save_dir):
                    print(f"Root: {root}")
                    for file in files:
                        print(f"File: {os.path.join(root, file)}")
            else:
                print("No save directory information available in results.")
            
            # Load the trained model for validation
            model_val = YOLO(os.path.join(save_dir, 'weights', 'best.pt'))  # Adjust path to your trained model
            
            # Perform validation and save results
            val_results = model_val.val(data=yaml_file, save=True)
            print("Validation completed.")

            # Print validation results
            print(f"Validation Results: {val_results}")
            
            # Optionally, list the contents of the validation results directory
            if hasattr(val_results, 'save_dir'):
                val_save_dir = val_results.save_dir
                print(f"Validation results saved in: {val_save_dir}")
                
                # List the contents of the validation results directory
                for root, dirs, files in os.walk(val_save_dir):
                    print(f"Root: {root}")
                    for file in files:
                        print(f"File: {os.path.join(root, file)}")
            else:
                print("No save directory information available in validation results.")
        else:
            print(f"'{yaml_file}' does not exist. Please check the path.")
    
    except Exception as e:
        print(f"An error occurred: {e}")




# def yolo_model(results_dir,yaml_file,model_name):
#     try:
#         # Check PyTorch version
#         logging.info("PyTorch Version: %s", torch.__version__)

#         # Check for GPU availability
#         gpu_available = torch.cuda.is_available()
#         logging.info("Is CUDA available? %s", gpu_available)
#         if gpu_available:
#             logging.info("CUDA version: %s", torch.version.cuda)
#             logging.info("Number of GPUs: %d", torch.cuda.device_count())
#             for i in range(torch.cuda.device_count()):
#                 logging.info("GPU %d: %s", i, torch.cuda.get_device_name(i))

#         # Define the path for the Results directory
#         # results_dir = "../Outputs/Preprocessing/Results"
#         os.makedirs(results_dir, exist_ok=True)
#         logging.info("Directory '%s' is created or already exists.", results_dir)

#         # Specify the correct path to the .yaml files
#         # yaml_file = "../Outputs/Preprocessing/datasets/data/custom_data.yaml"

#         # Proceed with your YOLOv8 implementation if the YAML file exists
#         if os.path.exists(yaml_file):
#             logging.info("'%s' exists.", yaml_file)

#             # Load the model configuration
#             model = YOLO(model_name)  # Using a pre-trained model weights file

#             # Train the model using the .yaml file
#             results = model.train(data=yaml_file, epochs=2, imgsz=512, batch=16)
#             logging.info("Training completed.")

#             # Print results to check the training output
#             logging.info("Training Results: %s", results)

#             # Verify where weights are saved
#             if hasattr(results, 'save_dir'):
#                 save_dir = results.save_dir  # This attribute should contain the path
#                 logging.info("Model weights should be saved in: %s", save_dir)
                
#                 # List the contents of the save directory
#                 for root, dirs, files in os.walk(save_dir):
#                     logging.info("Root: %s", root)
#                     for file in files:
#                         logging.info("File: %s", os.path.join(root, file))
#             else:
#                 logging.warning("No save directory information available in results.")
            
#             # Load the trained model for validation
#             model_val = YOLO(os.path.join(save_dir, 'weights', 'best.pt'))  # Adjust path to your trained model
            
#             # Perform validation and save results
#             val_results = model_val.val(data=yaml_file, save=True)
#             logging.info("Validation completed.")

#             # Print validation results
#             logging.info("Validation Results: %s", val_results)
            
#             # Optionally, list the contents of the validation results directory
#             if hasattr(val_results, 'save_dir'):
#                 val_save_dir = val_results.save_dir
#                 logging.info("Validation results saved in: %s", val_save_dir)
                
#                 # List the contents of the validation results directory
#                 for root, dirs, files in os.walk(val_save_dir):
#                     logging.info("Root: %s", root)
#                     for file in files:
#                         logging.info("File: %s", os.path.join(root, file))
#             else:
#                 logging.warning("No save directory information available in validation results.")
#         else:
#             logging.error("'%s' does not exist. Please check the path.", yaml_file)
    
#     except Exception as e:
#         logging.exception("An error occurred: %s", e)

# if __name__ == '__yolo_model__':
#     yolo_model()
