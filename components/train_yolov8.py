from ultralytics import YOLO
import os
import pandas as pd
import torch
import yaml
import torch

"""
The `yolo_model` function simplifies the process of training and validating a YOLO model. The process starts by verifying the PyTorch version 
and determining the availability of a GPU, including specific facts about CUDA if a GPU is detected. Subsequently, the function verifies the
existence of the results directory before continuing. Assuming the YAML configuration file is correct, it will load a pre-trained YOLO model 
and start training using supplied parameters such as epochs, image size, and batch size. After finishing, it verifies the location where 
the model weights are stored and displays the files included in that location. Subsequently, the function imports the trained model for 
the purpose of validation, executes the validation process, and stores the outcomes, all the while displaying the files present in the
validation results directory. Any mistakes that occur throughout these phases are detected and reported, guaranteeing reliable execution.
"""
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
