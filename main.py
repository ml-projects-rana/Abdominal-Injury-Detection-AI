import os
# import logging
import yaml
from components.train_preprocessing import train_preprocessing
from components.test_preprocessing import test_preprocessing
from components.train_yolov8 import yolo_model
from components.test_yolov8 import test_model

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_model_config(model_config_path):
    """Load model YAML configuration file."""
    with open(model_config_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    # Load configurations
    config_path = os.path.join('config', 'config.yaml')
    config = load_config(config_path)

    model_config_path = os.path.join('config', 'model.yaml')
    model_config = load_model_config(model_config_path)

    # Accessing configuration values
    train_processing = config['train_processing']
    test_processing = config['test_processing']
    # log = config['log']
    yolo = model_config['yolo']

    # Extract train preprocessing parameters
    scale_factor= train_processing['scale_factor']
    # data_path=train_processing['train_df']
    # updated_data_path=train_processing['updated_train_path']
    train_series_meta = train_processing['train_series_meta']
    train_df = train_processing['train_df']
    train_merged = train_processing['train_merged']
    segmentation_dir = train_processing['segmentation_dir']
    mask_dir = train_processing['mask_dir']
    annotation_dir = train_processing['annotation_dir']
    train_images_dir = train_processing['train_images_dir']
    train_images_png_dir = train_processing['train_images_png_dir']
    final_train_images_png_dir = train_processing['final_train_images_png_dir']
    final_mask_dir = train_processing['final_mask_dir']
    final_annotation_dir = train_processing['final_annotation_dir']
    matched_train_images_dir = train_processing['matched_train_images_dir']
    dataset_dir = train_processing['dataset_dir']
    split_ratio=train_processing['split_ratio']
    matched_output_path=train_processing['matched_output_path']
    unmatched_output_path=train_processing['unmatched_output_path']
    

    # Extract YOLO model training parameters
    results_dir = yolo['results']
    yaml_file = yolo['yaml_file']
    model_name = yolo['model_name']
    model_path=yolo['model_path']
    output_directory=yolo['output_directory']
    conf= yolo['conf']
    iou= yolo['iou'] 
    epoch=yolo['epoch']
    image_size=yolo['image_size']
    batch_size=yolo['batch_size']
    output_csv_path=yolo['output_csv_path']
    output_excel_path=yolo['output_excel_path']

    # Extract test preprocessing parameters
    test_images_dir = test_processing['test_images_dir']
    final_test_images_png_dir = test_processing['final_test_images_png_dir']
 
    # Ensure the directory exists for logging
    log_file = log['log_file']
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configure logging
    logging.basicConfig(filename=log_file, level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info("\n\nStarting training preprocessing")
    train_preprocessing(scale_factor,split_ratio, train_series_meta, train_df, train_merged, segmentation_dir, mask_dir, annotation_dir, 
                        train_images_dir, train_images_png_dir, final_train_images_png_dir, final_mask_dir, 
                        final_annotation_dir, matched_train_images_dir, dataset_dir,matched_output_path, unmatched_output_path)
    logging.info("\n\nCompleted training preprocessing")

    logging.info("\n\nStarting YOLO model training")
    yolo_model(results_dir, yaml_file, model_name,epoch, image_size, batch_size)
    logging.info("\n\nCompleted YOLO model training")

    logging.info("\n\nStarting test preprocessing")
    test_preprocessing(test_images_dir, final_test_images_png_dir)
    logging.info("\n\nCompleted test preprocessing")

    logging.info("\n\nStarting test model")
    test_model(model_path, final_test_images_png_dir, output_directory, conf, iou, output_csv_path,output_excel_path)
    logging.info("\n\nTest model execution done")
    
    logging.info("\n\nPipeline execution completed")





























# import os
# import logging
# import yaml
# from components.train_preprocessing import *
# from components.test_preprocessing import *
# from components.train_yolov8 import *

# def load_config(config_path):
#     with open(config_path, 'r') as file:
#         return yaml.safe_load(file)
    
# def load_model_config(model_config_path):
#     with open(model_config_path, 'r') as file:
#         return yaml.safe_load(file)


# if __name__ == "__main__":
#     config_path = os.path.join('config', 'config.yaml')
#     config = load_config(config_path)

#     model_config_path = os.path.join('config', 'model.yaml')
#     model_config = load_config(model_config_path)



#     # Accessing configuration values
#     train_processing = config['train_processing']
#     test_processing = config['test_processing']
#     log = config['log']

#     yolo=model_config['yolo']

#     train_series_meta=train_processing['train_series_meta']
#     train_df=train_processing['train_df']
#     train_merged=train_processing['train_merged']
#     segmentation_dir=train_processing['segmentation_dir']
#     mask_dir=train_processing['mask_dir']
#     annotation_dir=train_processing['annotation_dir']
#     train_images_dir=train_processing['train_images_dir']
#     train_images_png_dir=train_processing['train_images_png_dir']
#     final_train_images_png_dir=train_processing['final_train_images_png_dir']
#     final_mask_dir=train_processing['final_mask_dir']
#     final_annotation_dir=train_processing['final_annotation_dir']
#     final_mask_dir=train_processing['final_mask_dir']
#     final_annotation_dir=train_processing['final_annotation_dir']
#     matched_train_images_dir=train_processing['matched_train_images_dir']
#     dataset_dir=train_processing['dataset_dir']

#     results_dir=yolo['results']
#     yaml_file=yolo['yaml_file']
#     model_name=yolo['model_name']

#     test_images_dir=test_processing['test_images_dir']
#     final_train_images_png_dir=test_processing['final_train_images_png_dir']

#     train_preprocessing(train_series_meta, train_df, train_merged, segmentation_dir, mask_dir, annotation_dir, 
#                         train_images_dir, train_images_png_dir, final_train_images_png_dir, final_mask_dir, 
#                         final_annotation_dir, matched_train_images_dir, dataset_dir)
    
#     yolo_model(results_dir, yaml_file, model_name)
#     test_preprocessing(test_images_dir, final_train_images_png_dir)
        
    

    
#     # print(f"test_images_dir: {test_images_dir}")
#     # print(f"final_train_images_png_dir: {final_train_images_png_dir}")
#     # # print(f"model_name: {model_name}")


# # # Ensure the directory exists
# # log_dir = os.path.dirname(log_file)
# # if not os.path.exists(log_dir):
# #     os.makedirs(log_dir)

# # logging.basicConfig(filename=log_file, level=logging.DEBUG, 
# #                     format='%(asctime)s - %(levelname)s - %(message)s')