# Title:
### AI-Enabled Abdominal Injury Detection, Organ Localization, and Injury Severity Classification from CT Scans and Validation of Annotations Against Ground Truth


# Description:
The objective of this project is to create an artificial intelligence (AI) system that can be used to detect abdominal injuries, localize organs, and classify the severity of injuries by analyzing CT scans. Manually generated annotations in the AI system will undergo a comprehensive validation process against the segmented ground truth data to verify consistency and accuracy of the AI system created.

# Dataset:
https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data

# Installation:
### 1. Clone the repository
#### Using CMD on Windows:

cd C:\path\to\your\directory

git clone https://github.com/ml-projects-rana/Abdominal-Injury-Detection-AI.git

### 2. Navigate to the project directory
cd Abdominal-Injury-Detection-AI

### 3. Create the Environment 
conda create -n myenv python=3.9
conda activate myenv


### 4. Install the required dependencies
pip install -r requirements.txt

# Configuration:
### 1. config.yaml
Configuration to manage paths, directories, and settings related to the training and testing data processing workflows:

train_processing: Manages the paths and parameters for preprocessing the training data, including image transformations, mask generation, and dataset preparation.

test_processing: Specifies the directories and settings for processing the test data, ensuring consistency with the training data pipeline.

log: Defines the location for logging all processing activities, helping track the workflow and debug if necessary.

### 2. data.yaml
Paths and metadata for training and validation datasets, as well as the class structure used in the machine learning model:

train: Path to the directory containing training images.

val: Path to the directory containing validation images.

nc: Number of classes in the dataset, set to 11.

names: List of class names representing different conditions of organs, such as healthy and various injury levels for the bowel, liver, spleen, and kidneys.

### 3. model.yaml
Includes paths for saving results, loading the model and dataset configurations, and setting thresholds for confidence and IoU during detection. It also specifies the model architecture, input size, batch size, and number of training epochs to ensure a consistent and efficient workflow for object detection tasks.





