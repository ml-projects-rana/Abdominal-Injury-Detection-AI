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


# A. Injury Detection, Organ Localization, and Injury Severity Classification


## Pipeline:

![model_image_3 drawio](https://github.com/user-attachments/assets/ff4b6f35-ae38-4427-8bb5-1e82c7435351)


## Configuration:

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

## Results
![10937_12039_0342](https://github.com/user-attachments/assets/73380a0c-6230-472a-a63c-e3e3d21ce580) ![16327_63843_0192](https://github.com/user-attachments/assets/8dce4525-209f-42f1-8a20-db32ad1632d2)
![32425_26840_0326](https://github.com/user-attachments/assets/6cdbb9ca-3355-4684-842f-eb768963b28e)
![37429_15748_0440](https://github.com/user-attachments/assets/12379432-08e5-4530-9722-2c616998f5a5)

## B. Validation of Annotations Against Ground Truth

### B.1. Ground Truth Masks predicted using Total Segmentor Model 
### Configuration:
### 1. bowel_data.yaml
train: Path to the directory containing bowel training images.

val: Path to the directory containing bowel validation images.

nc: Number of classes in the dataset, set to 2.

names: List of class names representing different conditions of the bowel, specifically 'bowel_healthy' and 'bowel_injury'.

### 2. kidney_data.yaml
train: Path to the directory containing kidney training images.

val: Path to the directory containing kidney validation images.

nc: Number of classes in the dataset, set to 2.

names: List of class names representing different conditions of the kidney, specifically 'kidney_healthy', 'kidney_low', 'kidney_high'.

### 3. liver_data.yaml
train: Path to the directory containing liver training images.

val: Path to the directory containing liver validation images.

nc: Number of classes in the dataset, set to 2.

names: List of class names representing different conditions of the liver, specifically 'liver_healthy', 'liver_low', 'liver_high'.

### 4. spleen_data.yaml
train: Path to the directory containing spleen training images.

val: Path to the directory containing spleen validation images.

nc: Number of classes in the dataset, set to 2.

names: List of class names representing different conditions of the spleen, specifically 'spleen_healthy', 'spleen_low', 'spleen_high'.



