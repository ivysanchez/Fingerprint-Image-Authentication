# Fingerprint-Image-Authentication Project

![](UTA-DataScience-Logo.png)


This repository presents a computer vision project that aims to classify fingerprint images as either real or altered using the SOCOFing dataset. This project uses supervised deep learning methods for image classification and applies transfer learning, data augmentation, and model evaluation strategies to enhance performance. The SOCOFing dataset (Sokoto Coventry Fingerprint Dataset) contains over 6,000 fingerprint images from 600 African subjects, with both real and synthetically altered fingerprints across three levels of distortion (easy, medium, and hard). Dataset: (https://www.kaggle.com/datasets/ruizgara/socofing).

## Overview

The goal of this project is to detect altered fingerprints in image data. The task is framed as a binary classification problem with two classes: 'Real' and 'Altered'. The pipeline includes data cleaning, creating a subset of the data, data loading, data augmentation, model training, model testing, and performance evaluation on a validation set. The dataset was split into a training/test/validation set and three transfer learning models were used for model training (ResNet50, MobileNetV2, and EfficientNetB0). The models were used as baselines and also evaluated with data augmentation to see if it would improve performance. The ResNet50 model performed the best across all metrics, achieving an AUC of 0.9125 without augmentation and 0.9250 with augmentation. 

## Summary of Work Done



### Data
* **Dataset**: SOCOFing
* **Type**: Image files (grayscale .bmp)
  * **Input**: Directories containing fingerprint images categorized into 'Real' and 'Altered'
  * **Output**: Classified fingerprint images as either 'Real' or 'Altered'.
* **Size**: The original SOCOFing dataset contains 6000 real images and 49270 altered images across different difficulty levels.
    * Classes:
      
        * Real (unaltered)
      
        * Altered (easy, medium, hard)

* **Instances (Train, Test, Validation Split)**: A smaller subset of the dataset was created for this project, with 100 images per class ('Real' and 'Altered') for both training and testing (80/20 split). This resulted in 80 training images per class and 20 testing images per class.


***

#### Preprocessing / Clean Up

* Checked for and removed any corrupted images using the Pillow library. No corrupted images were found in the subset used.
* Resizing all images to 224×224 pixels
* Normalization image pixels
* A smaller subset of the SOCOFing dataset was created, containing 100 images per class ('Real' and 'Altered') for both training and testing. This resulted in 80 training images per class and 20 testing images per class.
* Dataset loaded using image_dataset_from_directory with batch prefetching to create TensorFlow Dataset objects for training and validation
* A couple of images used

Binary Labeled:
<img width="793" height="812" alt="Unknown-17" src="https://github.com/user-attachments/assets/27ae02c7-cc4f-41fb-9e7d-c628a2a95d96" />


Non-Binary Labeled:
<img width="966" height="990" alt="Unknown-18" src="https://github.com/user-attachments/assets/2869676e-7a2c-4fcf-bbd2-02a90e954c48" />




### Data Augmentation & Visualization

* Applied data augmentation layers (RandomFlip, RandomRotation, RandomZoom) to the training dataset to increase its variability and potentially improve model generalization.
* Images were loaded to verify success.
<img width="793" height="790" alt="Unknown-11" src="https://github.com/user-attachments/assets/1280fecc-fdad-46d0-a202-d779709ccb74" />


### Problem Formulation

* **Input**: 224×224 RGB fingerprint image
* **Output**: Class label (0 = 'Real', 1 = 'Altered')
* **Task**: Binary image classification







### Model Training 


* Architecture: Transfer learning was employed using pre-trained models on ImageNet:
    * ResNet50
    * MobileNetV2
    * EfficientNetB0
* The base model layers were frozen, and new classification layers were added on top.
* Batch Size: 10
* Models were compiled with the Adam optimizer, Binary Crossentropy loss, and Binary Accuracy metric.
* Models were trained for 10 epochs.
* Metrics: Accuracy, ROC-AUC





***
### Conclusions

Model Comparison (No Augmentation)
* The performance of the ResNet50, MobileNetV2, and EfficientNetB0 models without data augmentation was compared using ROC curves and AUC scores on the validation set.
* **Results**:
  * ResNet50 AUC: 0.8875
  * MobileNetV2 AUC: 0.8125
  * EfficientNetB0 AUC: 0.8175
*ResNet50 showed the best performance in this comparison.
<img width="846" height="855" alt="Unknown-15" src="https://github.com/user-attachments/assets/0cdd9bf7-76fb-4bf6-a6b1-72835381dbe0" />

ResNet50 (Baseline) training/validation:
<img width="1189" height="590" alt="Unknown-24" src="https://github.com/user-attachments/assets/881296c5-fcc8-46b0-bd74-d9b4e40123fe" />


Model Comparison (With Augmentation - ResNet50)
* The performance of the ResNet50 model with and without data augmentation was compared using ROC curves and AUC scores on the validation set.
* **Results**:
  * ResNet50 (Baseline) AUC: 0.9125
  * ResNet50 (Augmented) AUC: 0.9250
* Data augmentation slightly improved the performance of the ResNet50 model.
<img width="691" height="701" alt="Unknown-16" src="https://github.com/user-attachments/assets/f548688f-9161-4bef-9f2e-ffa2c8343dca" />


### Future Work
* Compare the augmented versions of MobileNetV2 and EfficientNetB0 with their non-augmented counterparts and the augmented ResNet50 model.
* Fine-tune the top layers of the best-performing transfer learning models with a lower learning rate to potentially improve performance further.
* Evaluate the best-performing model on the full dataset to assess its scalability and robustness.


## How to reproduce results

To reproduce the results of this project, follow these steps:

1. Download the dataset: Download the "SOCOFing" dataset from Kaggle. The notebook contains code to unzip the dataset.
2. Open notebooks in the following order:
    - `DataLoader.ipynb`
    - `TrainBaseModel.ipynb`
    - `TrainBaseModelAugmentation.ipynb`
    - `CompareAugmentation.ipynb`
    - `Train-MobileNetV2.ipynb`
    - `Train-EfficientNetB0.ipynb`
    - `CompareModels.ipynb`

3. Run the code cells: Execute the code cells in the notebook sequentially to reproduce the results.
   
**Resources:**
* Google Colab: Use Google Colab or Jupyter Notebook to run the code and leverage its computational resources.
* Kaggle: Access the dataset and potentially explore other related datasets.

### Overview of files in repository

| File Name                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `DataLoader.ipynb`               | Loads and prepares the dataset                                              |
| `TrainBaseModel.ipynb`           | Baseline training with ResNet50                                             |
| `TrainBaseModelAugmentation.ipynb` | Training ResNet50 with data augmentation                                 |
| `CompareAugmentation.ipynb`      | ROC curve comparison (augmented vs. baseline)                              |
| `Train-MobileNetV2.ipynb`        | Transfer learning using MobileNetV2                                        |
| `Train-EfficientNetB0.ipynb`     | Transfer learning using EfficientNetB0                                     |
| `CompareModels.ipynb`            | Loads 3 trained models (no augmentation) and compares results
| `FIA-fullcode.ipynb`             | Full pipeline from data loading to evaluation                              |



### Software Setup
* Required Packages: This project uses the following Python packages:
  * Standard Libraries:
   * keras
   * tendorflow
   * numpy
   * zipfile
   * shutil
   * OS
   * matplotlib
   * PIL (Pillow)
* Additional Libraries:
   * kagglehub (For downloading the dataset from Kaggle)






## **Citations**
* Gara Ruiz, A. (2018). SOCOFing Dataset. Kaggle. https://www.kaggle.com/datasets/ruizgara/socofing
