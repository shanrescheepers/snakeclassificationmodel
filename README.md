
![Accuracy Evidence](https://github.com/shanrescheepers/snakeclassificationmodel/blob/master/Images/prediction_accuracy_metrics.png)


# Venomous & Nonvenomous Snake AI Image Classifier
## Overview
This Python-based AI image classification model has been trained to distinguish between venomous and nonvenomous snakes. The model was created using Roboflow for data preprocessing and Google Colab for training for those who cannot afford a GPU or utilising a heavy duty device. The dataset used for training primarily includes images of Southern African local snake species.
This model was inspired by the challenges of differentiating between venomous & nonvenomous snakes in my local wildlife area.

### Features
- Classifies snake images into two categories: "Venomous" and "Nonvenomous."
- Achieves high accuracy in distinguishing between venomous and nonvenomous snake species.

## Usage & Installation:
1) Create account on [Roboflow](https://roboflow.com/) and read the documentation.
2) [Go to my Roboflow Project to access my dataset](https://universe.roboflow.com/shanr-scheepers) .

3) Follow the set of code found in the link.

4) Here is a [Google Colab Session](https://colab.research.google.com/github/shanrescheepers/snakeclassificationmodel/blob/master/RSASnakeClassifications.ipynb) if you're seeking my python implementation.

Use this in your project!
```bash
#To Clone the repository:
git clone https://github.com/shanrescheepers/snakeclassificationmodel.git
cd snakevenom-classifier
```

#### They basically include
- !pip install tensorflow import tensorflow as tf
- !pip install ultralytics from ultralytics import YOLO
- import os from IPython.display import display, Image from IPython import display
- !pip install roboflow from roboflow import Roboflow
Make sure you have tensorflow & python installed on yout device. I am using python3.


### Training
The model has been trained on a dataset of Southern African snake images. The training process involves the following steps:

#### Data Collection: 
Snake images from Southern African snake species were collected and organized into "Venomous" and "Nonvenomous" classname categories.

#### Data Preprocessing: 
[Roboflow](https://roboflow.com/) was used for data preprocessing, which included resizing, seperation into train, test and validation content, data augmentation, and classification.

#### Model Selection: 
YOLO image classification & recognition CNN model was chosen. YOLO which stands for "You Only Look Once," is a popular object detection and real-time image recognition model. YOLO is known for its speed and accuracy, making it widely used in various computer vision applications. YOLO has several versions, each with its own improvements and features.
[Read The Blog Here](https://blog.roboflow.com/train-yolov5-classification-custom-data/)

#### Training: 
The model was trained on Google Colab using the preprocessed dataset to predict the classification of different types of venomous or non venomous snakes given the input image date.

#### Evaluation: 
The model's performance was evaluated using accuracy metrics.

#### Model Accuracy Report
The model exhibits high accuracy of 0.60-0.70 in distinguishing between venomous and nonvenomous snake species, and even high .70's when adding South African snake sp. However, it may not perform as accurately with snake images from regions outside Southern Africa, as the training data is primarily focused on this specific area and because it is a newly trained AI image classification model.
![Accuracy Evidence](https://github.com/shanrescheepers/snakeclassificationmodel/blob/master/Images/prediction_accuracy_metrics.png)

#### Dataset
The dataset used for training is not included in this repository due to its size. If you wish to train the model using a different dataset or region, you can replace the training data with your own snake images.


#### License
This project is licensed under the MIT License. Feel free to modify and use the code according to your needs.

### Acknowledgments
[Roboflow](https://roboflow.com/)  for data preprocessing tools.

[Google Colab](https://colab.google/) for providing  open-source usage & GPU resources for model training.
[Python Documentation](https://www.python.org/) 
### Disclaimer
This model is primarily intended for educational and research purposes. It is not a substitute for professional expertise in identifying and handling snakes. Always exercise caution and seek assistance from experts when dealing with snakes, especially in the case of potential venomous species.





