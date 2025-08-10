<br />
<div align="center">
  <a href="">
    <img src="https://giphy.com/gifs/loop-fish-bored-l1J9L9Y81emaoNMoU">
  </a>

<h1 align="center">Multiclass Fish Image Classification</h1>

   <br><br>
    
   <h2 > Problem Statement </h2>
   <p>
    This project focuses on classifying fish images into multiple categories 
    using deep learning models. The task involves training a CNN from scratch 
    and leveraging transfer learning with pre-trained models to enhance 
    performance. The project also includes saving models for later use and 
    deploying a Streamlit application to predict fish categories from 
    user-uploaded images. 
    
  </p>
</div>


## Steps
**Data Preprocessing and Augmentation**

*  Rescale images to [0, 1] range.
*  Apply data augmentation techniques like rotation, zoom, and flipping to enhance model  robustness.
    
**Model Training**

* Train a CNN model from scratch.
* Experiment with five pre-trained models (e.g., VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
*	Fine-tune the pre-trained models on the fish dataset.
*	Save the trained model (max accuracy model ) in .h5 or .pkl format for future use.

**Model Evaluation** 

*	Compare metrics such as accuracy, precision, recall, F1-score, and confusion matrix across all models.
*	Visualize training history (accuracy and loss) for each model.


**Deployment**

*	Build a Streamlit application to:
*	Allow users to upload fish images.
*	Predict and display the fish category.
*	Provide model confidence scores.

## Demo
<div align="center">
<img src="https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms/blob/master/ReadMe_files/BE%20Project%20demo.gif" alt="logo" width="60%" height="400">
</div> 

## DataSet
Dataset: [Drive-link](https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd)

<div align="left">
    <img src="https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms/blob/master/ReadMe_files/dataset_description.png" alt="dataset_description" width="200" height="400">
  </a>
  </div>

## Key Features and Takeaways 

-   **Deep Learning Models**: Comparison of a custom-built CNN with five pre-trained models (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0) to find the best architecture for the task.
    
-   **Transfer Learning**: Effective use of pre-trained models to achieve high accuracy with a smaller, domain-specific dataset.
    
-   **Data Handling**: Implementation of data preprocessing and augmentation techniques to enhance model robustness.
    
-   **Model Deployment**: Creation of an interactive Streamlit application that allows users to upload images and get real-time fish classification predictions.
    
-   **Comprehensive Evaluation**: A detailed comparison of model performance using metrics like accuracy, precision, recall, F1-score, and a confusion matrix.
    
-   **Skills Developed**: Deep Learning, Python, TensorFlow/Keras, Streamlit, Data Preprocessing, Transfer Learning, Model Evaluation, Visualization, and Model Deployment.

## Training Curves and Confusion Matrix

**Custom CNN**
| ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|
**VGG16**

| ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|
**ResNet50**
| ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|
**MobileNet**
| ![Confusion Matrix](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Confusion Matrix](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|
**InceptionV3**
| ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|
**EfficientNetB0**
| ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Train vs Val Loss](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|

**Confusion Matrix**
| ![Confusion Matrix](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132334.png) | ![Confusion Matrix](https://github.com/AdItYaSiNhG/Brain_Tumor_MRI_ML/blob/main/assests/Screenshot%202025-07-24%20132344.png) |
|--------------------------------------------------|----------------------------------------------------------|

## Technologies used
[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)     ![](https://img.shields.io/badge/Tensorflow/Keras-blue.svg) ![](https://img.shields.io/badge/Streamlit-blue.svg) ![Deep Learning](https://img.shields.io/badge/CNN-Custom-green.svg) ![Deep Learning](https://img.shields.io/badge/VGG-16-green.svg) ![Deep Learning](https://img.shields.io/badge/ResNet-50-green.svg) ![Deep Learning](https://img.shields.io/badge/MobileNet-blue.svg)  ![Deep Learning](https://img.shields.io/badge/Inception-V3-green.svg) ![Deep Learning](https://img.shields.io/badge/Efficient-B0-green.svg) ![Version Control](https://img.shields.io/badge/Github-blue.svg)

## How to Run the Project 

#### 1. Clone the Repository

Bash

```
git clone https://github.com/your_username/Fish_Classifier.git
cd Fish Classification

```

#### 2. Setup the Environment

It is recommended to use a virtual environment.

Bash

```
pip install -r requirements.txt

```

#### 3. Dataset

The dataset is available as a zip file. Download it, extract it, and place it in the

`data/` directory.

#### 4. Model Training and Evaluation

Run the provided Python scripts or Jupyter notebooks to train the models and generate the evaluation report. The best-performing model will be saved for deployment.

#### 5. Launch the Streamlit Application

To run the web app for real-time predictions, use the following command:

Bash

```
streamlit run app.py

```

The application will open in your default web browser.


## Contributing to the project:

*   Fork the [Master Branch](https://github.com/SartajBhuvaji/Brain-Tumor-Classification-Using-Deep-Learning-Algorithms)
*   Clone your branch
*   If you want to contribute by adding your classification model, create a new folder inside ```Contributions/Your_Name```
*   Make sure to add a detailed README File.

</div>
























