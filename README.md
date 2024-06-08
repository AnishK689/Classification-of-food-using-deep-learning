# Classification of Food Using Deep Learning

This project aims to classify different categories of food using deep learning techniques. It consists of data preprocessing, model training, evaluation, and a graphical user interface (GUI) for image prediction. The GUI allows users to either upload an image or capture one using a camera to predict its food category.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Model Training](#model-training)
  - [Evaluation](#evaluation)
  - [Prediction](#prediction)
  - [GUI](#gui)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview

This project involves the following steps:
1. Preprocessing the food images.
2. Training a deep learning model on the preprocessed data.
3. Evaluating the model performance using accuracy and loss curves, classification report, and heatmaps.
4. Predicting the category of a single food image.
5. Implementing a GUI using Tkinter to upload or capture images and predict their categories.

## Requirements

- Python 3.x
- Jupyter Notebook
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV
- PIL
- Tkinter

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/food-classification.git
   cd food-classification
2. Launch the jupyter notebook

   
## Dataset

The dataset used for this project can be found [here](https://www.kaggle.com/datasets/anishhhhhhhh18/food-101-20-class). Download and extract the dataset into the `data` directory within the project folder.

## Usage

### Preprocessing

Run the preprocessing steps in the Jupyter Notebook to prepare the data for training. This includes resizing images, normalizing pixel values, and splitting the dataset into training and validation sets.

### Model Training

Train the deep learning model using the preprocessed data. The model architecture and training process are defined in the notebook.

### Evaluation

Evaluate the model's performance by visualizing accuracy and loss curves, generating a classification report, and creating heatmaps to show model predictions.

### Prediction

Predict the category of a single food image. This can be done using an image from the dataset, an online image, or an image captured using a camera.

### GUI

Use the Tkinter-based GUI to upload or capture an image and predict its category. The GUI provides buttons for selecting an image from the file system or capturing one using the camera.

## Results

The results of the model training and evaluation, including accuracy, loss curves, classification report, and heatmaps, are displayed in the Jupyter Notebook. The GUI allows for real-time prediction of food categories.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.
