# FireLite: Enhanced Fire Detection through Transfer Learning

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Dataset Description](#dataset-description)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Results](#results)
9. [Contact](#contact)

## Project Overview
FireLite is a deep learning model designed to detect fire in images using advanced convolutional neural networks. The model uses a combination of data augmentation techniques and a pre-trained architecture to classify images as fire or no fire, aiming to achieve high accuracy and robustness across various image conditions.

## Installation
To install and run this project locally, follow the steps below:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FireNetV3.0.git
   cd FireNetV3.0
2. Install the necessary libraries: You can install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt

## Usage
To run the model, you can use the provided Jupyter notebook (FireNetV3_0.ipynb). Here's how to use it:
1. Open the notebook
   ```bash
   jupyter notebook FireNetV3_0.ipynb
2. Follow the steps in the notebook to:
   * Load the dataset
   * Preprocess the data
   * Train the model
   * Evaluate the results
3. You can modify parameters like batch size, number of epochs, or optimizer in the notebook to suit your needs.

## Dataset Description
* <b> The dataset is proprietary; therefore, it is prohibited from being publicly published to avoid potential legal complications. </b>
* The dataset contains images of fire and non-fire scenarios.
* Ensure that the dataset is correctly split into training and testing sets.
* Data preprocessing involves rescaling the images and applying data augmentation techniques like rotations, flipping, and zooming.

## Model Architecture
FireLite is based on a convolutional neural network (CNN) with several key components:
* Pretrained Backbone: The model leverages a pretrained architecture (like VGG16 or ResNet) to enhance its performance.
* Custom Layers: Additional dense layers are added for classification.
* Activation Functions: Uses ReLU activations for hidden layers and softmax for the output layer.

The model can be customized to fit different requirements, and you can modify the architecture in the Jupyter notebook.

## Training

The model is trained using:
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy, Precision, Recall, and F1-score

## Evaluation
Once training is completed, the model is evaluated on a test set. The following metrics are reported:
* Accuracy
* Precision
* Recall
* F1-score

## Results

Here are the results achieved by the model:
* Accuracy: 99.18%
* Precision: 99.19%
* Recall: 99.18%
* F1-score: 99.18%

## Contact

For any questions or issues, please contact:

    Name: Md. Maruf Al Hossain Prince
    Email: marufhossain@iut-dhaka.com
    GitHub: Prince-Baust