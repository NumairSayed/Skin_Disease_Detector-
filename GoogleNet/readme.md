# GoogLeNet Image Classification

This project implements image classification using **GoogLeNet**, a deep learning model. It includes scripts for training, validation, testing, and inference on custom datasets.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Structure](#dataset-structure)
- [Data Split](#data-split)
- [Model Information](#model-information)
- [Running the Project](#running-the-project)
- [Results and Evaluation](#results-and-evaluation)
- [Conclusion](#conclusion)

## Environment Setup

We recommend using **Conda** for managing the Python environment for this project. Follow the steps below to set up the environment:

### Step 1: Clone the repository



### Step 2: Install Conda (if not already installed)
If you don’t have Conda installed, download and install it

### Step 3: Create and activate the Conda environment
Create a new Conda environment named googlenet-env and activate it:

conda create --name googlenet-env python=3.8
conda activate googlenet-env

### Step 4: Install dependencies
Once your Conda environment is active, install the necessary dependencies using the requirements.txt file:

pip install -r requirements.txt



### Dataset Structure
The dataset is expected to be in ImageFolder format, where each class has its own directory containing the respective images. The structure should look like this:

/img
    /class_1
        image_1.jpg
        image_2.jpg
        ...
    /class_2
        image_1.jpg
        image_2.jpg
        ...
    ...


Each subdirectory represents a class, and the images inside the subdirectory belong to that class. Make sure that all images are properly labeled based on their directory names.

#### Example
If you have two classes, cats and dogs, the structure would look like:

/img
    /cats
        cat1.jpg
        cat2.jpg
    /dogs
        dog1.jpg
        dog2.jpg



Data Split
The dataset will be split into training and testing sets using an 80-20 split by default. The split is handled automatically in the dataloader.py script using PyTorch’s random_split function:

80% of the images are used for training.
20% of the images are used for testing.
You don’t need to manually split the dataset; this will be done during the loading process.

Model Information
GoogLeNet Model
GoogLeNet is a deep learning model introduced by Google in 2014, which won the ILSVRC (ImageNet Large Scale Visual Recognition Challenge) 2014. It uses a novel Inception module to efficiently process images while keeping the model size small.

Key Features of GoogLeNet:
Inception Modules: A combination of 1x1, 3x3, and 5x5 convolutions, along with max pooling operations.
Efficient Parameter Use: Despite its depth, GoogLeNet has fewer parameters compared to larger models like VGGNet.
Number of Parameters
GoogLeNet has approximately 6.8 million parameters, significantly smaller than other deep learning models like VGGNet, which has over 138 million parameters. This makes GoogLeNet more computationally efficient while maintaining high accuracy.

### Running the Project

Training the Model:
python train.py


Validating the Model:
python validate.py


Testing the Model
To calculate additional metrics such as accuracy, precision, recall, and F1 score, use:
python test.py


Inference on New Images:
python inference.py







