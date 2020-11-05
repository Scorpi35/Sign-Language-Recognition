# Sign Language Detection

## Project Description
**Sign Language Detection** is an AI enabled project aimed to address the crucial problem of **communication gap** between the people with **mutism** and the normal people. Though the project is very far from perfect, it's a tiny contribution from my side and a glimpse of what AI can do.

## Key Concepts
* Image Recognition
* Semenatic Segmentation
* Convolutional Neural Network

## Tool and libraries
* **Language**:- Python
* **IDE**:- PyCharm
* **Libraries**:- Keras, TensorFlow, OpenCV

## Dataset

#### Training Dataset
* **Total Classes**:- 26 (Total Albhabets)
* **Images per class**:- 45500

#### Test Dataset
* **Total Classes**:- 26 (Total Albhabets)
* **Images per class**:- 6500

## Algorithm
1. Gestures image caputed through **Semantic Segmentation**
2. Training a CNN neural network

## Training Description

#### CNN (Convolutional Neural Network)
* **Model**:- Sequential, 3 Conv2D layer, 3 MaxPooling layer, 1 Flatten layer, 2 Dense and 1 Dropout layer
* **Optimizer**:- SGD(Stochastic Gradient Descent)
* **Epochs**:- 25

## How to run the program?

#### Install required libraries
* Run **pip install -r requirements.txt** 

#### Run the program 
* **Step 1:-** python run.py
* **Step 2:-** Let the program run first
* **Step 3:-** Do not change the position of the device once the program runs
* **Step 4:-** Give correspoding hand gestures for the letters (A- Z) and observe the result

## Future Extension
* Training the network with words gestures like Hello, Good and so on.
* Create a plugin for a messenger
