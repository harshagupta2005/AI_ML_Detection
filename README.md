AI/ML Image Classification Project

This repository contains a simple image classification project built in Python. The main goal is to help you understand the basics of machine learning and image processing. It uses a dummy model by default, which you can later replace with your own custom models.

Features

Dummy Model: A basic Keras model is included so you don’t have to start completely from scratch.

Image Preprocessing: Automatically prepares images for the model (e.g., resizing to the correct dimensions).

Self-Contained: No need to provide your own image files — the project generates a black image internally and runs predictions on it.

Prerequisites 

Before running this project, make sure you have these Python libraries installed:

tensorflow

opencv-python

numpy

How to Run

Clone the repository

git clone https://github.com/harshagupta2005/AI_ML_Detection


Navigate to the project folder

cd AI_ML_Detection


Install required libraries

pip install tensorflow opencv-python numpy


Run the code

python main.py

Example Output

When you run the script, you’ll see something like this in the terminal:

Created a dummy image file: sample_test_image.jpg  
Initializing classifier...  
... (More messages) ...  
Prediction Results:  
Predicted Class: Class A  
Confidence: 33.33%  
Cleaned up dummy image file: sample_test_image.jpg  


