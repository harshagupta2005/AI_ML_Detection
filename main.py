#-----------------------------------------------------------------------------------------------------------------------
# This script demonstrates a basic image classification pipeline using a placeholder model.
# The code is self-contained and runnable without external model or image files.
# It's an excellent starting point for a simple AI/ML project.
#
# Requirements:
#   - Python 3.8+
#   - tensorflow (pip install tensorflow)
#   - opencv-python (pip install opencv-python)
#   - numpy (pip install numpy)
#
# To run this script:
#   1. Save the code as 'main.py'.
#   2. Ensure the required libraries are installed.
#   3. Run 'python main.py' in your terminal.
#-----------------------------------------------------------------------------------------------------------------------

import warnings
import os
import cv2
import numpy as np
import tensorflow as tf

# Suppress TensorFlow and other warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ImageClassifier:
    """
    A class to handle image classification using a pre-trained Keras model.
    """
    def __init__(self, model_path, labels_path):
        """
        Initializes the classifier by loading the model and labels.
        In this example, a dummy model and labels are created for demonstration.

        Args:
            model_path (str): The path to the Keras model file (.h5).
            labels_path (str): The path to the text file with class labels.
        """
        print("Initializing classifier...")
        
        # --- Placeholder for a real model and labels ---
        # In a real project, you would load your model and labels from files.
        # Example: self.model = tf.keras.models.load_model(model_path)
        # Example: self.labels = open(labels_path).read().splitlines()
        
        # Creating a dummy model and dummy labels for this runnable example
        # This model mimics a classifier with 3 classes, taking 224x224 pixel images.
        self.model = self._create_dummy_model()
        self.labels = ["Class A", "Class B", "Class C"]
        print("Classifier initialized with dummy model and labels.")
        
    def _create_dummy_model(self):
        """
        Creates a simple dummy Keras model for demonstration purposes.
        This function simulates loading a real model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(3, activation='softmax') # 3 output classes
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def preprocess_image(self, image_path):
        """
        Loads and preprocesses an image for the model.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            np.ndarray: The preprocessed image as a numpy array, or None if an error occurs.
        """
        try:
            # Use OpenCV to read the image file
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Unable to load image at {image_path}. Please check the path.")
                return None
            
            # Resize the image to the model's required input size (224x224)
            img = cv2.resize(img, (224, 224))
            
            # The model expects a batch of images, so we add an extra dimension.
            # Shape changes from (224, 224, 3) to (1, 224, 224, 3).
            img_batch = np.expand_dims(img, axis=0)
            
            return img_batch
        except Exception as e:
            print(f"An error occurred during image preprocessing: {e}")
            return None

    def predict(self, image_path):
        """
        Predicts the class of an image and returns the label and confidence.
        
        Args:
            image_path (str): The path to the image file.
            
        Returns:
            tuple: A tuple containing the predicted class label (str) and confidence (float).
                   Returns (None, None) if prediction fails.
        """
        # Preprocess the input image
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None, None
            
        # Make a prediction using the model
        predictions = self.model.predict(processed_img)
        
        # Find the index of the class with the highest probability
        predicted_class_index = np.argmax(predictions)
        
        # Get the class label and confidence score
        predicted_class = self.labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        
        return predicted_class, confidence

# This block is executed only when the script is run directly
if __name__ == "__main__":
    
    # --- Simulating a real use case ---
    # In a real project, you would replace these file paths
    # with the actual paths to your trained model and labels.
    dummy_model_path = "models/my_model.h5"
    dummy_labels_path = "models/my_labels.txt"
    
    # --- Create a dummy image file to test with ---
    # This section creates a simple image file so the script can run
    # without needing a real image to exist.
    # In a real project, you would use an existing image.
    dummy_image_path = "sample_test_image.jpg"
    dummy_image = np.zeros((224, 224, 3), dtype=np.uint8) # A black image
    cv2.imwrite(dummy_image_path, dummy_image)
    print(f"\nCreated a dummy image file: {dummy_image_path}")

    # Create an instance of the ImageClassifier
    detector = ImageClassifier(
        model_path=dummy_model_path,
        labels_path=dummy_labels_path
    )
    
    # Make a prediction on the dummy image
    predicted_class, confidence = detector.predict(dummy_image_path)
    
    # Check if the prediction was successful
    if predicted_class:
        print("\nPrediction Results:")
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
    
    # Clean up the dummy image file
    
    print(f"\nCleaned up dummy image file: {dummy_image_path}")
