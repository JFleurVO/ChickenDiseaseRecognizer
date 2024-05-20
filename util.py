import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """
    # Convert image to (64, 64) size
    image = ImageOps.fit(image, (64, 64), Image.Resampling.LANCZOS)

    # Convert image to numpy array
    image_array = np.asarray(image)

    # Normalize image
    normalized_image_array = (image_array.astype(np.float32) / 255.0)

    # Reshape image for model input
    input_image = normalized_image_array.reshape((1, 64, 64, 3))

    # Make prediction
    prediction = model.predict(input_image)

    # Debugging: Print raw prediction array
    print("Raw prediction array:", prediction)

    # Get the index with highest probability
    predicted_index = np.argmax(prediction)
    
    # Get the predicted class name
    predicted_class_name = class_names[predicted_index]

    # Get the confidence score for the predicted class
    confidence_score = prediction[0][predicted_index]

    return predicted_class_name, confidence_score