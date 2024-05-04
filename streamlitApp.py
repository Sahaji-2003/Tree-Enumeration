import streamlit as st
from PIL import Image
import numpy as np
import efficientnet.keras as efn
import tensorflow as tf
from tensorflow.keras.layers import Dropout

model = tf.keras.models.load_model("A:/DeepFake project/saved_model/efn_model")
def classify_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)
    
    # Make a prediction
    prediction = model.predict(image)
    predicted_class = 'Real' if prediction[0][0] > 0.5 else 'Fake'
    return predicted_class

# Set up Streamlit
st.title("Deep Fake Classification")
st.write("Upload an image to classify it.")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image was uploaded
if uploaded_image is not None:
    # Load the image
    image = Image.open(uploaded_image)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Classify the image
    predicted_class = classify_image(image)
    
    # Display the predicted class
    st.write(f"Predicted Class: {predicted_class}")
