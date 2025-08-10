import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Define image dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Define the list of class names based on your dataset
# IMPORTANT: Update this list with the actual folder names from your dataset
class_names = [
    'animal fish', 
    'animal fish bass', 
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream', 
    'fish sea_food hourse_mackerel', 
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream', 
    'fish sea_food sea_bass', 
    'fish sea_food shrimp', 
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

# Use st.cache_resource to load models only once
@st.cache_resource
def load_all_models():
    """Loads all six trained models from .h5 files."""
    models = {}
    try:
        models['Custom CNN'] = tf.keras.models.load_model('custom_cnn_best_model.h5')
        models['VGG16'] = tf.keras.models.load_model('vgg16_fish_classifier.h5')
        models['ResNet50'] = tf.keras.models.load_model('resnet50_fish_classifier.h5')
        models['MobileNet'] = tf.keras.models.load_model('mobilenet_fish_classifier.h5')
        models['InceptionV3'] = tf.keras.models.load_model('inceptionv3_fish_classifier.h5')
        models['EfficientNetB0'] = tf.keras.models.load_model('efficientnetb0_fish_classifier.h5')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure all .h5 model files are in the same directory.")
        return None

# Load all models at the start of the application
models = load_all_models()

# Streamlit app title
st.title("Multiclass Fish Image Classification")
st.markdown("### Predict the species of a fish from an uploaded image.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("")
    
    st.markdown("### Classification Results")

    if models:
        # Preprocess the image for all models
        img = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        img_array = img_array / 255.0

       # Row 1: Custom CNN, VGG16, ResNet50
        col1, col2, col3 = st.columns(3)
        model_names_row1 = ['Custom CNN', 'VGG16', 'ResNet50']
        cols = [col1, col2, col3]

        for i, model_name in enumerate(model_names_row1):
            if model_name in models:
                with cols[i]:
                    st.markdown(f"#### {model_name}")
                    predictions = models[model_name].predict(img_array, verbose=0)
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = class_names[np.argmax(score)]
                    confidence = 100 * np.max(score)
                    st.info(f"**Prediction:** {predicted_class}")
                    st.success(f"**Confidence:** {confidence:.2f}%")

        st.write("---")

        # Row 2: MobileNet, InceptionV3, EfficientNetB0
        col4, col5, col6 = st.columns(3)
        model_names_row2 = ['MobileNet', 'InceptionV3', 'EfficientNetB0']
        cols = [col4, col5, col6]
        
        for i, model_name in enumerate(model_names_row2):
            if model_name in models:
                with cols[i]:
                    st.markdown(f"#### {model_name}")
                    predictions = models[model_name].predict(img_array, verbose=0)
                    score = tf.nn.softmax(predictions[0])
                    predicted_class = class_names[np.argmax(score)]
                    confidence = 100 * np.max(score)
                    st.info(f"**Prediction:** {predicted_class}")
                    st.success(f"**Confidence:** {confidence:.2f}%")

