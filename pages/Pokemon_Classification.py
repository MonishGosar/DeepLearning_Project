import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import base64

# Title with custom styling
st.markdown("<h1 style='text-align: center; color: #3c5aa6;'>Pokédex Classifier</h1>", unsafe_allow_html=True)

# Cache the model and class indices loading
@st.cache_resource
def load_model_and_class_indices():
    model = tf.keras.models.load_model('pokedex.h5')
    with open('class_indices.pkl', 'rb') as file:
        class_indices = pickle.load(file)
    return model, class_indices

# Prediction function
def predict_image_class(img, model, class_indices):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class[0]]
    return predicted_label

# Function to encode audio file to base64
def get_audio_base64(file_path):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    return audio_base64

# Display the main Pokémon image
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("pokemon_image.png", width=450)

# Load model and class indices
model, class_indices = load_model_and_class_indices()

# File uploader for Pokémon image
uploaded_file = st.file_uploader("Upload an image of a Pokémon", type=["jpg", "jpeg", "png"])

# Play audio and predict class if image is uploaded
if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown("<h2 style='color: #2a75bb;'>Who's that Pokémon?</h2>", unsafe_allow_html=True)
        
        # Play "Who's that Pokémon?" audio when a new image is uploaded
        audio_base64 = get_audio_base64("whos_that_pokemon.mp3")
        st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

        # Display spinner and predict the class
        with st.spinner("Analyzing..."):
            predicted_class = predict_image_class(img, model, class_indices)
        
        # Display predicted Pokémon name with custom styling
        st.markdown(f"<h2 style='color: #ffcb05; text-shadow: 1px 1px #3c5aa6;'>{predicted_class}</h2>", unsafe_allow_html=True)

# Sidebar button to return to the main page
if st.sidebar.button("Return to Main Page"):
    st.switch_page("app.py")
