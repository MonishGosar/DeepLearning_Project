import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer

# Load the model
@st.cache_resource
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Updated generate_poetry function with temperature control (optional)
def generate_poetry(model, tokenizer, seed_text, next_words, max_sequence_len=99, temperature=1.0):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding="pre")

        # Get the model's predicted probabilities
        predicted_probs = model.predict(token_list, verbose=0)[0]

        # Apply temperature scaling
        predicted_probs = np.log(predicted_probs + 1e-10) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        # Sample the next word index based on the probabilities
        predicted = np.random.choice(len(predicted_probs), p=predicted_probs)

        # Find the word corresponding to the predicted index
        output_word = next((word for word, index in tokenizer.word_index.items() if index == predicted), "")

        # Append the predicted word to the seed text
        seed_text += " " + output_word

    return seed_text

# Load the tokenizer and model
tokenizer = load_tokenizer('tokenizer(2).pickle')
model = load_model('poetry_model(3).h5')

# Get max sequence length from the model input
max_sequence_len = model.input_shape[1]

# Streamlit interface
st.title("Poem Generator")

# Input fields
seed_text = st.text_input("Enter the starting text for the poem:")
num_words = st.number_input("Enter the number of words to generate:", min_value=1, value=20)
temperature = st.slider("Set the temperature for text generation:", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Generate button
if st.button("Generate Poem"):
    if seed_text:
        # Generate poem using the model
        generated_poem = generate_poetry(model, tokenizer, seed_text, num_words, max_sequence_len, temperature)
        
        # Display the generated poem
        st.write("### Generated Poem")
        st.write(generated_poem)
    else:
        st.write("Please enter some starting text.")



