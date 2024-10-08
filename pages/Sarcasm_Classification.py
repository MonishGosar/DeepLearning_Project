# pages/page_sarcasm.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

@st.cache_resource
def load_sarcasm_model_and_tokenizer():
    with open('sarcasm_tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = tf.keras.models.load_model('sarcasm.h5')
    return model, tokenizer

def main():
    st.title('Sarcasm Detector')

    max_length = 100
    padding_type = 'post'
    trunc_type = 'post'

    model, tokenizer = load_sarcasm_model_and_tokenizer()

    headline = st.text_input('Enter a headline:')

    if st.button('Detect Sarcasm'):
        if headline:
            headline_sequence = tokenizer.texts_to_sequences([headline])
            
            padded_headline = pad_sequences(headline_sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            
            prediction = model.predict(padded_headline)
            
            confidence = prediction[0][0]
            is_sarcastic = confidence > 0.5
            
            st.write(f"{'Sarcastic' if is_sarcastic else 'Not Sarcastic'}: {headline}")
        else:
            st.warning("Please enter a headline.")

    if st.sidebar.button("Return to Main Page"):
        st.switch_page("app.py")

if __name__ == "__main__":
    main()
