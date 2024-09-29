import streamlit as st

# Set page config
st.set_page_config(page_title="Multi-Task DL App", layout="wide", page_icon="🤖")

# Main page content
st.title("Welcome to the Multi-Task Deep Learning App 🚀")

# Main page content
st.header("Available Tasks:")
tasks = [
    "Customer Satisfaction",
    "Football Player Segmentation",
    "Poetry Generator",
    "Pokédex Classifier",
    "Sarcasm Detector"
]

for task in tasks:
    st.write(f"- {task}")

