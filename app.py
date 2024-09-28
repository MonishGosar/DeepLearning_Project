import streamlit as st

st.set_page_config(page_title="Multi-Task ML App", layout="wide")

st.sidebar.title("Navigation")

pages = {
    "Home": "app.py",
    "Customer Satisfaction": "pages/Customer_Satisfaction.py",
    "Football Player Segmentation": "pages/Player_Segmentation.py",
    "Poetry Generator": "pages/Poetry_Generation.py",
    "Pokédex Classifier": "pages/Pokemon_Classification.py",
    "Sarcasm Detector": "pages/Sarcasm_Classification.py"
}

selection = st.sidebar.radio("Go to", list(pages.keys()))

if selection == "Home":
    st.title("Welcome to the Multi-Task Machine Learning App")
    st.write("This app showcases various machine learning tasks. Use the sidebar to navigate between different functionalities.")
    
    st.header("Available Tasks:")
    for task in list(pages.keys())[1:]:
        st.write(f"- {task}")
else:
    st.switch_page(pages[selection])