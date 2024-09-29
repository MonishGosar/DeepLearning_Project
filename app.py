import streamlit as st

# Set page config
st.set_page_config(page_title="Multi-Task ML App", layout="wide", page_icon="🤖")

# Main page content
st.title("Welcome to the Multi-Task Machine Learning App 🚀")
st.write("This app showcases various machine learning tasks. Use the sidebar to navigate between different functionalities.")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.success("Select a task above.")

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

# Additional information
st.markdown("""
    ## About This App

    This Multi-Task Machine Learning application demonstrates the versatility and power of AI across different domains. 
    Each task showcases a unique application of machine learning techniques.

    ### Task Descriptions:

    1. **Customer Satisfaction**: 
       Predict and analyze customer satisfaction levels using historical data and feedback.

    2. **Football Player Segmentation**: 
       Cluster football players based on their attributes and performance metrics to identify similar player types.

    3. **Poetry Generator**: 
       Create original poetry using advanced natural language processing models.

    4. **Pokédex Classifier**: 
       Identify Pokémon species from images, mimicking the functionality of a Pokédex.

    5. **Sarcasm Detector**: 
       Analyze text to detect sarcasm, demonstrating advanced sentiment analysis capabilities.

    ### How to Use:

    1. Select a task from the sidebar on the left.
    2. Follow the instructions on each task's page.
    3. Explore the results and insights provided by our ML models.

    ### Note:

    This app is for demonstration purposes. The models used may not reflect the most current or advanced versions available.

    Enjoy exploring the world of Machine Learning!
""")
