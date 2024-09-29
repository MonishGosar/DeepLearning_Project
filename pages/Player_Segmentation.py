import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import io

st.set_page_config(layout="wide")
st.title("Enhanced Football Player Segmentation")
st.write("Upload an image, and the model will analyze players from two teams.")

@st.cache_resource
def load_model():
    model = YOLO("yolov9c-seg.pt")
    return model

def determine_team_color(img, mask):
    masked_region = cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    pixels = masked_region.reshape((-1, 3))
    pixels = pixels[np.any(pixels > 0, axis=1)]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    dominant_color = dominant_colors[np.argmax(counts)]
    dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
    hue = dominant_color_hsv[0]
    if 0 <= hue <= 30 or 150 <= hue <= 180:
        return "team_a"
    elif 60 <= hue <= 90:
        return "team_b"
    else:
        return "unknown"

def apply_masks_on_image(img, masks, results):
    img = np.array(img)
    height, width, _ = img.shape
    overlay = np.zeros_like(img, dtype=np.uint8)
    for i, mask in enumerate(masks):
        mask_resized = cv2.resize(mask, (width, height))
        mask_resized = mask_resized.astype(bool)
        team = determine_team_color(img, mask_resized)
        if team == "team_a":
            color = [255, 0, 0]
        elif team == "team_b":
            color = [0, 0, 255]
        else:
            color = [128, 128, 128]
        overlay[mask_resized] = color
    alpha = 0.5
    highlighted_image = cv2.addWeighted(img, 1, overlay, alpha, 0)
    return highlighted_image

def segment_and_highlight_teams(model, img):
    img_np = np.array(img)
    results = model(img_np)
    result = results[0]
    masks = result.masks.data.cpu().numpy()
    highlighted_image = apply_masks_on_image(img_np, masks, results)
    return highlighted_image, results

def count_players_per_team(masks, img):
    team_a_count = 0
    team_b_count = 0
    unknown_count = 0
    for mask in masks:
        team = determine_team_color(img, mask)
        if team == "team_a":
            team_a_count += 1
        elif team == "team_b":
            team_b_count += 1
        else:
            unknown_count += 1
    return team_a_count, team_b_count, unknown_count

def plot_player_positions(results, img_shape):
    fig, ax = plt.subplots(figsize=(10, 6))
    for box in results[0].boxes.xyxy:
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        ax.scatter(x_center, y_center, c='red', s=50)
    ax.set_xlim(0, img_shape[1])
    ax.set_ylim(img_shape[0], 0)  # Invert y-axis to match image coordinates
    ax.set_aspect('equal')
    ax.set_title('Player Positions on Field')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    return fig

def generate_heatmap(results, img_shape):
    heatmap = np.zeros(img_shape[:2])
    for box in results[0].boxes.xyxy:
        x_center = int((box[0] + box[2]) / 2)
        y_center = int((box[1] + box[3]) / 2)
        heatmap[y_center, x_center] += 1
    
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(heatmap, cmap='hot', interpolation='nearest')
    ax.set_title('Player Activity Heatmap')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    fig.colorbar(im, ax=ax, label='Activity Intensity')
    return fig

model = load_model()

# Add a selectbox for different analysis options
analysis_option = st.selectbox(
    "Choose analysis type",
    ("Player Segmentation", "Player Count", "Player Positions", "Activity Heatmap")
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    highlighted_image, results = segment_and_highlight_teams(model, image)
    masks = results[0].masks.data.cpu().numpy()

    if analysis_option == "Player Segmentation":
        col1, col2 = st.columns([2, 2])
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        with col2:
            st.image(highlighted_image, caption='Players Highlighted by Teams', use_column_width=True)

    elif analysis_option == "Player Count":
        team_a_count, team_b_count, unknown_count = count_players_per_team(masks, img_np)
        st.write(f"Team A players: {team_a_count}")
        st.write(f"Team B players: {team_b_count}")
        st.write(f"Unclassified players: {unknown_count}")

        # Create a bar chart
        df = pd.DataFrame({
            'Team': ['Team A', 'Team B', 'Unclassified'],
            'Count': [team_a_count, team_b_count, unknown_count]
        })
        st.bar_chart(df.set_index('Team'))

    elif analysis_option == "Player Positions":
        st.image(image, caption='Uploaded Image', use_column_width=True)
        fig = plot_player_positions(results, img_np.shape)
        st.pyplot(fig)

    elif analysis_option == "Activity Heatmap":
        st.image(image, caption='Uploaded Image', use_column_width=True)
        fig = generate_heatmap(results, img_np.shape)
        st.pyplot(fig)

    # Add a feature to download the processed image
    if st.button('Download Processed Image'):
        # Convert the processed image to bytes
        img_bytes = cv2.imencode('.png', highlighted_image)[1].tobytes()
        st.download_button(
            label="Download image",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )

if st.sidebar.button("Return to Main Page"):
    st.switch_page("app.py")
