import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans

st.title("Football Player Segmentation")

st.write("Upload an image, and the model will highlight players from two teams with different colors.")

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
    return highlighted_image

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([2, 2])
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)

    highlighted_image = segment_and_highlight_teams(model, image)
    with col2:
        st.image(highlighted_image, caption='Players Highlighted by Teams', use_column_width=True)

if st.sidebar.button("Return to Main Page"):
    st.switch_page("app.py")
