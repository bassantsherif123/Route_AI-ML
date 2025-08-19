import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title='Traffic Light Detection')

st.title('🚦Traffic Light Detection')
model = YOLO("yolov5s.pt")

def classify_color(image, box):
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

    # DEFINE COLORS IN HSV: Make easier to seperate colors
    # Traffic lights can have Red, Green, or Yellow.
    # Red has two ranges because it wraps around in the HSV color circle.
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([80, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    # check for red
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)

    red_mask = red_mask1 | red_mask2
    red_pixels = cv2.countNonZero(red_mask)

    # check for green
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    green_pixels = cv2.countNonZero(green_mask)

    # check for yellow
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    yellow_pixels = cv2.countNonZero(yellow_mask)

    # Decide which color is dominant
    if max(red_pixels, green_pixels, yellow_pixels) == yellow_pixels:
        return "Yellow"
    elif max(red_pixels, green_pixels, yellow_pixels) == green_pixels:
        return "Green"
    elif max(red_pixels, green_pixels, yellow_pixels) == red_pixels:
        return "Red"
    else:
        return "UNKnOWN"


upload = st.file_uploader('📸 Choose File', type=["png", "jpg", "jpeg", "webp"])

if upload is not None:
    img = Image.open(upload)
    img_np = np.array(img)
    img_resized = cv2.resize(img_np, (640, 480))

    # Class 9: 'traffic light'
    results = results = model.predict(img_resized, classes=[9], verbose=False)

    st.image(results[0].plot(), caption='Detected Objects', use_column_width=True)

    st.sidebar.markdown('# Detected Colors🚦', unsafe_allow_html=True)
    for box in results[0].boxes.xyxy:
        color = classify_color(img_resized, box[:4])
        if color == 'Red':
            st.sidebar.markdown('🔴 **Red Light Detected!**', unsafe_allow_html=True)
        elif color == 'Green':
            st.sidebar.markdown('🟢 **Green Light Detected!**', unsafe_allow_html=True)
        elif color == 'Yellow':
            st.sidebar.markdown('🟡 **Yellow Light Detected!**', unsafe_allow_html=True)
