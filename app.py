import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="AI Traffic Violation Detection", page_icon="🚦", layout="centered")

st.title("🚦 AI-Powered Traffic Violation Detection System")
st.caption("By Hemanth Nag, REVA University")

st.markdown("### Upload a traffic image to detect possible violations")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image
    image = Image.open(uploaded_file)
    image_path = "uploaded_image.jpg"
    image.save(image_path)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Running AI model..."):
        # Load YOLOv8 model
        model = YOLO('yolov8n.pt')
        results = model(image_path)
        results[0].save(filename="detected_output.jpg")

        # Analyze detections
        df = results[0].pandas().xyxy[0]
        violations = []

        for _, row in df.iterrows():
            if row['name'] in ['motorcycle', 'person']:
                violations.append({
                    'Object': row['name'],
                    'Confidence': f"{row['confidence']:.2f}",
                    'Violation': 'No Helmet Detected (Assumed)' if row['name'] == 'person' else 'None'
                })

        if violations:
            report_df = pd.DataFrame(violations)
            report_df.to_csv("violation_report.csv", index=False)

            st.success("✅ Detection Completed!")
            st.image("detected_output.jpg", caption="Detection Result", use_column_width=True)
            st.markdown("### 📄 Violation Report")
            st.dataframe(report_df)
        else:
            st.warning("No detectable objects found in image.")
