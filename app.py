import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import time


st.set_page_config(page_title="YOLOv8 Demo", layout="wide")
st.title("YOLOv8 — Simple Web UI")

col1, col2 = st.columns([1, 2])

with col1:
    weights = st.text_input("Weights path", "yolov8n.pt")
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    imgsz = st.number_input("Image size", value=640)
    source_type = st.selectbox("Input source", ["Image upload", "Webcam", "Video file"])
    if 'model_weights' not in st.session_state:
        st.session_state.model_weights = None
    if st.button("Load / Reload model"):
        try:
            st.session_state.model = YOLO(weights)
            st.session_state.model_weights = weights
            st.success("Model loaded")
        except Exception as e:
            st.error(f"Failed to load model: {e}")

with col2:
    model = None
    if ('model' in st.session_state) and (st.session_state.model_weights == weights):
        model = st.session_state.model
    elif ('model' in st.session_state) and (st.session_state.model_weights != weights):
        try:
            model = YOLO(weights)
            st.session_state.model = model
            st.session_state.model_weights = weights
        except Exception as e:
            st.error(f"Failed to load model: {e}")

    if source_type == "Image upload":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded is not None and model is not None:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)
            with st.spinner("Running inference..."):
                arr = np.array(image)
                results = model.predict(source=arr, conf=conf, imgsz=imgsz)
                try:
                    vis = results[0].plot()
                    vis = Image.fromarray(vis)
                    st.image(vis, caption="Detections", use_column_width=True)
                except Exception:
                    st.warning("Could not render annotated image, but inference completed.")

    elif source_type == "Video file":
        vid = st.file_uploader("Upload a video file (mp4)", type=["mp4", "mov", "avi"])
        if vid is not None and model is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(vid.read())
            st.info("Running inference on uploaded video (will save annotated video to runs/detect)...")
            # Run model on the video file — ultralytics will handle the frames and create a saved result
            results = model.predict(source=tfile.name, conf=conf, imgsz=imgsz)
            st.success("Inference complete — check runs/detect/ for outputs")

    else:  # Webcam
        start = st.button("Start webcam")
        stop = st.button("Stop webcam")
        placeholder = st.empty()

        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False

        if start:
            st.session_state.camera_running = True
        if stop:
            st.session_state.camera_running = False

        if st.session_state.camera_running:
            if model is None:
                st.error("Load a model first.")
            else:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot open webcam")
                else:
                    try:
                        while st.session_state.camera_running:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            # BGR -> RGB
                            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            results = model.predict(source=img, conf=conf, imgsz=imgsz)
                            try:
                                vis = results[0].plot()
                                placeholder.image(vis, channels="RGB")
                            except Exception:
                                placeholder.image(img, channels="RGB")
                            # small sleep to yield control
                            time.sleep(0.03)
                    finally:
                        cap.release()
                        st.session_state.camera_running = False

st.markdown("---")
st.markdown("Run this with: `streamlit run app.py`")
