import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np


st.set_page_config(page_title="YOLOv8 Demo", layout="wide")
st.title("YOLOv8 — Simple Web UI")

col1, col2 = st.columns([1, 2])

with col1:
    weights = st.text_input("Weights path", "yolov8n.pt")
    conf = st.slider("Confidence", 0.0, 1.0, 0.25)
    imgsz = st.number_input("Image size", value=640)
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
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_column_width=True)
        if ('model' not in st.session_state) or (st.session_state.model_weights != weights):
            try:
                st.session_state.model = YOLO(weights)
                st.session_state.model_weights = weights
            except Exception as e:
                st.error(f"Model load failed: {e}")
        model = st.session_state.get('model')
        if model is not None:
            with st.spinner("Running inference..."):
                arr = np.array(image)
                results = model.predict(source=arr, conf=conf, imgsz=imgsz)
                try:
                    vis = results[0].plot()
                    vis = Image.fromarray(vis)
                    st.image(vis, caption="Detections", use_column_width=True)
                except Exception:
                    st.warning("Could not render annotated image, but inference completed.")
                # Try to show boxes info
                try:
                    boxes = results[0].boxes
                    if boxes is not None:
                        xyxy = getattr(boxes, 'xyxy', None)
                        confs = getattr(boxes, 'conf', None)
                        cls = getattr(boxes, 'cls', None)
                        info = []
                        if xyxy is not None:
                            xy = xyxy.tolist()
                            conf_list = confs.tolist() if confs is not None else [None]*len(xy)
                            cls_list = cls.tolist() if cls is not None else [None]*len(xy)
                            for i, box in enumerate(xy):
                                info.append({"xyxy": box, "conf": float(conf_list[i]) if conf_list[i] is not None else None, "class": int(cls_list[i]) if cls_list[i] is not None else None})
                        else:
                            info = str(boxes)
                        st.json(info)
                except Exception:
                    st.info("No box details available from results")

st.markdown("---")
st.markdown("Run this with: `streamlit run app.py`")
