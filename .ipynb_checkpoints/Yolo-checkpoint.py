# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Model path
model_path = 'best.pt'

# Page configuration
st.set_page_config(
    page_title="Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page header
st.title("Object Detection")
st.write(
    """
    Upload an image or paste a YouTube URL for object detection.
    """
)
st.markdown("---")

# Sidebar setup
st.sidebar.title("🔧 Configuration")
st.sidebar.subheader("Model Settings")

# Model options
st.sidebar.markdown("### Task Type")
model_type = st.sidebar.radio("Choose task:", ['Detection', 'Segmentation'], index=0)

st.sidebar.markdown("### Confidence Level")
confidence = st.sidebar.slider("Model Confidence Threshold", 25, 100, 40) / 100

# Model selection based on task type
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load the model
try:
    model = helper.load_model(model_path)
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as ex:
    st.sidebar.error("🚫 Failed to load model")
    st.sidebar.error(f"Error: {ex}")

# Image/YouTube source selection
st.sidebar.subheader("Source Configuration")
source_radio = st.sidebar.radio("Source Type", [settings.IMAGE, settings.YOUTUBE])

source_img = None

# Content for the selected source type
st.markdown("### Detection Results")

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png", "bmp", "webp"])

    if source_img:
        # Display the uploaded image
        st.image(source_img, caption="Uploaded Image", use_column_width=True)
        
        # Detection button
# Detection button
# Detection button
if st.button("🚀 Detect Objects"):
    try:
        uploaded_image = PIL.Image.open(source_img)
        res = model.predict(uploaded_image, conf=confidence)
        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        
        # Hiển thị hình ảnh tải lên ban đầu và hình ảnh đã nhận diện trong cùng một dòng
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.image(res_plotted, caption="Detection Results", use_column_width=True)

        # Expandable section for detection details
        with st.expander("🔍 Detection Details"):
            if boxes:
                for box in boxes:
                    # Lấy thông tin về class và confidence từ thuộc tính cls và conf
                    st.write(f"**Object Class**: {box.cls.item()}, **Confidence**: {box.conf.item():.2f}")
            else:
                st.write("No objects detected.")
    except Exception as ex:
        st.error("❗ Error processing the image.")
        st.error(ex)


elif source_radio == settings.YOUTUBE:
    st.write("🌐 **YouTube Mode Selected**")
    helper.play_youtube_video(confidence, model)

else:
    st.error("⚠️ Please select a valid source type!")

