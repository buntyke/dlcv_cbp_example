import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
import io

# API details
API_URL = "http://localhost:8000/predict"

st.title("ResNet50 Image Classifier")
st.write("Upload an image to get the top 3 predictions from the ResNet50 ImageNet model.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Read the file content
        file_content = uploaded_file.read()

        # Validate and open the image
        try:
            image = Image.open(io.BytesIO(file_content))
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image. Please upload a JPG or PNG file.")
            st.stop()

        # Send image to the API for prediction
        with st.spinner("Classifying the image..."):
            response = requests.post(API_URL, files={"file": file_content})

        # Check API response
        if response.status_code == 200:
            predictions = response.json().get("prediction", [])
            if predictions:
                st.success("Top 3 Predictions:")
                for idx, pred in enumerate(predictions, start=1):
                    st.write(f"{idx}. {pred['class']}: {pred['confidence']:.2f}%")
            else:
                st.warning("No predictions returned. Please try a different image.")
        else:
            st.error(f"Failed to get predictions. API returned status code: {response.status_code}")
            st.json(response.json())

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
