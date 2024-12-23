import streamlit as st
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

@st.cache_resource
def load_model():
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    return model, processor

def main():
    st.title("Image Classification App")
    st.write("Upload an image and get its description!")

    # Load model and processor
    model, processor = load_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Add a button to trigger classification
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                # Process the image
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)
                predicted_class = outputs.logits.argmax(-1).item()
                label = model.config.id2label[predicted_class]
                
                # Display results
                st.success(f"This image appears to be of: {label}")

if __name__ == "__main__":
    main()