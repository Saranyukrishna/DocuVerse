import streamlit as st
import time
from unstructured.partition.auto import partition
from unstructured.documents.elements import Image
from PIL import Image as PILImage
from io import BytesIO
from about import intro

def extract_images(uploaded_file, **kwargs):
    try:
        elements = partition(file=uploaded_file, **kwargs)
        image_elements = [el for el in elements if isinstance(el, Image)]

        if not image_elements:
            st.warning("No images found in the document.")
            return []

        extracted_images = []
        for idx, el in enumerate(image_elements):
            if el.image_data:
                image_bytes = el.image_data.getvalue()
                img = PILImage.open(BytesIO(image_bytes))
                extracted_images.append((f"Image {idx+1}", img))
        return extracted_images
    except Exception as e:
        st.error(f"Failed to extract images: {e}")
        return []

def main():
    intro()

    uploaded_file = st.file_uploader("Upload a document (PDF, PPTX, DOCX)", type=['pdf', 'pptx', 'docx'])

    if uploaded_file is not None:
        start_time = time.time()
        with st.spinner("Extracting images..."):
            images = extract_images(uploaded_file, strategy="hi_res")
            if images:
                for label, img in images:
                    st.image(img, caption=label, use_column_width=True)
            else:
                st.info("No images extracted.")
        execution_time = time.time() - start_time
        st.write(f"Execution time: {execution_time:.2f} seconds")

if __name__ == '__main__':
    main()
