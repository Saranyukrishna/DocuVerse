import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from pptx import Presentation
from PIL import Image
import io
import time

def extract_text_images_pdf(file):
    text = ""
    images = []

    pdf = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text += page.get_text()

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(image_bytes))
            images.append((f"Page {page_num + 1} - Image {img_index + 1}", img_pil))
    pdf.close()
    return text, images

def extract_text_images_docx(file):
    text = ""
    images = []

    doc = Document(file)
    for para in doc.paragraphs:
        text += para.text + "\n"

    for rel in doc.part._rels:
        rel_obj = doc.part._rels[rel]
        if "image" in rel_obj.target_ref:
            img_data = rel_obj.target_part.blob
            img_pil = Image.open(io.BytesIO(img_data))
            images.append(("DOCX Image", img_pil))
    return text, images

def extract_text_images_pptx(file):
    text = ""
    images = []

    prs = Presentation(file)
    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
            if shape.shape_type == 13:  # Picture
                img_stream = shape.image.blob
                img_pil = Image.open(io.BytesIO(img_stream))
                images.append((f"Slide {i+1} Image", img_pil))
    return text, images

def main():
    st.title("Text & Image Extractor (PDF, DOCX, PPTX)")

    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "pptx"])
    if uploaded_file:
        start_time = time.time()
        filetype = uploaded_file.name.split('.')[-1].lower()

        if filetype == "pdf":
            text, images = extract_text_images_pdf(uploaded_file)
        elif filetype == "docx":
            text, images = extract_text_images_docx(uploaded_file)
        elif filetype == "pptx":
            text, images = extract_text_images_pptx(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return

        st.subheader("Extracted Text")
        if text.strip():
            st.text_area("Text", text, height=400)
        else:
            st.info("No text found.")

        st.subheader("Extracted Images")
        if images:
            for caption, img in images:
                st.image(img, caption=caption, use_column_width=True)
        else:
            st.info("No images found.")

        st.success(f"Done in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
