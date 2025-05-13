import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from pptx import Presentation
from PIL import Image
import io
import os
import time
from pathlib import Path

OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
TEXT_FILE = OUTPUT_DIR / "extracted_text.txt"

def save_text(text):
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

def save_image(image_pil, image_count):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGES_DIR / f"image_{image_count}.png"
    image_pil.save(image_path)
    return str(image_path)

def extract_pdf(file):
    text = ""
    image_paths = []
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    image_count = 1

    for page in pdf:
        text += page.get_text()
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            img_bytes = base_image["image"]
            img_pil = Image.open(io.BytesIO(img_bytes))
            img_path = save_image(img_pil, image_count)
            image_paths.append(img_path)
            image_count += 1
    pdf.close()
    return text, image_paths

def extract_docx(file):
    text = ""
    image_paths = []
    doc = Document(file)
    image_count = 1

    for para in doc.paragraphs:
        text += para.text + "\n"

    for rel in doc.part._rels:
        rel_obj = doc.part._rels[rel]
        if "image" in rel_obj.target_ref:
            img_data = rel_obj.target_part.blob
            img_pil = Image.open(io.BytesIO(img_data))
            img_path = save_image(img_pil, image_count)
            image_paths.append(img_path)
            image_count += 1
    return text, image_paths

def extract_pptx(file):
    text = ""
    image_paths = []
    prs = Presentation(file)
    image_count = 1

    for i, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
            if shape.shape_type == 13:  # Picture
                img_stream = shape.image.blob
                img_pil = Image.open(io.BytesIO(img_stream))
                img_path = save_image(img_pil, image_count)
                image_paths.append(img_path)
                image_count += 1
    return text, image_paths

def main():
    st.title("Document Extractor: Text + Images + LLM Ready")

    uploaded_file = st.file_uploader("Upload PDF, DOCX, or PPTX", type=["pdf", "docx", "pptx"])
    if uploaded_file:
        start_time = time.time()
        file_ext = uploaded_file.name.split('.')[-1].lower()

        # Clear previous data
        if OUTPUT_DIR.exists():
            for f in OUTPUT_DIR.glob("*"):
                if f.is_file():
                    f.unlink()
            if IMAGES_DIR.exists():
                for img in IMAGES_DIR.glob("*"):
                    img.unlink()

        if file_ext == "pdf":
            text, image_paths = extract_pdf(uploaded_file)
        elif file_ext == "docx":
            text, image_paths = extract_docx(uploaded_file)
        elif file_ext == "pptx":
            text, image_paths = extract_pptx(uploaded_file)
        else:
            st.error("Unsupported file type.")
            return

        save_text(text)

        st.success(f"Text and {len(image_paths)} image(s) saved in 'output/'")
        st.text_area("Extracted Text", text, height=300)

        for img_path in image_paths:
            st.image(img_path, caption=img_path, use_column_width=True)

        st.markdown(f"**Text File:** `{TEXT_FILE}`")
        st.markdown(f"**Image Directory:** `{IMAGES_DIR}`")

        st.success(f"Extraction done in {time.time() - start_time:.2f} seconds")

        if st.button("Pass to LLM"):
            st.info("Text and image paths are ready to be passed to your LLM.")
            st.code(f"with open('{TEXT_FILE}') as f: content = f.read()\n# image_paths = {image_paths}", language="python")

if __name__ == "__main__":
    main()
