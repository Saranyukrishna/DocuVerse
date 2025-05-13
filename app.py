import streamlit as st
import fitz
from docx import Document
from pptx import Presentation
from PIL import Image
import io
import os
from pathlib import Path
import base64
import numpy as np
import cohere
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

co = cohere.ClientV2(api_key=cohere_api_key)
genai.configure(api_key=gemini_api_key)
client = genai

OUTPUT_DIR = Path("output")
IMAGES_DIR = OUTPUT_DIR / "images"
TEXT_FILE = OUTPUT_DIR / "extracted_text.txt"
MAX_PIXELS = 1568 * 1568


def save_image(image_pil, image_count):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMAGES_DIR / f"image_{image_count}.png"
    image_pil.save(img_path)
    return str(img_path)


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

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text += shape.text + "\n"
            if shape.shape_type == 13:
                img_stream = shape.image.blob
                img_pil = Image.open(io.BytesIO(img_stream))
                img_path = save_image(img_pil, image_count)
                image_paths.append(img_path)
                image_count += 1
    return text, image_paths


def resize_image(pil_image):
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))


def base64_from_image(img_path):
    pil_image = Image.open(img_path)
    img_format = pil_image.format or "PNG"
    resize_image(pil_image)
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format=img_format)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/{img_format.lower()};base64,{encoded}"


@st.cache_data
def generate_embeddings(image_paths):
    doc_embeddings = []
    for path in image_paths:
        api_input_document = {
            "content": [
                {"type": "image", "image": base64_from_image(path)},
            ]
        }
        resp = co.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            inputs=[api_input_document],
        )
        emb = np.asarray(resp.embeddings.float[0])
        doc_embeddings.append(emb)
    return np.vstack(doc_embeddings)


def search_image(question, embeddings, image_paths):
    resp = co.embed(
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
        texts=[question],
    )
    query_emb = np.asarray(resp.embeddings.float[0])
    scores = np.dot(query_emb, embeddings.T)
    top_idx = np.argmax(scores)
    return image_paths[top_idx]


def ask_gemini(question, img_path):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""Answer the question based on the following image.
Don't use markdown.
Please provide enough context for your answer.

Question: {question}"""
    img = Image.open(img_path)
    response = model.generate_content([prompt, img])
    return response.text


st.title("Document Q&A: Extract, View, Ask (Text + Image)")

uploaded_file = st.file_uploader("Upload PDF, DOCX or PPTX", type=["pdf", "docx", "pptx"])
question = st.text_input("Ask a question based on image content")

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()

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
        st.error("Unsupported file.")
        st.stop()

    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(text)

    st.success("Extraction complete!")
    st.subheader("Extracted Images")

    row_images = st.container()
    with row_images:
        cols = st.columns(5)
        for i, img_path in enumerate(image_paths):
            with cols[i % 5]:
                st.image(img_path, caption=f"Image {i + 1}", use_container_width=True)

    selected_img = None
    for img_path in image_paths:
        if st.button(f"Ask based on {Path(img_path).name}"):
            selected_img = img_path

    if question:
        if selected_img:
            st.image(selected_img, caption="Selected Image", use_container_width=True)
            answer = ask_gemini(question, selected_img)
            st.success("Answer:")
            st.write(answer)
        else:
            st.warning("Please click a button to select an image.")

    st.subheader("Full Extracted Text")
    st.text_area("Text", text, height=300)
