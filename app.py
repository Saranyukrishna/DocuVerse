import streamlit as st
import fitz  # PyMuPDF
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
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Initialize APIs
cohere_api_key = os.getenv("COHERE_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not cohere_api_key or not gemini_api_key:
    st.error("API keys not found. Please check your .env file")
    st.stop()

try:
    co = cohere.Client(api_key=cohere_api_key)
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    st.error(f"Failed to initialize API clients: {str(e)}")
    st.stop()

# Configuration
MAX_PIXELS = 1568 * 1568
SUPPORTED_TYPES = ["pdf", "docx", "pptx"]
GEMINI_MODEL = "gemini-1.5-flash"  # Current recommended model

# Create temporary directory
OUTPUT_DIR = Path(tempfile.mkdtemp())
IMAGES_DIR = OUTPUT_DIR / "images"
TEXT_FILE = OUTPUT_DIR / "extracted_text.txt"

def cleanup():
    """Remove temporary files"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

def save_image(image_pil, image_count):
    """Save image to temporary directory"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMAGES_DIR / f"image_{image_count}.png"
    image_pil.save(img_path)
    return str(img_path)

def resize_image(pil_image):
    """Resize image if too large"""
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

def base64_from_image(img_path):
    """Convert image to base64"""
    try:
        pil_image = Image.open(img_path)
        img_format = pil_image.format or "PNG"
        resize_image(pil_image)
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format=img_format)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/{img_format.lower()};base64,{encoded}"
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def extract_pdf(file):
    """Extract text and images from PDF"""
    text = ""
    image_paths = []
    image_count = 1
    
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as pdf:
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
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
    return text, image_paths

def extract_docx(file):
    """Extract text and images from DOCX"""
    text = ""
    image_paths = []
    image_count = 1
    
    try:
        doc = Document(file)
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
    except Exception as e:
        st.error(f"Error processing DOCX: {str(e)}")
    return text, image_paths

def extract_pptx(file):
    """Extract text and images from PPTX"""
    text = ""
    image_paths = []
    image_count = 1
    
    try:
        prs = Presentation(file)
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
                if shape.shape_type == 13:  # Picture shape
                    img_stream = shape.image.blob
                    img_pil = Image.open(io.BytesIO(img_stream))
                    img_path = save_image(img_pil, image_count)
                    image_paths.append(img_path)
                    image_count += 1
    except Exception as e:
        st.error(f"Error processing PPTX: {str(e)}")
    return text, image_paths

def ask_gemini(question, img_path):
    """Query Gemini about an image"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""Answer the question based on the following image.
Be concise but provide enough context for your answer.
Question: {question}"""
        
        img = Image.open(img_path)
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"Error querying Gemini: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Q&A with Image Analysis")

# File upload section
with st.expander("Upload Document", expanded=True):
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=SUPPORTED_TYPES,
        help=f"Supported formats: {', '.join(SUPPORTED_TYPES)}"
    )

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.text = ""
    st.session_state.image_paths = []
    st.session_state.selected_img = None

# Process file when uploaded
if uploaded_file and not st.session_state.processed:
    with st.spinner("Extracting content from document..."):
        file_ext = uploaded_file.name.split(".")[-1].lower()
        
        try:
            if file_ext == "pdf":
                st.session_state.text, st.session_state.image_paths = extract_pdf(uploaded_file)
            elif file_ext == "docx":
                st.session_state.text, st.session_state.image_paths = extract_docx(uploaded_file)
            elif file_ext == "pptx":
                st.session_state.text, st.session_state.image_paths = extract_pptx(uploaded_file)
            
            # Save text to file
            OUTPUT_DIR.mkdir(exist_ok=True)
            with open(TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(st.session_state.text)
            
            st.session_state.processed = True
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Failed to process document: {str(e)}")
            cleanup()

# Show extracted content
if st.session_state.processed:
    # Display images in a grid
    if st.session_state.image_paths:
        st.subheader("📸 Extracted Images")
        cols = st.columns(4)
        for i, img_path in enumerate(st.session_state.image_paths):
            with cols[i % 4]:
                st.image(img_path, caption=f"Image {i+1}", use_container_width=True)  # Updated parameter
                if st.button(f"Select Image {i+1}", key=f"select_{i}"):
                    st.session_state.selected_img = img_path
    
    # Show selected image
    if st.session_state.selected_img:
        st.subheader("🔍 Selected Image")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(st.session_state.selected_img, use_container_width=True)  # Updated parameter
        with col2:
            question = st.text_input("Ask a question about this image")
            if question:
                with st.spinner("Analyzing image..."):
                    answer = ask_gemini(question, st.session_state.selected_img)
                    if "Error querying Gemini" in answer:
                        st.error(answer)
                    else:
                        st.markdown(f"**Answer:** {answer}")
    
    # Show extracted text
    st.subheader("📝 Extracted Text")
    st.text_area("Full Text", st.session_state.text, height=300)

# Cleanup when done
if st.button("Clear Session"):
    cleanup()
    st.session_state.clear()
    st.rerun()

import atexit
atexit.register(cleanup)
