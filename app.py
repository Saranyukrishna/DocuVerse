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

def ask_gemini(question, context=None, img_path=None):
    """Query Gemini with optional context and/or image"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        if img_path and context:
            # Both image and text context
            prompt = f"""Answer the question based on the following image and additional context if relevant.
If the question is general and not related to the context, answer it generally.

Context: {context}
Question: {question}"""
            img = Image.open(img_path)
            response = model.generate_content([prompt, img])
        elif img_path:
            # Image only
            prompt = f"""Answer the question based on the following image if relevant.
If the question is general and not about the image, answer it generally.

Question: {question}"""
            img = Image.open(img_path)
            response = model.generate_content([prompt, img])
        elif context:
            # Text context only
            prompt = f"""Answer the question based on the following context if relevant.
If the question is general and not related to the context, answer it generally.

Context: {context}
Question: {question}"""
            response = model.generate_content(prompt)
        else:
            # No context - general question
            response = model.generate_content(question)
            
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
    st.session_state.active_tab = "text"  # Track which tab is active

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
    # Create tabs for text and image analysis
    tab1, tab2, tab3 = st.tabs(["📝 Text Analysis", "🖼️ Image Analysis", "💬 General Chat"])
    
    with tab1:
        st.subheader("Text Analysis")
        st.text_area("Extracted Text", st.session_state.text, height=200)
        
        text_question = st.text_input("Ask a question about the text content")
        if text_question:
            with st.spinner("Analyzing text..."):
                answer = ask_gemini(text_question, context=st.session_state.text)
                st.write("Answer:", answer)

    with tab2:
        st.subheader("Image Analysis")
        if st.session_state.image_paths:
            selected_image = st.selectbox("Select an image to analyze", st.session_state.image_paths)
            st.image(selected_image, caption="Selected Image", use_column_width=True)

            image_question = st.text_input("Ask a question about the image")
            if image_question:
                with st.spinner("Analyzing image..."):
                    answer = ask_gemini(image_question, img_path=selected_image, context=st.session_state.text)
                    st.write("Answer:", answer)
        else:
            st.write("No images found in the document.")
    
    with tab3:
        st.subheader("General Chat")
        general_question = st.text_input("Ask any general question")
        if general_question:
            with st.spinner("Thinking..."):
                answer = ask_gemini(general_question)
                st.write("Answer:", answer)

else:
    # If no document is uploaded, show general chat interface
    st.subheader("General Chat")
    general_question = st.text_input("Ask any question")
    if general_question:
        with st.spinner("Thinking..."):
            answer = ask_gemini(general_question)
            st.write("Answer:", answer)

# Clean up when done
st.session_state.cleanup = cleanup
