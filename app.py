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
from langchain.schema import HumanMessage, AIMessage

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
MAX_PIXELS = 1024 * 1024  # Reduced from original
SUPPORTED_TYPES = ["pdf", "docx", "pptx"]
GEMINI_MODEL = "gemini-1.5-flash"
IMAGE_QUALITY = 75  # For JPEG compression

# Create temporary directory
OUTPUT_DIR = Path(tempfile.mkdtemp())
IMAGES_DIR = OUTPUT_DIR / "images"
TEXT_FILE = OUTPUT_DIR / "extracted_text.txt"

def cleanup():
    """Remove temporary files"""
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

def save_image(image_pil, image_count):
    """Save optimized image to temporary directory"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = IMAGES_DIR / f"image_{image_count}.jpg"  # Using JPEG for better compression
    
    # Optimize image before saving
    image_pil = resize_image(image_pil)
    
    # Save with optimized quality
    image_pil.save(
        img_path, 
        format="JPEG", 
        quality=IMAGE_QUALITY, 
        optimize=True
    )
    return str(img_path)

def resize_image(pil_image):
    """Resize image if too large with better aspect ratio handling"""
    org_width, org_height = pil_image.size
    if org_width * org_height > MAX_PIXELS:
        scale_factor = (MAX_PIXELS / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    return pil_image

def base64_from_image(img_path):
    """Convert image to optimized base64"""
    try:
        pil_image = Image.open(img_path)
        pil_image = resize_image(pil_image)
        
        with io.BytesIO() as buffer:
            pil_image.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

# [Previous extraction functions remain the same...]

def summarize_text(text):
    """Generate a summary of the extracted text"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""Please provide a concise summary of the following text, highlighting the key points:
        
        {text[:15000]}  # Limiting context size
        
        Summary:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate summary: {str(e)}"

def extract_key_points(text):
    """Extract key points from text"""
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""Extract the most important key points from this text as bullet points:
        
        {text[:15000]}  # Limiting context size
        
        Key Points:"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not extract key points: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("📄 Document Q&A with Image Analysis")

# Initialize session state with new text analysis features
if 'text_summary' not in st.session_state:
    st.session_state.text_summary = ""
if 'key_points' not in st.session_state:
    st.session_state.key_points = ""

# [Previous session state initialization remains...]

# Process file when uploaded - add text analysis features
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
            
            # Generate text analysis
            with st.spinner("Analyzing text content..."):
                st.session_state.text_summary = summarize_text(st.session_state.text)
                st.session_state.key_points = extract_key_points(st.session_state.text)
            
            # Save text to file
            OUTPUT_DIR.mkdir(exist_ok=True)
            with open(TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(st.session_state.text)
            
            st.session_state.processed = True
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"Failed to process document: {str(e)}")
            cleanup()

# [Previous tab setup remains the same until Text Analysis tab...]

# Text Analysis Tab - Enhanced
with tab1:
    st.subheader("Text Analysis")
    
    if st.session_state.processed:
        # Display text analysis features in columns
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📝 Summary", expanded=True):
                st.write(st.session_state.text_summary)
        
        with col2:
            with st.expander("🔑 Key Points", expanded=True):
                st.write(st.session_state.key_points)
        
        # Full text view with improved display
        with st.expander("📄 View Full Extracted Text"):
            st.text_area(
                "Full Text", 
                st.session_state.text, 
                height=300, 
                label_visibility="collapsed"
            )
            st.download_button(
                "Download Full Text",
                data=st.session_state.text,
                file_name="extracted_text.txt",
                mime="text/plain"
            )
    
    # Enhanced chat interface for text analysis
    text_chat_container = st.container()
    
    # Add suggested questions
    if st.session_state.processed and not st.session_state.text_chat_history:
        st.markdown("**Suggested questions:**")
        cols = st.columns(2)
        questions = [
            "What are the main topics covered?",
            "Can you summarize the key findings?",
            "Are there any important dates mentioned?",
            "Who are the main people or organizations mentioned?"
        ]
        
        for i, question in enumerate(questions):
            with cols[i % 2]:
                if st.button(question):
                    st.session_state.text_input = question  # Pre-fill the input
    
    user_text_input = st.text_input(
        "Ask about the text content:", 
        key="text_input",
        placeholder="Type your question here...",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        text_send_button = st.button("Send", key="text_send")
    with col2:
        if st.button("Clear Chat", key="text_clear"):
            st.session_state.text_chat_history = []
            st.rerun()
    
    if text_send_button and user_text_input:
        st.session_state.text_chat_history.append(HumanMessage(content=user_text_input))
        if user_text_input.lower() == 'close the chat':
            st.stop()
        
        with st.spinner("Analyzing text..."):
            # Include both the full text and summary in context
            context = f"Document Summary:\n{st.session_state.text_summary}\n\nFull Text:\n{st.session_state.text[:10000]}"  # Limit context size
            answer = ask_gemini(user_text_input, context=context)
            st.session_state.text_chat_history.append(AIMessage(content=answer))
            st.session_state.scroll = True
            st.rerun()
    
    render_chat(text_chat_container, st.session_state.text_chat_history)

# [Rest of the code remains the same...]
