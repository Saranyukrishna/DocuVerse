A powerful Streamlit application that extracts text and images from PDF, DOCX, and PPTX files, then allows you to ask questions about the content using AI models (Gemini, Groq) with optional web search capabilities.

Features
Multi-format Support
Process PDF, DOCX, and PPTX files

Text Extraction
Extract and analyze text content

Image Extraction
Identify and extract embedded images

AI-powered Analysis

Text Q&A using Gemini or Groq

Image analysis with Gemini's vision capabilities

General chat with optional web search

Image Enhancement
Adjust contrast and sharpness of extracted images

Web Search Integration
Augment answers with current web results when needed

Technologies Used
Python 3.10+

Streamlit – Web application framework

PyMuPDF (fitz) – PDF processing

python-docx – DOCX processing

python-pptx – PPTX processing

Pillow (PIL) – Image processing

Google Gemini AI – Multimodal AI model

Groq – Fast LLM inference

Tavily – Web search API

LangChain – Chat message schemas

Check out the live app here:
https://saranyu-docuverse.streamlit.app/
