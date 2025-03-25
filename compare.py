
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import fitz  # PyMuPDF for PDF handling
import google.generativeai as genai
from PIL import Image

# Configure the API key
genai.configure(api_key=os.getenv("google_api_key"))

# Load the Gemini model
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Function to extract images from PDF
def extract_image_from_pdf(pdf_content):
    with fitz.open(stream=pdf_content, filetype="pdf") as doc:
        for page in doc:
            pixmap = page.get_pixmap()  # Render page as image
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            return img  # Extract only the first page image

# Function to compare two dashboard images
def get_gemini_comparison(img1, img2):
    prompt = (
        "Compare these two dashboard images based on the following parameters:\n"
        "1. **Text Similarity** (30% Weight): Identify changes such as text additions, reductions, format changes, or value replacements. Even minor changes should significantly impact the score.\n"
        "2. **Numerical Data Accuracy** (20% Weight): Detect differences in key metrics, KPIs, and numerical values. Even slight shifts in data values should notably reduce the similarity score.\n"
        "3. **Layout Structure** (10% Weight): Assess differences in element positioning, alignment, spacing, and visual organization. Slight adjustments should moderately impact the score.\n"
        "4. **Color Scheme Similarity** (10% Weight): Identify variations in color themes, additions, reductions, or subtle hue changes. Minor changes should reduce the score proportionately.\n"
        "5. **Graph Design** (20% Weight): Evaluate changes in graph types, axis labeling, data visualization, and design consistency. Minor adjustments should have a smaller impact on the score.\n"
        "6. **Font Style Consistency** (10% Weight): Assess variations in font type, size, or clarity. Slight changes should contribute minimally to the overall reduction in similarity.\n"
        "Assign the highest impact to text-related differences, followed by numerical data, layout structure, and other visual aspects.\n"
        "Provide individual similarity scores out of 100 for each parameter, along with a weighted overall similarity score."
    )
    response = model.generate_content([prompt, img1, img2])
    return response.text

# Streamlit UI configuration
st.set_page_config(page_title="Dashboard Comparison Tool")
st.header("Dashboard Comparison Application")

# File uploaders for PDF files
uploaded_file1 = st.file_uploader("Upload PDF File 1", type=["pdf", "png", "jpg", "jpeg"])
uploaded_file2 = st.file_uploader("Upload PDF File 2", type=["pdf", "png", "jpg", "jpeg"])

# Comparison logic
if st.button("Compare PDFs"):
    if uploaded_file1 and uploaded_file2:
        # Extract images from PDFs
        img1 = extract_image_from_pdf(uploaded_file1.read())
        img2 = extract_image_from_pdf(uploaded_file2.read())

        # Display uploaded images
        st.image(img1, caption="PDF 1 - Dashboard", use_column_width=True)
        st.image(img2, caption="PDF 2 - Dashboard", use_column_width=True)

        # Perform comparison
        comparison_result = get_gemini_comparison(img1, img2)

        # Display comparison result
        st.subheader("Comparison Results:")
        st.write(comparison_result)
    else:
        st.error("Please upload both PDF files for comparison.")
