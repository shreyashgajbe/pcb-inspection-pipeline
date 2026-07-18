import streamlit as st
from PIL import Image
import os
from google import genai

# --- CONFIGURATIONS & PATHS ---
IMAGE_DIR = "images"
DOCS_DIR = "docs"
KNOWLEDGE_BASE_FILE = os.path.join(DOCS_DIR, "pcb_inspection_guide.txt")


def inspect_pcb(image_path):
    # Initialize the Gemini client
    client = genai.Client()
    
    # Open the image using PIL
    raw_image = Image.open(image_path)
    
    prompt = """
You are an IPC Certified PCB Quality Inspector.
Analyze the uploaded PCB image. Your task is to inspect the PCB for manufacturing defects.

Return the result exactly in this text format:
PCB Status: [PASS or FAIL]
Detected Components: [List main components]
Detected Defects: [Name the primary defect, e.g., Missing Component, Solder Bridge, Misalignment, Short Circuit, None]
Defect Location: [e.g., Top Left, Center, Bottom Right, None]
Severity: [Low/Medium/High]
Confidence: [Low/Medium/High]

Do not guess. If uncertain, state that manual inspection is recommended.
"""

    # Call the cloud multi-modal vision model
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[raw_image, prompt]
    )
    
    return response.text

def query_knowledge_base(defect_name):
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        return "⚠️ Error: Reference manual not found."
        
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding="utf-8") as f:
        content = f.read()
        
    # Split manual sections by tag
    sections = content.split("[DEFECT: ")
    
    for section in sections:
        # Check if the text matches the model's detected defect name
        if defect_name.lower() in section.lower():
            return f"ℹ️ Standard IPC Repair Guide for [{defect_name.strip()}]:\n" + section.strip()
            
    return f"ℹ️ No specific standard procedures found in manual for: {defect_name}."


def parse_defect_type(output_text):
    """Helper utility to extract the defect string out of the LLaVA response"""
    defect = "None"
    for line in output_text.split("\n"):
        if "Detected Defects:" in line:
            defect = line.split("Detected Defects:")[1].strip()
    return defect


def run_inspection_pipeline(image_filename):
    image_path = os.path.join(IMAGE_DIR, image_filename)

    # 1. Run Vision AI Assessment
    vision_result = inspect_pcb(image_path)

    # 2. Extract Defect Entity & Check Knowledge Base (RAG)
    detected_defect = parse_defect_type(vision_result)

    repair_guide = "N/A (PCB Passed Inspection)"
    if "none" not in detected_defect.lower() and detected_defect != "":
        repair_guide = query_knowledge_base(detected_defect)

    # 3. Format the Final QA Report
    final_report = f"""==========================================
             PCB AI INSPECTION REPORT
==========================================
Target Image File: {image_filename}
------------------------------------------
[1. VISION INTERPRETATION RESULTS]
{vision_result}

------------------------------------------
[2. RAG GUIDELINES & REMEDIAL ACTIONS]
{repair_guide}
==========================================
"""
    # 4. Save report to disk
    os.makedirs("reports", exist_ok=True)
    report_path = f"reports/report_{os.path.splitext(image_filename)[0]}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"\n📁 Report successfully logged to: {report_path}")
    print(final_report)

    return final_report


# --- PAGE SETUP ---
st.set_page_config(page_title="PCB Inspector", page_icon="🔧", layout="centered")

st.title("🔧 PCB Missing Component Inspector")
st.write("Upload a PCB image to analyze it for defects using LLaVA.")

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("Choose a PCB image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image cleanly
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded PCB Image", use_container_width=True)
    
    # --- TRIGGER BUTTON ---
    if st.button("Run Inspection Pipeline"):
        with st.spinner("Analyzing image with LLaVA... Please wait..."):
            try:
                # Save the uploaded file temporarily so your backend pipeline can read it
                temp_dir = "images"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                image.save(temp_path)

                # Call your real function using the filename, just like your notebook did
                final_report = run_inspection_pipeline(uploaded_file.name)

                # Display the real output from LLaVA inside the app interface
                st.success("Inspection Complete!")
                st.subheader("Analysis Report")
                st.text_area("Log Output", final_report, height=400)

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")