import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64
import json
from datetime import datetime

# Backend configuration
BACKEND_URL = "http://127.0.0.1:8001"

# Page configuration
st.set_page_config(
    page_title="SmartSense AI Real Estate",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        word-wrap: break-word;
    }
    .user-message {
        background-color: #d9edf7;
        color: #000000;
        text-align: right;
    }
    .assistant-message {
        background-color: #f2f2f2;
        color: #000000;
    }
    .property-card {
        border: 1px solid #ccc;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .json-output {
        background-color: #f4f4f4;
        padding: 0.8rem;
        border-radius: 0.4rem;
        overflow-x: auto;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Sidebar navigation
with st.sidebar:
    st.markdown("## SmartSense AI")
    page = st.radio(
        "Navigation",
        ["Chat Assistant", "Property Search", "Floorplan Parser", "Data Ingestion", "Reports"]
    )
    st.markdown("---")
    st.text_input("User ID", value=st.session_state.user_id, key="user_id_input")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared.")

st.markdown('<h1 class="main-header">SmartSense AI Real Estate Platform</h1>', unsafe_allow_html=True)

# --- Chat Assistant ---
if page == "Chat Assistant":
    st.markdown('<h2 class="sub-header">Intelligent Property Assistant</h2>', unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            role_class = "user-message" if message["role"]=="user" else "assistant-message"
            st.markdown(f'<div class="chat-message {role_class}">{message["content"]}</div>', unsafe_allow_html=True)
            
            if "data" in message:
                st.markdown('<div class="json-output">', unsafe_allow_html=True)
                st.json(message["data"])
                st.markdown('</div>', unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4,1])
        with col1:
            user_input = st.text_input("Ask about properties...", placeholder="Search or query details")
        with col2:
            submit = st.form_submit_button("Send")
    
    if submit and user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        with st.spinner("Processing..."):
            try:
                response = requests.post(f"{BACKEND_URL}/chat", data={"query": user_input, "user_id": st.session_state.user_id})
                if response.status_code == 200:
                    result = response.json()
                    assistant_message = {
                        "role":"assistant",
                        "content": result.get("message","No response"),
                        "data": result.get("data", {})
                    }
                    st.session_state.messages.append(assistant_message)
                    st.experimental_rerun()
                else:
                    st.error(f"Error from backend: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# --- Floorplan Parser ---
elif page == "Floorplan Parser":
    st.markdown('<h2 class="sub-header">Floorplan Parser</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Floorplan", type=["jpg","jpeg","png"])
    visualize = st.checkbox("Visualize Detections")
    
    if uploaded_file and st.button("Parse Floorplan"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Floorplan", use_container_width=True)
        
        with st.spinner("Parsing..."):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            data = {"visualize": visualize}
            resp = requests.post(f"{BACKEND_URL}/parse-floorplan", files=files, data=data)
            
            if resp.status_code == 200:
                result = resp.json()
                with col2:
                    st.subheader("Parsed Output")
                    st.markdown('<div class="json-output">', unsafe_allow_html=True)
                    st.json(result.get("parsed", {}))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.subheader("Top Room Detections")
                    for d in result.get("detections", [])[:5]:
                        st.write(f"{d['label']}: {d['score']:.2f}")
                
                if visualize and result.get("visualized_image_base64"):
                    img = Image.open(BytesIO(base64.b64decode(result["visualized_image_base64"])))
                    st.subheader("Visualized Floorplan")
                    st.image(img, use_container_width=True)
                
                if result.get("report_pdf_name"):
                    st.markdown(f"[Download PDF Report]({BACKEND_URL}/download-pdf/{result['report_pdf_name']})")
            else:
                st.error("Failed to parse floorplan")

# --- Reports ---
elif page == "Reports":
    st.markdown('<h2 class="sub-header">Generate Reports</h2>', unsafe_allow_html=True)
    
    report_type = st.selectbox("Report Type", ["property_analysis", "market_research", "comparison", "renovation", "investment"])
    
    data_input = st.text_area("Report Data (JSON)", height=200, value=json.dumps({"properties": [], "location": "Bangalore"}, indent=2))
    
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            try:
                response = requests.post(f"{BACKEND_URL}/generate-report", data={
                    "report_type": report_type,
                    "data": data_input,
                    "user_id": st.session_state.user_id
                })
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success") and result.get("report"):
                        st.success(f"Report generated: {report_type}")
                        st.markdown(f"[Download PDF]({BACKEND_URL}{result['report']['download_link']})")
                    else:
                        st.error("Report generation failed. Check JSON input structure.")
                else:
                    st.error("Failed to generate report")
            except Exception as e:
                st.error(f"Error: {str(e)}")
