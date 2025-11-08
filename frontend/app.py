import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import base64

BACKEND_URL = "http://127.0.0.1:8001"

st.title("SmartSense Floorplan Parser & Chatbot")

# --- FLOORPLAN PARSE ---
st.header("Parse Floorplan")
uploaded_file = st.file_uploader("Upload Floorplan Image", type=["jpg","png","jpeg"])
visualize = st.checkbox("Visualize Detections")

if uploaded_file:
    with st.spinner("Processing floorplan..."):
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        data = {"visualize": visualize}
        resp = requests.post(f"{BACKEND_URL}/parse-floorplan", files=files, data=data)
    if resp.status_code == 200:
        result = resp.json()
        st.subheader("Parsed JSON Data")
        st.json(result.get("parsed", {}))
        st.subheader("Detections")
        st.json(result.get("detections", []))

        img_base64 = result.get("visualized_image_base64")
        vis_path = result.get("visualized_image_path")
        if img_base64:
            st.subheader("Visualized Floorplan")
            img = Image.open(BytesIO(base64.b64decode(img_base64)))
            st.image(img, use_column_width=True)
            if vis_path:
                filename = vis_path.split("/")[-1]
                download_url = f"{BACKEND_URL}/download-image/{filename}"
                st.markdown(f"[Download Visualized Image]({download_url})")
    else:
        st.error(resp.text)

# --- CHAT ---
st.header("Chat & Generate Report")
user_query = st.text_input("Enter your query:")
if st.button("Generate Report") and user_query:
    with st.spinner("Generating report..."):
        resp = requests.post(f"{BACKEND_URL}/chat", data={"user_query": user_query})
    if resp.status_code == 200:
        result = resp.json()
        st.success(result.get("message"))
        st.markdown(f"[Download Report]({BACKEND_URL}{result.get('download_link')})")
    else:
        st.error(resp.text)
