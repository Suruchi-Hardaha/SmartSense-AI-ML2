import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import torch
from parse_floorplan import create_model, parse_floorplan
from io import BytesIO
from PIL import Image, ImageDraw
import base64
from reportlab.pdfgen import canvas
import json
from datetime import datetime
import logging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
import json


# ETL imports
from etl import run_etl_from_excel, get_pg_engine, init_pinecone
from sentence_transformers import SentenceTransformer

#  Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Load environment variables 
load_dotenv()

#  Environment variables 
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "floorplan_model_weights.pth")
USE_CUDA = os.getenv("USE_CUDA", "0") == "1"
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "9"))

# Database config
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "smartsense")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "admin")

# Vector DB & embeddings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "smartsense-properties")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

#  App initialization
app = FastAPI(title="SmartSense AI Real Estate Platform")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Folders
TMP_DIR = Path("tmp_parse")
TMP_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

#  Helper functions 
def get_model_weights(model_path_or_url: str) -> str:
    model_path = MODEL_DIR / "floorplan_model_weights.pth"
    if Path(model_path_or_url).exists():
        return str(model_path_or_url)
    if "drive.google.com" in model_path_or_url:
        if "file/d/" in model_path_or_url:
            file_id = model_path_or_url.split("file/d/")[1].split("/")[0]
        elif "id=" in model_path_or_url:
            file_id = model_path_or_url.split("id=")[1]
        else:
            raise ValueError("Cannot parse Google Drive file ID from URL")
        import gdown
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
        return str(model_path)
    raise FileNotFoundError(f"Model path does not exist: {model_path_or_url}")



def create_wrapped_pdf_report(filename: str, report_type: str, data: dict):
    pdf_path = f"./{filename}"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    margin = 50
    line_height = 12

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, height - 50, f"Report Type: {report_type}")

    # Prepare JSON string
    json_str = json.dumps(data, indent=2)
    lines = simpleSplit(json_str, "Helvetica", 10, width - 2 * margin)

    # Draw JSON line by line
    c.setFont("Helvetica", 10)
    y = height - 80
    for line in lines:
        if y < 50:  # Add new page if needed
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50
        c.drawString(margin, y, line)
        y -= line_height

    c.save()
    return pdf_path


def create_pdf_report(filename: str, parsed_data: dict, detections: list):
    """Generate PDF report for parsed floorplan or report data with wrapped text."""
    pdf_path = TMP_DIR / filename
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "SmartSense AI Report")
    
    y = height - 80
    c.setFont("Helvetica", 10)
    
    # Write parsed JSON nicely
    json_str = json.dumps(parsed_data, indent=2)
    lines = simpleSplit(json_str, "Helvetica", 10, width - 100)
    for line in lines:
        c.drawString(50, y, line)
        y -= 12
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 10)
    
    # Write detections
    if detections:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Detections:")
        y -= 20
        c.setFont("Helvetica", 10)
        for det in detections:
            text = f"{det.get('label', 'N/A')} - score: {det.get('score', 0):.2f}"
            lines = simpleSplit(text, "Helvetica", 10, width - 100)
            for line in lines:
                c.drawString(50, y, line)
                y -= 12
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 10)
    
    c.save()
    return pdf_path

#  Load detection model 
logger.info("Loading floorplan detection model...")
DETECT_MODEL = create_model(NUM_CLASSES)
local_model_path = get_model_weights(MODEL_WEIGHTS)
DETECT_MODEL.load_state_dict(torch.load(local_model_path, map_location=DEVICE))
DETECT_MODEL.to(DEVICE)
DETECT_MODEL.eval()
logger.info("Model loaded successfully.")

# Initialize DB and embedding model 
pg_engine = get_pg_engine(PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB)
pinecone_index = init_pinecone(PINECONE_API_KEY, PINECONE_INDEX, dim=384)
embed_model = SentenceTransformer(EMBED_MODEL)

#  Endpoints
@app.get("/")
async def root():
    return {"status": "running", "message": "SmartSense AI Real Estate Platform"}

@app.post("/parse-floorplan")
async def parse_floorplan_endpoint(file: UploadFile = File(...), visualize: bool = Form(False)):
    tmp_path = TMP_DIR / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    response = {}
    try:
        parsed_json, detections = parse_floorplan(str(tmp_path), DETECT_MODEL, device=DEVICE)
        response["parsed"] = parsed_json
        response["detections"] = detections
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    if visualize:
        try:
            img = Image.open(tmp_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            for d in detections:
                bbox = d.get("bbox")
                label = d.get("label")
                score = d.get("score", 0)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                    draw.text((x1, max(0, y1 - 12)), f"{label}:{score:.2f}")
            vis_filename = f"vis_{file.filename}"
            vis_path = TMP_DIR / vis_filename
            img.save(vis_path)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            response["visualized_image_base64"] = img_base64
            response["visualized_image_path"] = str(vis_path)
        except Exception as e:
            response["visualize_error"] = str(e)

    pdf_filename = f"{file.filename.split('.')[0]}_report.pdf"
    pdf_path = create_wrapped_pdf_report(pdf_filename, parsed_json, detections)
    response["report_pdf_name"] = pdf_filename
    response["report_pdf_path"] = str(pdf_path)

    return JSONResponse(content=response)

@app.post("/ingest")
async def ingest_data(background_tasks: BackgroundTasks, excel_file: UploadFile = File(...), images_dir: str = Form(...), certs_dir: str = Form(None)):
    excel_path = TMP_DIR / excel_file.filename
    with open(excel_path, "wb") as f:
        f.write(await excel_file.read())
    background_tasks.add_task(
        run_etl_from_excel,
        str(excel_path),
        images_dir,
        certs_dir,
        pg_engine,
        pinecone_index,
        embed_model,
        parse_floorplan,
        DETECT_MODEL,
        DEVICE,
        0.6
    )
    return {"success": True, "message": "ETL started", "file": excel_file.filename}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/download-image/{filename}")
async def download_image(filename: str):
    path = TMP_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path, media_type="image/png", filename=filename)

@app.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    path = TMP_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "PDF not found"})
    return FileResponse(path, media_type="application/pdf", filename=filename)

# Chat endpoint 
@app.post("/chat")
async def chat_endpoint(query: str = Form(...), user_id: str = Form(...)):
    message = f"Received your query: {query}"
    data = {}  # Optional: Add property search results
    return {"message": message, "data": data, "citations": []}

#  Search endpoint 
@app.post("/search")
async def search_endpoint(
    location: str = Form(None),
    bhk: int = Form(None),
    property_type: str = Form(None),
    min_price: int = Form(0),
    max_price: int = Form(0),
    limit: int = Form(10)
):
    query_parts = []
    if location:
        query_parts.append(location)
    if bhk:
        query_parts.append(f"{bhk}BHK")
    if property_type:
        query_parts.append(property_type)
    query = " ".join(query_parts) if query_parts else "property"
    vector = embed_model.encode(query, show_progress_bar=False).tolist()
    results = pinecone_index.query(vector=vector, top_k=limit, include_metadata=True)
    hits = [match.metadata for match in results.matches]
    return {"query": query, "properties": hits, "count": len(hits)}

# Generate Report endpoint 
@app.post("/generate-report")
async def generate_report_endpoint(
    report_type: str = Form(...),
    data: str = Form(...),
    user_id: str = Form(...)
):
    try:
        parsed_data = json.loads(data)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON data"})
    pdf_filename = f"{report_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf_path = TMP_DIR / pdf_filename
    c = canvas.Canvas(str(pdf_path))
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, f"Report Type: {report_type}")
    c.drawString(50, 780, "Data:")
    y = 760
    for k, v in parsed_data.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 20
        if y < 50:
            c.showPage()
            y = 800
    c.save()
    return {"success": True, "report": {"report_type": report_type, "download_link": f"/download-pdf/{pdf_filename}"}}

# Run server 
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
