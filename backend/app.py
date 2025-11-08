import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
import uvicorn
import torch
from parse_floorplan import create_model, parse_floorplan  # your module
from io import BytesIO
from PIL import Image, ImageDraw
import base64
import gdown
from reportlab.pdfgen import canvas

load_dotenv()

# ======= ENVIRONMENT VARIABLES =======
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "floorplan_model_weights.pth")
USE_CUDA = os.getenv("USE_CUDA", "0") == "1"
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "9"))

DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

# ======= APP INITIALIZATION =======
app = FastAPI(title="SmartSense Floorplan Parser")

# ======= TEMP & MODEL PATHS =======
TMP_DIR = Path("tmp_parse")
TMP_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ======= HELPER FUNCTIONS =======
def get_model_weights(model_path_or_url: str) -> str:
    """Return local path to model weights, downloading from Google Drive if needed."""
    model_path = MODEL_DIR / "floorplan_model_weights.pth"
    if Path(model_path_or_url).exists():
        return str(model_path_or_url)
    if "drive.google.com" in model_path_or_url:
        # Extract file ID
        if "file/d/" in model_path_or_url:
            file_id = model_path_or_url.split("file/d/")[1].split("/")[0]
        elif "id=" in model_path_or_url:
            file_id = model_path_or_url.split("id=")[1]
        else:
            raise ValueError("Cannot parse Google Drive file ID from URL")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
        return str(model_path)
    raise FileNotFoundError(f"Model path does not exist: {model_path_or_url}")

# ======= LOAD MODEL =======
DETECT_MODEL = create_model(NUM_CLASSES)
local_model_path = get_model_weights(MODEL_WEIGHTS)
DETECT_MODEL.load_state_dict(torch.load(local_model_path, map_location=DEVICE))
DETECT_MODEL.to(DEVICE)
DETECT_MODEL.eval()

# ======= PDF REPORT GENERATOR =======
def create_pdf_report(filename: str, parsed_data: dict, detections: list):
    pdf_path = TMP_DIR / filename
    c = canvas.Canvas(str(pdf_path))
    c.setFont("Helvetica", 12)
    c.drawString(50, 800, "SmartSense Floorplan Report")
    c.drawString(50, 780, "Parsed JSON Data:")
    y = 760
    for k, v in parsed_data.items():
        c.drawString(60, y, f"{k}: {v}")
        y -= 20
    c.drawString(50, y, "Detections:")
    y -= 20
    for det in detections:
        c.drawString(60, y, f"{det.get('label')} - score: {det.get('score')}")
        y -= 15
        if y < 50:
            c.showPage()
            y = 800
    c.save()
    return pdf_path

# ======= ENDPOINTS =======
@app.post("/parse-floorplan")
async def parse_floorplan_endpoint(
    file: UploadFile = File(...),
    visualize: bool = Form(False),
):
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

    # ===== Visualization =====
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

    # ===== PDF Report =====
    pdf_filename = f"{file.filename.split('.')[0]}_report.pdf"
    pdf_path = create_pdf_report(pdf_filename, parsed_json, detections)
    response["report_pdf_path"] = str(pdf_path)
    response["report_pdf_name"] = pdf_filename

    return JSONResponse(content=response)


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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)
