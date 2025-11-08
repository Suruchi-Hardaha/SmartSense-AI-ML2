# app.py - FastAPI app for ingestion and single-image parse
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
import uvicorn
import traceback

# load .env
load_dotenv()

# app config from env
PG_HOST = os.getenv("PG_HOST")
PG_PORT = os.getenv("PG_PORT")
PG_DB   = os.getenv("PG_DB")
PG_USER = os.getenv("PG_USER")
PG_PASS = os.getenv("PG_PASS")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", None)
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "smartsense-properties")

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "notebooks/task1/floorplan_model_weights.pth")
USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

from etl import get_pg_engine, create_properties_table, init_pinecone, run_etl_from_excel
from sentence_transformers import SentenceTransformer
import pinecone
from parse_floorplan import create_model, parse_floorplan  # your file; must exist

import torch

app = FastAPI(title="SmartSense ETL API")

# global resources
PG_ENGINE = None
PINECONE_INDEX_OBJ = None
EMBED_MODEL = None
DETECT_MODEL = None
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def startup_event():
    global PG_ENGINE, PINECONE_INDEX_OBJ, EMBED_MODEL, DETECT_MODEL
    # Postgres
    PG_ENGINE = get_pg_engine(PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB)
    create_properties_table(PG_ENGINE)

    # embedding model
    print("[startup] loading embedding model...")
    EMBED_MODEL = SentenceTransformer(os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))

    # pinecone init and index
    print("[startup] initializing Pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
    if PINECONE_INDEX not in pinecone.list_indexes():
        # dimension from embedder
        dim = EMBED_MODEL.get_sentence_embedding_dimension()
        pinecone.create_index(PINECONE_INDEX, dimension=dim)
    PINECONE_INDEX_OBJ = pinecone.Index(PINECONE_INDEX)

    # detection model loaded via parse_floorplan.create_model
    print("[startup] loading detection model...")
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "9"))  # ensure this matches training
    DETECT_MODEL = create_model(NUM_CLASSES)
    DETECT_MODEL.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    DETECT_MODEL.to(DEVICE)
    DETECT_MODEL.eval()
    print("[startup] ready.")


@app.post("/ingest")
async def ingest_excel(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    images_dir: str = Form(...),
    certs_dir: str = Form(None),
    conf_thresh: float = Form(0.6)
):
    """
    Upload an Excel file (multipart form) and specify where images/certs are located.
    This enqueues a background task that runs the ETL and returns immediately.
    """
    tmp_dir = Path("/tmp/smart_sense_ingest")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # schedule background run
    def bg_run():
        try:
            run_etl_from_excel(
                excel_path=str(tmp_path),
                images_dir=images_dir,
                certs_dir=certs_dir,
                engine=PG_ENGINE,
                pinecone_index=PINECONE_INDEX_OBJ,
                embed_model=EMBED_MODEL,
                parse_fn=parse_floorplan,
                model=DETECT_MODEL,
                device=DEVICE,
                conf_thresh=conf_thresh
            )
        except Exception as e:
            print("ETL bg job failed:", e)
            traceback.print_exc()

    background_tasks.add_task(bg_run)
    return {"status": "ingest started", "file": file.filename}


@app.post("/parse-floorplan")
async def parse_floorplan_endpoint(file: UploadFile = File(...), conf_thresh: float = 0.6, visualize: bool = False):
    """
    Debug endpoint: upload a single image (multipart) and get parsed JSON + optionally visualized annotated image.
    """
    tmp_dir = Path("/tmp/smart_sense_parse")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        parsed_json, detections = parse_floorplan(str(tmp_path), DETECT_MODEL, device=DEVICE, conf_thresh=conf_thresh)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    response = {"parsed": parsed_json, "detections": detections}

    # optionally produce visualization image (draw boxes)
    if visualize:
        try:
            from PIL import Image, ImageDraw, ImageFont
            img = Image.open(tmp_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            # simple font
            for d in detections:
                bbox = d.get("bbox")
                label = d.get("label")
                score = d.get("score", 0)
                if bbox:
                    x1, y1, x2, y2 = bbox
                    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
                    draw.text((x1, max(0, y1-12)), f"{label}:{score:.2f}")
            out_path = tmp_path.with_name(f"vis_{file.filename}")
            img.save(out_path)
            return FileResponse(str(out_path), media_type="image/jpeg")
        except Exception as e:
            # if visualization fails, still return parsed JSON
            response["visualize_error"] = str(e)

    return JSONResponse(content=response)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
