# app.py - FastAPI app for ingestion and single-image parse
import os
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv
import uvicorn
import traceback
import torch
from sentence_transformers import SentenceTransformer
from parse_floorplan import create_model, parse_floorplan
from etl import get_pg_engine, create_properties_table, run_etl_from_excel
from pinecone import Pinecone, ServerlessSpec
import gdown

from fastapi import WebSocket, WebSocketDisconnect
from agents import SessionMemory, route_query, plan_tasks, structured_data_agent, rag_agent, renovation_agent, report_agent

# Initialize memory
memory = SessionMemory()


# Load environment variables
load_dotenv()

# Environment variables
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "smartsense")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "admin")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-aws")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "smartsense-properties")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "floorplan_model_weights.pth")
USE_CUDA = os.getenv("USE_CUDA", "0") == "1"

app = FastAPI(title="SmartSense ETL API")

# Global variables
PG_ENGINE = None
PINECONE_INDEX_OBJ = None
EMBED_MODEL = None
DETECT_MODEL = None
DEVICE = torch.device("cuda" if USE_CUDA and torch.cuda.is_available() else "cpu")


def download_model_if_needed():
    """Download model weights if MODEL_WEIGHTS is a Google Drive link."""
    model_path = Path("floorplan_model_weights.pth")
    if os.path.exists(model_path):
        print("[model] Found existing model weights locally.")
        return str(model_path)

    if "drive.google.com" in MODEL_WEIGHTS:
        print("[model] Downloading model weights from Google Drive...")
        # Extract file id
        if "file/d/" in MODEL_WEIGHTS:
            file_id = MODEL_WEIGHTS.split("file/d/")[1].split("/")[0]
        elif "id=" in MODEL_WEIGHTS:
            file_id = MODEL_WEIGHTS.split("id=")[1]
        else:
            raise ValueError("Could not extract Google Drive file ID from MODEL_WEIGHTS URL")

        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
        print("[model] Model weights downloaded successfully.")
    else:
        print("[model] Using local model path:", MODEL_WEIGHTS)
        model_path = Path(MODEL_WEIGHTS)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    return str(model_path)


@app.on_event("startup")
def startup_event():
    global PG_ENGINE, PINECONE_INDEX_OBJ, EMBED_MODEL, DETECT_MODEL

    # PostgreSQL setup
    PG_ENGINE = get_pg_engine(PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB)
    create_properties_table(PG_ENGINE)

    # Embedding model
    print("[startup] Loading embedding model...")
    EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

    # Pinecone initialization (new SDK)
    print("[startup] Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if PINECONE_INDEX not in existing_indexes:
        print(f"[pinecone] Creating new index: {PINECONE_INDEX}")
        dim = EMBED_MODEL.get_sentence_embedding_dimension()
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    else:
        print(f"[pinecone] Using existing index: {PINECONE_INDEX}")

    PINECONE_INDEX_OBJ = pc.Index(PINECONE_INDEX)

    # Model setup
    print("[startup] Loading detection model...")
    model_path = download_model_if_needed()
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", "9"))
    DETECT_MODEL = create_model(NUM_CLASSES)
    DETECT_MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
    DETECT_MODEL.to(DEVICE)
    DETECT_MODEL.eval()

    print("[startup] Application ready!")


@app.post("/ingest")
async def ingest_excel(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    images_dir: str = Form(...),
    certs_dir: str = Form(None),
    conf_thresh: float = Form(0.6),
):
    tmp_dir = Path("tmp_ingest")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

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
                conf_thresh=conf_thresh,
            )
        except Exception as e:
            print("ETL background task failed:", e)
            traceback.print_exc()

    background_tasks.add_task(bg_run)
    return {"status": "ingest started", "file": file.filename}


@app.post("/parse-floorplan")
async def parse_floorplan_endpoint(
    file: UploadFile = File(...), conf_thresh: float = 0.6, visualize: bool = False
):
    tmp_dir = Path("tmp_parse")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        parsed_json, detections = parse_floorplan(str(tmp_path), DETECT_MODEL, device=DEVICE, conf_thresh=conf_thresh)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    response = {"parsed": parsed_json, "detections": detections}

    if visualize:
        try:
            from PIL import Image, ImageDraw
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
            out_path = tmp_path.with_name(f"vis_{file.filename}")
            img.save(out_path)
            return FileResponse(str(out_path), media_type="image/jpeg")
        except Exception as e:
            response["visualize_error"] = str(e)

    return JSONResponse(content=response)
@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            user_id = data.get("user_id")
            message = data.get("message")
            if not user_id or not message:
                await websocket.send_json({"error": "user_id and message required"})
                continue

            memory.init_session(user_id)
            memory.add_message(user_id, "user", message)

            # ----------------- Agent Routing -----------------
            agent_names = route_query(message)
            responses = []

            for agent_name in agent_names:
                if agent_name == "StructuredDataAgent":
                    res = structured_data_agent(message, PG_ENGINE)
                elif agent_name == "RAGAgent":
                    res = rag_agent(message, PINECONE_INDEX_OBJ, EMBED_MODEL)
                elif agent_name == "RenovationAgent":
                    res = renovation_agent(message)
                elif agent_name == "ReportAgent":
                    res = report_agent(message)
                else:
                    res = "Unknown agent."

                memory.update_agent_context(user_id, agent_name, {"last_response": res})
                responses.append(f"[{agent_name}]: {res}")

            final_response = "\n\n".join(responses)
            memory.add_message(user_id, "agent", final_response)

            await websocket.send_json({"response": final_response})

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for user: {user_id}")



if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
