import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import json
from datetime import datetime
import logging

# ===== Logging setup =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== Load environment variables =====
load_dotenv()

# ===== App initialization =====
app = FastAPI(title="SmartSense AI Real Estate Platform")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Folders =====
TMP_DIR = Path("tmp_parse")
TMP_DIR.mkdir(exist_ok=True, parents=True)
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True, parents=True)

# ===== Initialize Mock Orchestrator =====
try:
    # Try to import the real orchestrator first
    from orchestrator import AgentOrchestrator
    
    # If successful, try to initialize it
    try:
        agent_config = {}  # Add your config here
        orchestrator = AgentOrchestrator(agent_config)
        logger.info("Real orchestrator initialized successfully.")
        ORCHESTRATOR_TYPE = "real"
    except Exception as e:
        logger.warning(f"Could not initialize real orchestrator: {e}")
        raise ImportError("Falling back to mock")
        
except ImportError:
    # Fall back to mock orchestrator
    try:
        import sys
        sys.path.append('/mnt/user-data/outputs')
        from mock_orchestrator import AgentOrchestrator
        orchestrator = AgentOrchestrator()
        logger.info("Mock orchestrator initialized successfully.")
        ORCHESTRATOR_TYPE = "mock"
    except Exception as e:
        logger.error(f"Could not initialize any orchestrator: {e}")
        orchestrator = None
        ORCHESTRATOR_TYPE = "none"

# ===== Endpoints =====
@app.get("/")
async def root():
    return {
        "status": "running", 
        "message": "SmartSense AI Real Estate Platform",
        "orchestrator": ORCHESTRATOR_TYPE
    }

@app.get("/health")
async def health_check():
    services_status = {
        "orchestrator": orchestrator is not None,
        "orchestrator_type": ORCHESTRATOR_TYPE
    }
    
    return {
        "status": "healthy" if orchestrator else "limited",
        "timestamp": datetime.now().isoformat(),
        "services": services_status
    }

# ===== Main Chat endpoint =====
@app.post("/chat")
async def chat_endpoint(query: str = Form(...), user_id: str = Form(...)):
    """
    Main chat endpoint that processes queries through the orchestrator
    """
    try:
        if not orchestrator:
            return {
                "success": False,
                "message": "Sorry, the AI system is currently unavailable. Please try again later.",
                "data": {},
                "error": "orchestrator_unavailable"
            }
        
        logger.info(f"Processing query: '{query}' for user: {user_id}")
        
        # Process query through orchestrator
        result = await orchestrator.process_query(query, user_id)
        
        if result.get("success"):
            # Extract data for response
            response_data = result.get("data", {})
            message = result.get("message", "Query processed successfully")
            
            # Format response based on data content
            if "properties" in response_data:
                prop_count = len(response_data["properties"])
                if prop_count > 0:
                    message = f"Found {prop_count} properties. " + message
                else:
                    message = "No properties found matching your criteria. Try adjusting your search."
            
            if "renovation_estimate" in response_data:
                renovation_data = response_data["renovation_estimate"]
                total_cost = renovation_data.get("total_cost", 0)
                timeline = renovation_data.get("timeline_weeks", 0)
                if total_cost > 0:
                    message += f" Renovation estimate: ₹{total_cost:,}"
                    if timeline > 0:
                        message += f" (Timeline: {timeline} weeks)"
            
            if "market_data" in response_data:
                market_data = response_data["market_data"]
                avg_price = market_data.get("avg_price_sqft", 0)
                if avg_price > 0:
                    message += f" Current market rate: ₹{avg_price:,}/sqft"
            
            return {
                "success": True,
                "message": message,
                "data": response_data,
                "citations": result.get("citations", []),
                "metadata": result.get("metadata", {}),
                "execution_time": result.get("metadata", {}).get("execution_time", 0)
            }
        else:
            return {
                "success": False,
                "message": result.get("message", "Query processing failed"),
                "data": {},
                "error": result.get("error_type", "processing_error")
            }
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return {
            "success": False,
            "message": f"An error occurred: {str(e)}",
            "data": {},
            "error": "internal_error"
        }

# ===== Simple endpoints for basic functionality =====

@app.post("/search")
async def search_endpoint(
    location: str = Form(None),
    bhk: int = Form(None),
    property_type: str = Form(None),
    min_price: int = Form(0),
    max_price: int = Form(0),
    limit: int = Form(10)
):
    """Simple property search endpoint"""
    try:
        # Build a search query
        query_parts = []
        if bhk:
            query_parts.append(f"{bhk}BHK")
        if location:
            query_parts.append(f"in {location}")
        if property_type:
            query_parts.append(property_type)
        if min_price or max_price:
            if min_price and max_price:
                query_parts.append(f"between {min_price} and {max_price}")
            elif max_price:
                query_parts.append(f"under {max_price}")
        
        query = " ".join(query_parts) if query_parts else "properties"
        
        # Use orchestrator if available
        if orchestrator:
            result = await orchestrator.process_query(query, "search_user")
            if result.get("success"):
                properties = result.get("data", {}).get("properties", [])
                return {
                    "success": True,
                    "query": query,
                    "properties": properties[:limit],
                    "count": len(properties),
                    "filters_applied": {
                        "location": location,
                        "bhk": bhk,
                        "property_type": property_type,
                        "price_range": [min_price, max_price] if min_price or max_price else None
                    }
                }
        
        # Fallback mock response
        return {
            "success": True,
            "query": query,
            "properties": [],
            "count": 0,
            "message": "Search functionality requires full system initialization"
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Search failed: {str(e)}"})

@app.post("/generate-report")
async def generate_report_endpoint(
    report_type: str = Form(...),
    data: str = Form(...),
    user_id: str = Form(...)
):
    """Generate report endpoint"""
    try:
        # Parse input data
        try:
            parsed_data = json.loads(data)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON data"})
        
        if orchestrator:
            # Use orchestrator for report generation
            result = await orchestrator.generate_report_from_data(report_type, parsed_data, user_id)
            return result
        else:
            # Simple fallback
            return {
                "success": True,
                "report": {
                    "report_type": report_type,
                    "download_link": f"/download-pdf/sample_report.pdf",
                    "message": "Report generation requires full system initialization"
                },
                "message": "Report placeholder generated"
            }
    
    except Exception as e:
        logger.error(f"Report generation error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Report generation failed: {str(e)}"})

@app.post("/parse-floorplan")
async def parse_floorplan_endpoint(file: UploadFile = File(...), visualize: bool = Form(False)):
    """Simple floorplan parsing endpoint"""
    
    # Save uploaded file
    tmp_path = TMP_DIR / file.filename
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    
    # Mock response (replace with actual computer vision when available)
    response = {
        "parsed": {
            "rooms": 2,
            "halls": 1,
            "kitchens": 1,
            "bathrooms": 2,
            "rooms_detail": [
                {"label": "Bedroom", "count": 2, "approx_area": None},
                {"label": "Hall", "count": 1, "approx_area": None},
                {"label": "Kitchen", "count": 1, "approx_area": None},
                {"label": "Bathroom", "count": 2, "approx_area": None}
            ]
        },
        "detections": [
            {"label": "bedroom", "score": 0.92, "bbox": [100, 100, 200, 200]},
            {"label": "kitchen", "score": 0.88, "bbox": [250, 150, 350, 250]},
            {"label": "bathroom", "score": 0.85, "bbox": [300, 300, 380, 380]}
        ],
        "message": "Mock floorplan analysis - Replace with actual computer vision model"
    }
    
    return JSONResponse(content=response)

@app.post("/ingest")
async def ingest_data(background_tasks: BackgroundTasks, excel_file: UploadFile = File(...), images_dir: str = Form(...), certs_dir: str = Form(None)):
    """Data ingestion endpoint"""
    
    # Save uploaded file
    excel_path = TMP_DIR / excel_file.filename
    with open(excel_path, "wb") as f:
        f.write(await excel_file.read())
    
    # Mock ETL process
    logger.info(f"Mock ETL process started for {excel_file.filename}")
    
    return {
        "success": True, 
        "message": f"ETL process simulated for {excel_file.filename}. Full ETL requires database setup.",
        "file": excel_file.filename
    }

@app.get("/download-pdf/{filename}")
async def download_pdf(filename: str):
    """Download PDF files"""
    
    # Check different directories
    for directory in [TMP_DIR, REPORTS_DIR]:
        path = directory / filename
        if path.exists():
            return FileResponse(path, media_type="application/pdf", filename=filename)
    
    # Create a simple PDF if not found
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    pdf_path = TMP_DIR / filename
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "SmartSense AI Report")
    c.drawString(100, 730, f"Report: {filename}")
    c.drawString(100, 710, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, 680, "This is a placeholder report.")
    c.drawString(100, 660, "Full functionality requires complete system setup.")
    c.save()
    
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename)

@app.get("/download-image/{filename}")
async def download_image(filename: str):
    """Download image files"""
    path = TMP_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "File not found"})
    return FileResponse(path, media_type="image/png", filename=filename)

# ===== Run server =====
if __name__ == "__main__":
    print(f"🚀 Starting SmartSense AI Backend with {ORCHESTRATOR_TYPE} orchestrator")
    uvicorn.run("simple_backend:app", host="0.0.0.0", port=8001, reload=True)