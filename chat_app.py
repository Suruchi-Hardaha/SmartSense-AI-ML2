# chat_app.py
import os
import json
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from sqlalchemy import text
from collections import defaultdict
from pinecone import Pinecone
from fastapi.responses import JSONResponse

# Load environment variables if using .env
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Environment / Global setup
# ----------------------------
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "admin")
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DB = os.getenv("PG_DB", "smartsense")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "smartsense-properties")

# Pinecone init
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX)

# Placeholder SQLAlchemy engine
from sqlalchemy import create_engine
PG_ENGINE = create_engine(f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/{PG_DB}", future=True)

# Memory component per user_id
user_memory: Dict[str, dict] = defaultdict(dict)

# FastAPI app
app = FastAPI(title="SmartSense Chat API")

# ----------------------------
# Agent Implementations
# ----------------------------
def structured_data_agent(query: str, user_id: str):
    """Executes SQL queries safely against Postgres."""
    try:
        stmt = text(query)
        result = PG_ENGINE.execute(stmt)
        rows = [dict(row._mapping) for row in result.fetchall()]
        return {"agent": "StructuredDataAgent", "response": rows}
    except Exception as e:
        return {"agent": "StructuredDataAgent", "response": f"SQL Error: {str(e)}"}

def rag_agent(query: str, user_id: str):
    """Fetch info from Pinecone vector index (RAG)."""
    try:
        # Query vector using embedding model placeholder
        # For simplicity, assume query is property_id
        res = pinecone_index.fetch(ids=[query])
        vectors = res.vectors
        if not vectors:
            return {"agent": "RAGAgent", "response": "No relevant info found."}
        vec = list(vectors.values())[0]
        metadata = vec.metadata
        return {"agent": "RAGAgent", "response": metadata}
    except Exception as e:
        return {"agent": "RAGAgent", "response": f"RAG fetch error: {str(e)}"}

def renovation_agent(query: str, user_id: str):
    """Estimate renovation cost based on parsed_json or size in sqft."""
    # Dummy estimation logic
    sqft = 1700  # placeholder
    cost_per_sqft = 500
    total_cost = sqft * cost_per_sqft
    return {"agent": "RenovationAgent", "response": f"Estimated renovation cost for ~{sqft} sqft: {total_cost} INR"}

def report_agent(query: str, user_id: str):
    """Generate reports - placeholder."""
    return {"agent": "ReportAgent", "response": "Report generated: [PDF downloadable link placeholder]"}

def query_router(message: str, user_id: str):
    """Decide which agent(s) to call based on intent keywords."""
    # Simple keyword-based routing for demonstration
    if "renovation" in message.lower():
        return [renovation_agent]
    elif "report" in message.lower():
        return [report_agent]
    elif message.upper().startswith("PROP-"):  # property lookup
        return [rag_agent]
    elif "list properties" in message.lower():
        return [lambda msg, uid: structured_data_agent("SELECT property_id, title, location, price FROM properties LIMIT 5;", uid)]
    else:
        # fallback to RAG
        return [rag_agent]

# ----------------------------
# WebSocket Endpoint
# ----------------------------
@app.websocket("/chat")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                user_id = msg.get("user_id", "anonymous")
                message = msg.get("message", "")
            except Exception:
                await ws.send_text(json.dumps({"error": "Invalid message format, expected JSON with user_id and message"}))
                continue

            # Save message in memory
            if "history" not in user_memory[user_id]:
                user_memory[user_id]["history"] = []
            user_memory[user_id]["history"].append({"user": message})

            # Decide agents to call
            agents_to_call = query_router(message, user_id)
            responses = []
            for agent in agents_to_call:
                agent_resp = agent(message, user_id)
                responses.append(agent_resp)
                user_memory[user_id]["history"].append(agent_resp)

            # Send combined response
            await ws.send_text(json.dumps({"responses": responses}))

    except WebSocketDisconnect:
        print(f"WebSocket disconnected")

