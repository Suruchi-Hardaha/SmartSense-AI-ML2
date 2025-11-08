# agents.py
import json
from typing import List, Dict, Any
import random

class SessionMemory:
    """
    Store per-user conversation history and agent context.
    """
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def init_session(self, user_id: str):
        if user_id not in self.sessions:
            self.sessions[user_id] = {
                "conversation": [],
                "preferences": {},
                "agent_context": {}
            }

    def add_message(self, user_id: str, role: str, content: str):
        self.init_session(user_id)
        self.sessions[user_id]["conversation"].append({
            "role": role,
            "content": content
        })

    def update_agent_context(self, user_id: str, agent_name: str, context: dict):
        self.init_session(user_id)
        self.sessions[user_id]["agent_context"][agent_name] = context

    def get_session(self, user_id: str):
        self.init_session(user_id)
        return self.sessions[user_id]


# ---------------- Query Router ----------------
def route_query(user_message: str) -> List[str]:
    """
    Simple intent detection.
    Returns list of agent names to invoke.
    """
    message = user_message.lower()
    agents = []

    if any(w in message for w in ["find", "list", "2bhk", "3bhk", "price", "location"]):
        agents.append("StructuredDataAgent")
    if any(w in message for w in ["report", "summary", "pdf"]):
        agents.append("ReportAgent")
    if any(w in message for w in ["renovation", "estimate", "cost"]):
        agents.append("RenovationAgent")
    if any(w in message for w in ["details", "info", "describe", "document"]):
        agents.append("RAGAgent")

    if not agents:
        agents.append("RAGAgent")  # fallback
    return agents


# ---------------- Planner / Task Decomposer ----------------
def plan_tasks(user_message: str) -> List[str]:
    """
    Split complex queries into sub-tasks.
    For now, just return the user message itself.
    """
    return [user_message]


# ---------------- Structured Data Agent ----------------
def structured_data_agent(message: str, pg_engine) -> str:
    """
    Execute SQL queries on PostgreSQL based on message.
    """
    # Very simple keyword-based query
    message = message.lower()
    sql = "SELECT property_id, title, location, price FROM properties LIMIT 5;"
    try:
        with pg_engine.begin() as conn:
            result = conn.execute(sql)
            rows = result.fetchall()
            if rows:
                return "\n".join([f"{r.property_id} | {r.title} | {r.location} | {r.price}" for r in rows])
            else:
                return "No properties found."
    except Exception as e:
        return f"SQL Error: {e}"


# ---------------- RAG Agent ----------------
def rag_agent(message: str, pinecone_index, embed_model) -> str:
    """
    Retrieve similar properties / documents from Pinecone vector DB.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    query_vector = embed_model.encode(message).tolist()
    results = pinecone_index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True
    )
    if results.matches:
        answers = []
        for m in results.matches:
            meta = m.metadata
            answers.append(f"{meta.get('title')} - {meta.get('location')} (Price: {meta.get('price')})")
        return "RAG Results:\n" + "\n".join(answers)
    return "No relevant info found in RAG index."


# ---------------- Renovation Estimation Agent ----------------
def renovation_agent(message: str) -> str:
    """
    Simple rule-based cost estimation.
    """
    base_cost = 500  # per sq ft
    size_sqft = random.randint(800, 2000)  # placeholder
    est_cost = base_cost * size_sqft
    return f"Estimated renovation cost for ~{size_sqft} sqft: {est_cost} INR"


# ---------------- Report Generation Agent ----------------
def report_agent(message: str) -> str:
    """
    Placeholder report generation.
    """
    return "Report generated: [PDF downloadable link placeholder]"

