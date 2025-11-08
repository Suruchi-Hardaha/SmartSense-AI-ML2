# etl.py -- helper functions for ETL ingestion (Postgres + Pinecone)
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import PyPDF2
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
import pinecone
from tqdm import tqdm

# load env externally (app will do python-dotenv)
def get_pg_engine(pg_user, pg_pass, pg_host, pg_port, pg_db):
    url = f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
    engine = create_engine(url, future=True)
    return engine

def create_properties_table(engine):
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS properties (
        id SERIAL PRIMARY KEY,
        property_id TEXT UNIQUE,
        title TEXT,
        long_description TEXT,
        location TEXT,
        price BIGINT,
        seller_type TEXT,
        listing_date TIMESTAMP,
        seller_contact TEXT,
        metadata_tags TEXT,
        image_file TEXT,
        parsed_json JSONB,
        certs_text TEXT,
        created_at TIMESTAMP DEFAULT now()
    );
    """
    with engine.begin() as conn:
        conn.execute(text(create_table_sql))

def upsert_property(engine, row: Dict[str,Any], parsed_json: Dict[str,Any], certs_text: str):
    upsert_sql = text("""
    INSERT INTO properties(property_id, title, long_description, location, price,
        seller_type, listing_date, seller_contact, metadata_tags, image_file, parsed_json, certs_text)
    VALUES(:property_id, :title, :long_description, :location, :price, :seller_type, :listing_date,
           :seller_contact, :metadata_tags, :image_file, :parsed_json::jsonb, :certs_text)
    ON CONFLICT (property_id) DO UPDATE SET
      title = EXCLUDED.title,
      long_description = EXCLUDED.long_description,
      location = EXCLUDED.location,
      price = EXCLUDED.price,
      seller_type = EXCLUDED.seller_type,
      listing_date = EXCLUDED.listing_date,
      seller_contact = EXCLUDED.seller_contact,
      metadata_tags = EXCLUDED.metadata_tags,
      image_file = EXCLUDED.image_file,
      parsed_json = EXCLUDED.parsed_json,
      certs_text = EXCLUDED.certs_text,
      created_at = now();
    """)
    with engine.begin() as conn:
        conn.execute(upsert_sql, {
            "property_id": str(row.get("property_id")),
            "title": row.get("title"),
            "long_description": row.get("long_description"),
            "location": row.get("location"),
            "price": int(row.get("price")) if pd.notna(row.get("price")) else None,
            "seller_type": row.get("seller_type"),
            "listing_date": row.get("listing_date"),
            "seller_contact": row.get("seller_contact"),
            "metadata_tags": row.get("metadata_tags"),
            "image_file": row.get("image_file"),
            "parsed_json": json.dumps(parsed_json),
            "certs_text": certs_text
        })

def extract_text_from_certs(cert_field: str, certs_dir: Optional[str]):
    """
    cert_field: pipe-separated filenames or URLs. If local files exist in certs_dir, parse PDFs with PyPDF2.
    Returns concatenated text or empty string.
    """
    if pd.isna(cert_field) or not cert_field:
        return ""
    pieces = []
    for token in str(cert_field).split("|"):
        token = token.strip()
        if not token:
            continue
        # look local
        if certs_dir:
            path = Path(certs_dir) / token
            if path.exists():
                try:
                    with open(path, "rb") as fh:
                        reader = PyPDF2.PdfReader(fh)
                        txt = []
                        for p in reader.pages:
                            extracted = p.extract_text() or ""
                            txt.append(extracted)
                        pieces.append("\n".join(txt))
                        continue
                except Exception as e:
                    pieces.append(f"[PDF_PARSE_ERROR:{token}]")
                    continue
        # fallback: just include filename or URL
        pieces.append(token)
    return "\n\n".join(pieces)

def init_pinecone(api_key: str, environment: Optional[str], index_name: str, dim: int):
    pinecone.init(api_key=api_key, environment=environment)
    idx_list = pinecone.list_indexes()
    if index_name not in idx_list:
        pinecone.create_index(index_name, dimension=dim)
    index = pinecone.Index(index_name)
    return index

def embed_text(embed_model: SentenceTransformer, text: str):
    if not text:
        return embed_model.encode("", show_progress_bar=False).tolist()
    emb = embed_model.encode(text, show_progress_bar=False)
    return emb.tolist()

def run_etl_from_excel(
    excel_path: str,
    images_dir: str,
    certs_dir: Optional[str],
    engine,
    pinecone_index,
    embed_model: SentenceTransformer,
    parse_fn,    # parse_fn(image_path, model, device, conf_thresh) -> (parsed_json, detections)
    model,
    device,
    conf_thresh: float = 0.6
):
    df = pd.read_excel(excel_path, engine="openpyxl")
    print(f"[ETL] loaded {len(df)} rows from {excel_path}")

    # For each row: parse image, insert to postgres, index to pinecone
    for _, row in tqdm(df.iterrows(), total=len(df), desc="ETL rows"):
        prop_id = str(row.get("property_id"))
        image_file = str(row.get("image_file"))
        image_path = Path(images_dir) / image_file
        parsed_json = {}
        detections = []
        if image_path.exists():
            try:
                parsed_json, detections = parse_fn(str(image_path), model, device=device, conf_thresh=conf_thresh)
            except Exception as e:
                print(f"[WARN] parse_floorplan failed for {image_path}: {e}")
                parsed_json = {}
        else:
            print(f"[WARN] image not found: {image_path}")

        # cert text
        certs_text = extract_text_from_certs(row.get("certificates", ""), certs_dir)

        # upsert to postgres
        upsert_property(engine, row, parsed_json, certs_text)

        # text to embed
        text_for_embed = " ".join(filter(None, [
            str(row.get("title") or ""),
            str(row.get("long_description") or ""),
            str(row.get("location") or ""),
            str(certs_text or ""),
            str(row.get("metadata_tags") or "")
        ]))
        vector = embed_text(embed_model, text_for_embed)

        # upsert to pinecone (id = property_id)
        metadata = {
            "property_id": prop_id,
            "title": row.get("title"),
            "location": row.get("location"),
            "price": row.get("price"),
            "image_file": image_file,
            "parsed_json": parsed_json
        }
        pinecone_index.upsert([(prop_id, vector, metadata)])

    print("[ETL] done")
