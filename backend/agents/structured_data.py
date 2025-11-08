# agents/structured_data.py - Structured Data Agent

from typing import Dict, Any, Optional, List
from agents.base import BaseAgent, AgentResponse, AgentType
from sqlalchemy import create_engine, text, select, and_, or_
from sqlalchemy.orm import Session
import pandas as pd
import json
from datetime import datetime, timedelta

class StructuredDataAgent(BaseAgent):
    """
    Structured Data Agent - Runs SQL queries against PostgreSQL
    Handles property searches, filters, and aggregations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("StructuredData", AgentType.STRUCTURED_DATA, config)
        
        # Database connection
        self.engine = None
        if config and "db_config" in config:
            db_cfg = config["db_config"]
            self.engine = create_engine(
                f"postgresql+psycopg2://{db_cfg['user']}:{db_cfg['password']}@"
                f"{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
            )
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Execute SQL queries based on input parameters
        """
        try:
            query_type = input_data.get("query_type", "search")
            filters = input_data.get("filters", {})
            aggregations = input_data.get("aggregations", [])
            limit = input_data.get("limit", 10)
            
            if query_type == "search":
                results = await self._search_properties(filters, limit)
            elif query_type == "aggregate":
                results = await self._aggregate_properties(filters, aggregations)
            elif query_type == "detail":
                results = await self._get_property_details(input_data.get("property_id"))
            else:
                results = await self._custom_query(input_data.get("sql_query"))
            
            return AgentResponse(
                success=True,
                data=results,
                message=f"Retrieved {len(results) if isinstance(results, list) else 1} results",
                metadata={
                    "query_type": query_type,
                    "filters_applied": filters
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Database query failed: {str(e)}"
            )
    
    async def _search_properties(self, filters: Dict[str, Any], limit: int) -> List[Dict]:
        """Search properties with filters"""
        query = """
            SELECT 
                property_id,
                title,
                long_description,
                location,
                price,
                seller_type,
                listing_date,
                seller_contact,
                metadata_tags,
                image_file,
                parsed_json,
                certs_text
            FROM properties
            WHERE 1=1
        """
        
        params = {}
        
        # Apply filters
        if filters.get("location"):
            query += " AND LOWER(location) LIKE LOWER(:location)"
            params["location"] = f"%{filters['location']}%"
        
        if filters.get("bhk"):
            query += " AND parsed_json->>'rooms' = :bhk"
            params["bhk"] = str(filters["bhk"])
        
        if filters.get("budget"):
            query += " AND price <= :budget"
            params["budget"] = filters["budget"]
        
        if filters.get("property_type"):
            query += " AND LOWER(metadata_tags) LIKE LOWER(:property_type)"
            params["property_type"] = f"%{filters['property_type']}%"
        
        if filters.get("min_price"):
            query += " AND price >= :min_price"
            params["min_price"] = filters["min_price"]
        
        if filters.get("max_price"):
            query += " AND price <= :max_price"
            params["max_price"] = filters["max_price"]
        
        # Add sorting
        sort_by = filters.get("sort_by", "listing_date")
        sort_order = filters.get("sort_order", "DESC")
        query += f" ORDER BY {sort_by} {sort_order}"
        
        # Add limit
        query += f" LIMIT {limit}"
        
        # Execute query
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            
            # Convert to list of dicts
            properties = []
            for row in rows:
                prop = dict(row._mapping)
                # Parse JSON fields
                if prop.get("parsed_json"):
                    prop["parsed_json"] = json.loads(prop["parsed_json"]) if isinstance(prop["parsed_json"], str) else prop["parsed_json"]
                properties.append(prop)
            
            return properties
    
    async def _aggregate_properties(self, filters: Dict[str, Any], aggregations: List[str]) -> Dict[str, Any]:
        """Perform aggregations on property data"""
        results = {}
        
        # Base query with filters
        base_conditions = self._build_filter_conditions(filters)
        
        with self.engine.connect() as conn:
            # Count properties
            if "count" in aggregations:
                count_query = f"SELECT COUNT(*) as count FROM properties WHERE {base_conditions}"
                result = conn.execute(text(count_query), filters)
                results["count"] = result.scalar()
            
            # Average price
            if "avg_price" in aggregations:
                avg_query = f"SELECT AVG(price) as avg_price FROM properties WHERE {base_conditions}"
                result = conn.execute(text(avg_query), filters)
                results["avg_price"] = float(result.scalar() or 0)
            
            # Price range
            if "price_range" in aggregations:
                range_query = f"""
                    SELECT MIN(price) as min_price, MAX(price) as max_price 
                    FROM properties WHERE {base_conditions}
                """
                result = conn.execute(text(range_query), filters)
                row = result.fetchone()
                results["price_range"] = {
                    "min": row[0],
                    "max": row[1]
                }
            
            # Group by location
            if "by_location" in aggregations:
                location_query = f"""
                    SELECT location, COUNT(*) as count, AVG(price) as avg_price
                    FROM properties 
                    WHERE {base_conditions}
                    GROUP BY location
                    ORDER BY count DESC
                """
                result = conn.execute(text(location_query), filters)
                results["by_location"] = [dict(row._mapping) for row in result]
            
            # Group by property type
            if "by_type" in aggregations:
                type_query = f"""
                    SELECT 
                        CASE 
                            WHEN metadata_tags LIKE '%apartment%' THEN 'apartment'
                            WHEN metadata_tags LIKE '%house%' THEN 'house'
                            WHEN metadata_tags LIKE '%villa%' THEN 'villa'
                            ELSE 'other'
                        END as property_type,
                        COUNT(*) as count,
                        AVG(price) as avg_price
                    FROM properties
                    WHERE {base_conditions}
                    GROUP BY property_type
                """
                result = conn.execute(text(type_query), filters)
                results["by_type"] = [dict(row._mapping) for row in result]
        
        return results
    
    async def _get_property_details(self, property_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific property"""
        query = """
            SELECT *,
                   EXTRACT(day FROM NOW() - listing_date) as days_on_market
            FROM properties
            WHERE property_id = :property_id
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"property_id": property_id})
            row = result.fetchone()
            
            if row:
                prop = dict(row._mapping)
                # Parse JSON fields
                if prop.get("parsed_json"):
                    prop["parsed_json"] = json.loads(prop["parsed_json"]) if isinstance(prop["parsed_json"], str) else prop["parsed_json"]
                
                # Add computed fields
                prop["price_per_sqft"] = self._calculate_price_per_sqft(prop)
                prop["amenities"] = self._extract_amenities(prop)
                
                return prop
            
            return {}
    
    async def _custom_query(self, sql_query: str) -> List[Dict]:
        """Execute custom SQL query (with safety checks)"""
        # Safety checks
        forbidden_keywords = ["DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER"]
        query_upper = sql_query.upper()
        
        for keyword in forbidden_keywords:
            if keyword in query_upper:
                raise ValueError(f"Forbidden SQL keyword: {keyword}")
        
        # Execute query
        with self.engine.connect() as conn:
            result = conn.execute(text(sql_query))
            return [dict(row._mapping) for row in result]
    
    def _build_filter_conditions(self, filters: Dict[str, Any]) -> str:
        """Build WHERE clause conditions from filters"""
        conditions = ["1=1"]
        
        if filters.get("location"):
            conditions.append(f"LOWER(location) LIKE LOWER('%{filters['location']}%')")
        
        if filters.get("min_price"):
            conditions.append(f"price >= {filters['min_price']}")
        
        if filters.get("max_price"):
            conditions.append(f"price <= {filters['max_price']}")
        
        return " AND ".join(conditions)
    
    def _calculate_price_per_sqft(self, property_data: Dict) -> Optional[float]:
        """Calculate price per square foot if data available"""
        if property_data.get("price") and property_data.get("parsed_json"):
            parsed = property_data["parsed_json"]
            # Try to extract area from parsed data
            if isinstance(parsed, dict):
                for room in parsed.get("rooms_detail", []):
                    if room.get("approx_area"):
                        total_area = sum(r.get("approx_area", 0) for r in parsed["rooms_detail"])
                        if total_area > 0:
                            return property_data["price"] / total_area
        
        return None
    
    def _extract_amenities(self, property_data: Dict) -> List[str]:
        """Extract amenities from metadata tags"""
        amenities = []
        
        if property_data.get("metadata_tags"):
            tags = property_data["metadata_tags"].lower()
            
            amenity_keywords = [
                "parking", "gym", "pool", "security", "lift", "elevator",
                "garden", "balcony", "terrace", "furnished", "unfurnished",
                "semi-furnished", "power backup", "water supply"
            ]
            
            for keyword in amenity_keywords:
                if keyword in tags:
                    amenities.append(keyword.title())
        
        return amenities

    def get_properties_for_renovation(self, area: Optional[int] = None, rooms: Optional[int] = None) -> List[Dict]:
        """Get properties matching renovation criteria"""
        query = """
            SELECT property_id, title, location, parsed_json
            FROM properties
            WHERE parsed_json IS NOT NULL
        """
        
        params = {}
        
        if rooms:
            query += " AND (parsed_json->>'rooms')::int = :rooms"
            params["rooms"] = rooms
        
        query += " LIMIT 5"
        
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            properties = []
            for row in result:
                prop = dict(row._mapping)
                if prop.get("parsed_json"):
                    prop["parsed_json"] = json.loads(prop["parsed_json"]) if isinstance(prop["parsed_json"], str) else prop["parsed_json"]
                properties.append(prop)
            
            return properties