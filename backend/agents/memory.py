# agents/memory.py - Memory Component Agent

from typing import Dict, Any, Optional, List
from agents.base import BaseAgent, AgentResponse, AgentType
import json
import redis
from datetime import datetime, timedelta
import pickle
import hashlib

class MemoryAgent(BaseAgent):
    """
    Memory Component - Persists user preferences and conversation history
    Supports multiple types of memory: short-term, long-term, and episodic
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Memory", AgentType.MEMORY, config)
        
        # Redis connection for persistent storage
        self.redis_client = None
        if config and "redis_config" in config:
            redis_cfg = config["redis_config"]
            self.redis_client = redis.Redis(
                host=redis_cfg.get("host", "localhost"),
                port=redis_cfg.get("port", 6379),
                db=redis_cfg.get("db", 0),
                decode_responses=False  # We'll handle encoding/decoding
            )
        
        # In-memory cache for current session
        self.session_memory = {}
        
        # Memory types
        self.memory_types = {
            "preferences": self._handle_preferences,
            "search_history": self._handle_search_history,
            "conversation": self._handle_conversation,
            "saved_properties": self._handle_saved_properties,
            "comparisons": self._handle_comparisons
        }
        
        # Default TTL for different memory types (in seconds)
        self.ttl_config = {
            "short_term": 3600,  # 1 hour
            "medium_term": 86400,  # 1 day
            "long_term": 2592000,  # 30 days
            "permanent": None  # No expiration
        }
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Handle memory operations: store, retrieve, update, delete
        """
        try:
            operation = input_data.get("operation", "retrieve")
            memory_type = input_data.get("memory_type", "preferences")
            user_id = input_data.get("user_id", "default_user")
            data = input_data.get("data")
            
            if operation == "store":
                result = await self._store_memory(user_id, memory_type, data)
            elif operation == "retrieve":
                result = await self._retrieve_memory(user_id, memory_type)
            elif operation == "update":
                result = await self._update_memory(user_id, memory_type, data)
            elif operation == "delete":
                result = await self._delete_memory(user_id, memory_type)
            elif operation == "analyze":
                result = await self._analyze_memory(user_id)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return AgentResponse(
                success=True,
                data=result,
                message=f"Memory {operation} completed successfully",
                metadata={
                    "memory_type": memory_type,
                    "user_id": user_id,
                    "operation": operation
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Memory operation failed: {str(e)}"
            )
    
    async def _store_memory(self, user_id: str, memory_type: str, data: Any) -> Dict:
        """Store data in memory"""
        key = self._generate_key(user_id, memory_type)
        
        # Determine TTL based on memory type
        if memory_type == "conversation":
            ttl = self.ttl_config["short_term"]
        elif memory_type in ["search_history", "comparisons"]:
            ttl = self.ttl_config["medium_term"]
        else:
            ttl = self.ttl_config["long_term"]
        
        # Store in session memory
        self.session_memory[key] = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "type": memory_type
        }
        
        # Store in Redis if available
        if self.redis_client:
            try:
                serialized = pickle.dumps(self.session_memory[key])
                if ttl:
                    self.redis_client.setex(key, ttl, serialized)
                else:
                    self.redis_client.set(key, serialized)
            except Exception as e:
                self.logger.error(f"Redis storage error: {str(e)}")
        
        return {"stored": True, "key": key, "ttl": ttl}
    
    async def _retrieve_memory(self, user_id: str, memory_type: str) -> Any:
        """Retrieve data from memory"""
        key = self._generate_key(user_id, memory_type)
        
        # Check session memory first
        if key in self.session_memory:
            return self.session_memory[key]
        
        # Check Redis if available
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    deserialized = pickle.loads(data)
                    self.session_memory[key] = deserialized
                    return deserialized
            except Exception as e:
                self.logger.error(f"Redis retrieval error: {str(e)}")
        
        # Handle specific memory types
        if memory_type in self.memory_types:
            return await self.memory_types[memory_type]("retrieve", user_id, None)
        
        return None
    
    async def _update_memory(self, user_id: str, memory_type: str, data: Any) -> Dict:
        """Update existing memory"""
        # Retrieve existing data
        existing = await self._retrieve_memory(user_id, memory_type)
        
        if existing and isinstance(existing, dict) and "data" in existing:
            # Merge with existing data
            if isinstance(existing["data"], dict) and isinstance(data, dict):
                existing["data"].update(data)
            elif isinstance(existing["data"], list) and isinstance(data, list):
                existing["data"].extend(data)
            else:
                existing["data"] = data
            
            existing["last_updated"] = datetime.now().isoformat()
            
            # Store updated data
            return await self._store_memory(user_id, memory_type, existing["data"])
        else:
            # No existing data, create new
            return await self._store_memory(user_id, memory_type, data)
    
    async def _delete_memory(self, user_id: str, memory_type: str) -> Dict:
        """Delete memory data"""
        key = self._generate_key(user_id, memory_type)
        
        # Remove from session memory
        if key in self.session_memory:
            del self.session_memory[key]
        
        # Remove from Redis if available
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                self.logger.error(f"Redis deletion error: {str(e)}")
        
        return {"deleted": True, "key": key}
    
    async def _analyze_memory(self, user_id: str) -> Dict:
        """Analyze user's memory patterns and preferences"""
        analysis = {
            "user_id": user_id,
            "preferences": {},
            "patterns": {},
            "recommendations": []
        }
        
        # Retrieve all memory types
        preferences = await self._retrieve_memory(user_id, "preferences")
        search_history = await self._retrieve_memory(user_id, "search_history")
        saved_properties = await self._retrieve_memory(user_id, "saved_properties")
        
        # Analyze preferences
        if preferences and isinstance(preferences, dict):
            pref_data = preferences.get("data", {})
            analysis["preferences"] = {
                "budget_range": pref_data.get("budget_range"),
                "preferred_locations": pref_data.get("locations", []),
                "property_types": pref_data.get("property_types", []),
                "preferred_bhk": pref_data.get("bhk")
            }
        
        # Analyze search patterns
        if search_history and isinstance(search_history, dict):
            history_data = search_history.get("data", [])
            if isinstance(history_data, list):
                # Extract patterns
                locations = {}
                price_ranges = []
                
                for search in history_data:
                    if "location" in search:
                        loc = search["location"]
                        locations[loc] = locations.get(loc, 0) + 1
                    if "price" in search:
                        price_ranges.append(search["price"])
                
                analysis["patterns"]["most_searched_locations"] = sorted(
                    locations.items(), key=lambda x: x[1], reverse=True
                )[:5]
                
                if price_ranges:
                    analysis["patterns"]["avg_price_searched"] = sum(price_ranges) / len(price_ranges)
        
        # Generate recommendations
        if analysis["preferences"]:
            if analysis["preferences"].get("budget_range"):
                analysis["recommendations"].append(
                    f"Focus on properties within ₹{analysis['preferences']['budget_range'][0]:,} - ₹{analysis['preferences']['budget_range'][1]:,}"
                )
            
            if analysis["preferences"].get("preferred_locations"):
                analysis["recommendations"].append(
                    f"Prioritize searches in: {', '.join(analysis['preferences']['preferred_locations'])}"
                )
        
        return analysis
    
    async def _handle_preferences(self, operation: str, user_id: str, data: Any) -> Any:
        """Handle user preferences"""
        if operation == "retrieve":
            # Default preferences
            return {
                "data": {
                    "budget_range": [2500000, 10000000],
                    "locations": [],
                    "property_types": ["apartment", "house"],
                    "bhk": [2, 3],
                    "amenities": ["parking", "security"],
                    "possession": "ready_to_move"
                },
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    async def _handle_search_history(self, operation: str, user_id: str, data: Any) -> Any:
        """Handle search history"""
        if operation == "retrieve":
            return {
                "data": [],
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    async def _handle_conversation(self, operation: str, user_id: str, data: Any) -> Any:
        """Handle conversation history"""
        if operation == "retrieve":
            return {
                "data": {
                    "messages": [],
                    "context": {}
                },
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    async def _handle_saved_properties(self, operation: str, user_id: str, data: Any) -> Any:
        """Handle saved properties"""
        if operation == "retrieve":
            return {
                "data": [],
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    async def _handle_comparisons(self, operation: str, user_id: str, data: Any) -> Any:
        """Handle property comparisons"""
        if operation == "retrieve":
            return {
                "data": [],
                "timestamp": datetime.now().isoformat()
            }
        
        return None
    
    def _generate_key(self, user_id: str, memory_type: str) -> str:
        """Generate unique key for memory storage"""
        return f"smartsense:memory:{user_id}:{memory_type}"
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get complete user context for other agents"""
        context = {
            "user_id": user_id,
            "preferences": None,
            "recent_searches": [],
            "saved_properties": [],
            "conversation_context": {}
        }
        
        # Gather all relevant memory data
        for memory_type in ["preferences", "search_history", "saved_properties", "conversation"]:
            key = self._generate_key(user_id, memory_type)
            if key in self.session_memory:
                data = self.session_memory[key].get("data")
                
                if memory_type == "preferences":
                    context["preferences"] = data
                elif memory_type == "search_history" and isinstance(data, list):
                    context["recent_searches"] = data[-5:]  # Last 5 searches
                elif memory_type == "saved_properties":
                    context["saved_properties"] = data
                elif memory_type == "conversation":
                    context["conversation_context"] = data.get("context", {})
        
        return context
    
    async def learn_from_interaction(self, user_id: str, interaction_data: Dict) -> Dict:
        """Learn from user interactions to improve preferences"""
        # Retrieve current preferences
        preferences = await self._retrieve_memory(user_id, "preferences")
        
        if not preferences:
            preferences = {"data": {}}
        
        pref_data = preferences.get("data", {})
        
        # Update preferences based on interaction
        if "selected_property" in interaction_data:
            prop = interaction_data["selected_property"]
            
            # Update location preference
            if "location" in prop:
                if "locations" not in pref_data:
                    pref_data["locations"] = []
                if prop["location"] not in pref_data["locations"]:
                    pref_data["locations"].append(prop["location"])
            
            # Update price range
            if "price" in prop:
                if "budget_range" not in pref_data:
                    pref_data["budget_range"] = [prop["price"] * 0.8, prop["price"] * 1.2]
                else:
                    # Adjust range based on selected property
                    min_budget = min(pref_data["budget_range"][0], prop["price"] * 0.9)
                    max_budget = max(pref_data["budget_range"][1], prop["price"] * 1.1)
                    pref_data["budget_range"] = [min_budget, max_budget]
        
        # Store updated preferences
        await self._update_memory(user_id, "preferences", pref_data)
        
        return {"learned": True, "updated_preferences": pref_data}