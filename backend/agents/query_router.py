# agents/query_router.py - Query Router Agent

import re
from typing import Dict, Any, Optional, List, Tuple
from agents.base import BaseAgent, AgentResponse, AgentType
import spacy
from transformers import pipeline

class QueryRouterAgent(BaseAgent):
    """
    Query Router Agent - Detects intent and extracts slots from user queries
    Routes queries to appropriate agents based on intent
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("QueryRouter", AgentType.QUERY_ROUTER, config)
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Intent patterns
        self.intent_patterns = {
            "property_search": [
                r"find.*(?:property|house|apartment|flat|home)",
                r"show.*(?:property|house|apartment|flat|home)",
                r"search.*(?:property|house|apartment|flat|home)",
                r"looking for.*(?:property|house|apartment|flat|home)",
                r"(?:2|3|4)[\s-]?bhk",
                r"properties in",
                r"apartments near"
            ],
            "renovation_estimate": [
                r"renovation.*cost",
                r"estimate.*renovation",
                r"how much.*renovate",
                r"renovation budget",
                r"remodel.*cost"
            ],
            "report_generation": [
                r"generate.*report",
                r"create.*(?:report|summary|document)",
                r"pdf.*report",
                r"detailed.*(?:report|analysis)",
                r"download.*report"
            ],
            "market_research": [
                r"market.*(?:rate|price|trend)",
                r"property.*prices",
                r"neighborhood.*info",
                r"area.*details",
                r"locality.*information"
            ],
            "complex_query": [
                r"and.*also",
                r"then.*generate",
                r"multiple.*properties",
                r"compare.*properties"
            ]
        }
        
        # Slot extraction patterns
        self.slot_patterns = {
            "location": r"(?:in|near|at|around)\s+([\w\s]+?)(?:\s|,|$)",
            "bhk": r"(\d+)[\s-]?bhk",
            "budget": r"(?:budget|price|under|below|around)\s*(?:of|is)?\s*(?:inr|rs\.?|₹)?\s*([\d,]+(?:\s*(?:lakh|lac|l|crore|cr))?)",
            "property_type": r"(apartment|flat|house|villa|plot|office|shop)",
            "rooms": r"(\d+)\s*(?:room|bedroom)",
            "area_sqft": r"(\d+)\s*(?:sq\.?\s*ft|sqft|square\s*feet)"
        }
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Analyze user query to detect intent and extract slots
        """
        try:
            query = input_data.get("query", "").lower()
            
            # Detect intent
            intent = self._detect_intent(query)
            
            # Extract slots
            slots = self._extract_slots(query)
            
            # Determine target agents
            target_agents = self._determine_target_agents(intent, slots)
            
            # Create routing decision
            routing_decision = {
                "intent": intent,
                "slots": slots,
                "target_agents": target_agents,
                "requires_planning": len(target_agents) > 1 or intent == "complex_query"
            }
            
            return AgentResponse(
                success=True,
                data=routing_decision,
                message=f"Query routed successfully. Intent: {intent}",
                metadata={"confidence": self._calculate_confidence(intent, query)}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Query routing failed: {str(e)}"
            )
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1
            scores[intent] = score
        
        # Get the intent with highest score
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        
        # Default intent for general queries
        return "property_search"
    
    def _extract_slots(self, query: str) -> Dict[str, Any]:
        """Extract relevant slots/entities from the query"""
        slots = {}
        
        for slot_name, pattern in self.slot_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                # Process budget values
                if slot_name == "budget":
                    value = self._parse_budget(value)
                # Process BHK values
                elif slot_name == "bhk":
                    value = int(value)
                # Process area values
                elif slot_name == "area_sqft":
                    value = int(value)
                
                slots[slot_name] = value
        
        # Use spaCy for location extraction if not found with regex
        if "location" not in slots:
            doc = self.nlp(query)
            locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
            if locations:
                slots["location"] = locations[0]
        
        return slots
    
    def _parse_budget(self, budget_str: str) -> int:
        """Parse budget string to integer value in rupees"""
        budget_str = budget_str.lower().replace(",", "")
        
        # Extract number
        import re
        num_match = re.search(r"([\d.]+)", budget_str)
        if not num_match:
            return 0
        
        value = float(num_match.group(1))
        
        # Convert to rupees based on unit
        if "crore" in budget_str or "cr" in budget_str:
            value *= 10000000
        elif "lakh" in budget_str or "lac" in budget_str or "l" in budget_str:
            value *= 100000
        
        return int(value)
    
    def _determine_target_agents(self, intent: str, slots: Dict[str, Any]) -> List[AgentType]:
        """Determine which agents should handle this query"""
        agents = []
        
        if intent == "property_search":
            agents.append(AgentType.STRUCTURED_DATA)
            if slots.get("location"):
                agents.append(AgentType.WEB_RESEARCH)
        
        elif intent == "renovation_estimate":
            agents.append(AgentType.RENOVATION_ESTIMATION)
            if slots:
                agents.append(AgentType.STRUCTURED_DATA)
        
        elif intent == "report_generation":
            agents.append(AgentType.REPORT_GENERATION)
            agents.append(AgentType.STRUCTURED_DATA)
        
        elif intent == "market_research":
            agents.append(AgentType.WEB_RESEARCH)
            agents.append(AgentType.RAG)
        
        elif intent == "complex_query":
            agents.append(AgentType.PLANNER)
        
        # Always include memory for context
        agents.append(AgentType.MEMORY)
        
        return agents
    
    def _calculate_confidence(self, intent: str, query: str) -> float:
        """Calculate confidence score for the intent detection"""
        matched_patterns = 0
        total_patterns = len(self.intent_patterns.get(intent, []))
        
        if total_patterns == 0:
            return 0.5
        
        for pattern in self.intent_patterns.get(intent, []):
            if re.search(pattern, query, re.IGNORECASE):
                matched_patterns += 1
        
        return min(matched_patterns / max(total_patterns * 0.3, 1), 1.0)