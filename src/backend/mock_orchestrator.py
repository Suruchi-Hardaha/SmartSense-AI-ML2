# mock_orchestrator.py - Simplified orchestrator for testing

import asyncio
import random
from datetime import datetime
from typing import Dict, Any
import re

class MockAgentOrchestrator:
    """
    Mock orchestrator that simulates multi-agent responses
    Use this when the full agent system is not available
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    async def process_query(self, query: str, user_id: str = "default") -> Dict[str, Any]:
        """Process query and return mock response"""
        
        try:
            # Simulate processing time
            await asyncio.sleep(random.uniform(0.5, 1.5))
            
            # Analyze query for intent
            query_lower = query.lower()
            
            # Property search + renovation estimation
            if "renovation" in query_lower and any(word in query_lower for word in ["find", "search", "apartment", "bhk"]):
                return await self._handle_property_renovation_query(query, user_id)
            
            # Pure property search
            elif any(word in query_lower for word in ["find", "search", "show", "apartment", "bhk", "property"]):
                return await self._handle_property_search(query, user_id)
            
            # Renovation only
            elif "renovation" in query_lower or "estimate" in query_lower:
                return await self._handle_renovation_query(query, user_id)
            
            # Market research
            elif any(word in query_lower for word in ["market", "rate", "price", "trend"]):
                return await self._handle_market_research(query, user_id)
            
            # General query
            else:
                return await self._handle_general_query(query, user_id)
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing query: {str(e)}",
                "error_type": "processing_error",
                "data": {}
            }
    
    async def _handle_property_renovation_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle combined property search and renovation queries"""
        
        # Extract parameters
        bhk = self._extract_bhk(query)
        location = self._extract_location(query)
        budget = self._extract_budget(query)
        
        # Mock properties
        properties = self._generate_mock_properties(bhk, location, budget, count=random.randint(5, 12))
        
        # Mock renovation estimate
        renovation_estimate = self._generate_renovation_estimate(bhk)
        
        # Mock market data
        market_data = self._generate_market_data(location)
        
        return {
            "success": True,
            "message": f"Found {len(properties)} properties in {location or 'your area'}. Renovation estimate: ₹{renovation_estimate['total_cost']:,}",
            "data": {
                "properties": properties,
                "renovation_estimate": renovation_estimate,
                "market_data": market_data
            },
            "citations": [
                {"index": 1, "source": "Property Database", "relevance": 0.95},
                {"index": 2, "source": "Market Analysis", "relevance": 0.87}
            ],
            "metadata": {
                "agents_used": ["query_router", "planner", "structured_data", "renovation_estimation", "web_research"],
                "complexity": "high",
                "execution_time": random.uniform(1.8, 3.2)
            }
        }
    
    async def _handle_property_search(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle property search queries"""
        
        bhk = self._extract_bhk(query)
        location = self._extract_location(query)
        budget = self._extract_budget(query)
        
        properties = self._generate_mock_properties(bhk, location, budget, count=random.randint(8, 15))
        market_data = self._generate_market_data(location)
        
        return {
            "success": True,
            "message": f"Found {len(properties)} properties matching your criteria",
            "data": {
                "properties": properties,
                "market_data": market_data
            },
            "citations": [
                {"index": 1, "source": "Property Database", "relevance": 0.92}
            ],
            "metadata": {
                "agents_used": ["query_router", "structured_data", "web_research"],
                "execution_time": random.uniform(1.2, 2.1)
            }
        }
    
    async def _handle_renovation_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle renovation estimation queries"""
        
        bhk = self._extract_bhk(query) or 2
        area_sqft = self._extract_area(query) or 1000
        
        renovation_estimate = self._generate_renovation_estimate(bhk, area_sqft)
        
        return {
            "success": True,
            "message": f"Renovation estimate for {bhk}BHK ({area_sqft} sqft): ₹{renovation_estimate['total_cost']:,}",
            "data": {
                "renovation_estimate": renovation_estimate
            },
            "citations": [
                {"index": 1, "source": "Renovation Cost Database", "relevance": 0.89}
            ],
            "metadata": {
                "agents_used": ["query_router", "renovation_estimation"],
                "execution_time": random.uniform(0.8, 1.5)
            }
        }
    
    async def _handle_market_research(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle market research queries"""
        
        location = self._extract_location(query)
        market_data = self._generate_market_data(location)
        
        return {
            "success": True,
            "message": f"Market analysis for {location or 'the area'} shows average rates of ₹{market_data['avg_price_sqft']:,}/sqft",
            "data": {
                "market_data": market_data
            },
            "citations": [
                {"index": 1, "source": "Market Research API", "relevance": 0.94}
            ],
            "metadata": {
                "agents_used": ["query_router", "web_research", "rag"],
                "execution_time": random.uniform(1.0, 2.0)
            }
        }
    
    async def _handle_general_query(self, query: str, user_id: str) -> Dict[str, Any]:
        """Handle general queries"""
        
        return {
            "success": True,
            "message": "I understand you're asking about real estate. Could you be more specific about properties, renovation, or market information?",
            "data": {
                "suggestions": [
                    "Try: 'Find 2BHK apartments in Koramangala'",
                    "Try: 'Estimate renovation costs for 1200 sqft apartment'",
                    "Try: 'What are current market rates in HSR Layout?'"
                ]
            },
            "metadata": {
                "agents_used": ["query_router"],
                "execution_time": random.uniform(0.3, 0.8)
            }
        }
    
    def _extract_bhk(self, query: str) -> int:
        """Extract BHK from query"""
        match = re.search(r'(\d+)\s*bhk', query.lower())
        return int(match.group(1)) if match else None
    
    def _extract_location(self, query: str) -> str:
        """Extract location from query"""
        locations = [
            "Koramangala", "HSR Layout", "Indiranagar", "Whitefield", 
            "Electronic City", "Marathahalli", "BTM Layout", "Jayanagar",
            "Rajajinagar", "Malleshwaram", "Yelahanka", "Hebbal"
        ]
        
        for location in locations:
            if location.lower() in query.lower():
                return location
        
        # Try to extract from pattern "in X"
        match = re.search(r'in\s+([A-Za-z\s]+?)(?:\s|,|$)', query)
        if match:
            return match.group(1).strip().title()
        
        return "Bangalore"
    
    def _extract_budget(self, query: str) -> int:
        """Extract budget from query"""
        # Look for patterns like "under 50 lakhs", "below 1 crore"
        budget_patterns = [
            r'(?:under|below|up\s+to)\s*(?:rs\.?\s*|₹\s*)?(\d+(?:\.\d+)?)\s*(lakh|crore)',
            r'(?:rs\.?\s*|₹\s*)?(\d+(?:\.\d+)?)\s*(lakh|crore)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, query.lower())
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                
                if unit == "crore":
                    return int(value * 10000000)
                elif unit == "lakh":
                    return int(value * 100000)
        
        return None
    
    def _extract_area(self, query: str) -> int:
        """Extract area from query"""
        match = re.search(r'(\d+)\s*(?:sq\.?\s*ft|sqft)', query.lower())
        return int(match.group(1)) if match else None
    
    def _generate_mock_properties(self, bhk: int, location: str, budget: int, count: int = 10) -> list:
        """Generate mock property data"""
        
        properties = []
        base_prices = {
            "Koramangala": 8000,
            "HSR Layout": 7500,
            "Indiranagar": 9000,
            "Whitefield": 5500,
            "Electronic City": 5000,
            "Marathahalli": 6000,
            "BTM Layout": 6500,
            "Jayanagar": 7000,
            "Rajajinagar": 6800,
            "Malleshwaram": 7200
        }
        
        base_price = base_prices.get(location, 6000)
        bhk = bhk or 2
        
        for i in range(count):
            area = random.randint(800, 1800)
            price_per_sqft = base_price + random.randint(-1000, 1500)
            total_price = area * price_per_sqft
            
            # Filter by budget if specified
            if budget and total_price > budget:
                continue
            
            property_data = {
                "id": f"prop_{i+1:03d}",
                "title": f"{bhk}BHK {random.choice(['Apartment', 'Flat', 'Unit'])} in {location}",
                "location": location,
                "price": total_price,
                "area_sqft": area,
                "bhk": bhk,
                "bathrooms": random.randint(1, bhk),
                "property_type": "apartment",
                "amenities": random.sample([
                    "Parking", "Gym", "Swimming Pool", "Security", 
                    "Power Backup", "Lift", "Garden", "Club House"
                ], random.randint(3, 6)),
                "age": random.choice(["New", "1-2 years", "2-5 years", "5-10 years"]),
                "furnishing": random.choice(["Unfurnished", "Semi-furnished", "Fully furnished"]),
                "floor": f"{random.randint(1, 12)}/{random.randint(12, 20)}",
                "facing": random.choice(["North", "South", "East", "West", "North-East", "North-West"]),
                "possession": random.choice(["Ready to Move", "Under Construction"]),
                "price_per_sqft": price_per_sqft
            }
            
            properties.append(property_data)
        
        return properties[:count]
    
    def _generate_renovation_estimate(self, bhk: int = 2, area_sqft: int = 1000) -> Dict[str, Any]:
        """Generate mock renovation estimate"""
        
        bhk = bhk or 2
        area_sqft = area_sqft or (bhk * 500)  # Estimate if not provided
        
        # Base costs per sqft for different categories
        cost_breakdown = {
            "flooring": area_sqft * random.randint(80, 120),
            "painting": area_sqft * random.randint(25, 45),
            "electrical": area_sqft * random.randint(40, 70),
            "plumbing": area_sqft * random.randint(30, 60),
            "kitchen": random.randint(150000, 300000),
            "bathrooms": random.randint(80000, 150000) * bhk,
            "false_ceiling": area_sqft * random.randint(50, 90),
            "woodwork": area_sqft * random.randint(100, 200),
            "labor": 0,  # Will be calculated
            "miscellaneous": 0  # Will be calculated
        }
        
        # Calculate labor and misc
        material_cost = sum(cost_breakdown.values())
        cost_breakdown["labor"] = int(material_cost * 0.25)
        cost_breakdown["miscellaneous"] = int(material_cost * 0.05)
        
        total_cost = sum(cost_breakdown.values())
        
        return {
            "area_sqft": area_sqft,
            "bhk": bhk,
            "cost_breakdown": cost_breakdown,
            "total_cost": total_cost,
            "cost_per_sqft": int(total_cost / area_sqft),
            "timeline_weeks": random.randint(8, 16),
            "quality_level": "standard",
            "payment_schedule": [
                {"milestone": "Project Start", "percentage": 20, "amount": int(total_cost * 0.2)},
                {"milestone": "Material Purchase", "percentage": 30, "amount": int(total_cost * 0.3)},
                {"milestone": "50% Completion", "percentage": 25, "amount": int(total_cost * 0.25)},
                {"milestone": "90% Completion", "percentage": 20, "amount": int(total_cost * 0.2)},
                {"milestone": "Final Handover", "percentage": 5, "amount": int(total_cost * 0.05)}
            ],
            "recommendations": [
                "Use premium materials for long-term durability",
                "Plan electrical points carefully for future needs",
                "Consider modular kitchen for better space utilization"
            ]
        }
    
    def _generate_market_data(self, location: str = "Bangalore") -> Dict[str, Any]:
        """Generate mock market data"""
        
        base_rates = {
            "Koramangala": 8500,
            "HSR Layout": 7800,
            "Indiranagar": 9200,
            "Whitefield": 5800,
            "Electronic City": 5200,
            "Marathahalli": 6300,
            "BTM Layout": 6800,
            "Jayanagar": 7300,
            "Rajajinagar": 7100,
            "Malleshwaram": 7500
        }
        
        avg_price_sqft = base_rates.get(location, 6500)
        
        return {
            "location": location,
            "avg_price_sqft": avg_price_sqft,
            "price_trend": {
                "monthly_change": round(random.uniform(-2.0, 4.0), 1),
                "yearly_change": round(random.uniform(8.0, 18.0), 1),
                "direction": random.choice(["increasing", "stable", "decreasing"])
            },
            "demand_level": random.choice(["High", "Medium", "Low"]),
            "supply_status": random.choice(["Good", "Limited", "Abundant"]),
            "price_range": {
                "min": int(avg_price_sqft * 0.8),
                "max": int(avg_price_sqft * 1.3)
            },
            "rental_yield": round(random.uniform(2.5, 4.5), 1),
            "investment_rating": round(random.uniform(3.5, 4.8), 1),
            "key_factors": [
                "IT sector growth",
                "Infrastructure development",
                "Metro connectivity",
                "Educational institutions"
            ],
            "upcoming_projects": [
                "New metro line (2025)",
                "IT park expansion",
                "Shopping complex development"
            ]
        }
    
    async def generate_report_from_data(self, report_type: str, data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Generate mock report"""
        
        # Simulate report generation time
        await asyncio.sleep(random.uniform(1.0, 2.5))
        
        report_filename = f"{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return {
            "success": True,
            "report": {
                "report_type": report_type,
                "download_link": f"/download-pdf/{report_filename}",
                "pages": random.randint(8, 15),
                "generated_at": datetime.now().isoformat()
            },
            "message": "Report generated successfully"
        }


# Export for use in main app
AgentOrchestrator = MockAgentOrchestrator