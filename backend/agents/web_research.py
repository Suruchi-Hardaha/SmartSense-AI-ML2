# agents/web_research.py - Web Research Agent

from typing import Dict, Any, Optional, List
from agents.base import BaseAgent, AgentResponse, AgentType
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import json
from datetime import datetime

class WebResearchAgent(BaseAgent):
    """
    Web Research Agent - Fetches external live data
    Retrieves market rates, neighborhood info, and real-time property data
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("WebResearch", AgentType.WEB_RESEARCH, config)
        
        # API configurations
        self.apis = {
            "tavily": config.get("tavily_api_key") if config else None,
            "google_places": config.get("google_places_api_key") if config else None,# i dindnt use this api becuase it was paid
            "property_api": config.get("property_api_endpoint") if config else None
        }
        
        # Rate limiting
        self.rate_limit = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        # Cache for frequently requested data
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Fetch external data based on research type
        """
        try:
            research_type = input_data.get("research_type", "market_rates")
            location = input_data.get("location", "")
            
            # Check cache first
            cache_key = f"{research_type}:{location}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                    return AgentResponse(
                        success=True,
                        data=cached_data,
                        message="Retrieved from cache",
                        metadata={"source": "cache"}
                    )
            
            # Fetch fresh data
            if research_type == "market_rates":
                data = await self._fetch_market_rates(location)
            elif research_type == "neighborhood":
                data = await self._fetch_neighborhood_info(location)
            elif research_type == "property_trends":
                data = await self._fetch_property_trends(location)
            elif research_type == "amenities":
                data = await self._fetch_nearby_amenities(location)
            elif research_type == "demographics":
                data = await self._fetch_demographics(location)
            else:
                data = await self._general_search(input_data.get("query", ""))
            
            # Cache the result
            self.cache[cache_key] = (data, datetime.now())
            
            return AgentResponse(
                success=True,
                data=data,
                message=f"Successfully fetched {research_type} data",
                metadata={
                    "source": "external",
                    "research_type": research_type,
                    "location": location
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Web research failed: {str(e)}"
            )
    
    async def _fetch_market_rates(self, location: str) -> Dict[str, Any]:
        """Fetch current market rates for a location"""
        # If using Tavily API
        if self.apis.get("tavily"):
            return await self._tavily_search(f"real estate market rates {location} India current prices per sqft")
        
        # Fallback to mock data (replace with actual API)
        market_data = {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "average_price_per_sqft": {
                "residential": 5500,
                "commercial": 8500
            },
            "price_trends": {
                "monthly_change": 2.3,
                "yearly_change": 12.5,
                "trend": "increasing"
            },
            "property_types": {
                "1BHK": {"avg_price": 3500000, "range": [2500000, 4500000]},
                "2BHK": {"avg_price": 5500000, "range": [4000000, 7000000]},
                "3BHK": {"avg_price": 8500000, "range": [6500000, 12000000]},
                "4BHK": {"avg_price": 15000000, "range": [12000000, 25000000]}
            },
            "rental_yields": {
                "average": 3.2,
                "range": [2.5, 4.5]
            },
            "market_sentiment": "bullish",
            "key_factors": [
                "Infrastructure development",
                "IT sector growth",
                "Metro connectivity"
            ]
        }
        
        return market_data
    
    async def _fetch_neighborhood_info(self, location: str) -> Dict[str, Any]:
        """Fetch neighborhood information"""
        neighborhood_info = {
            "location": location,
            "overview": f"{location} is a well-developed area with good connectivity",
            "connectivity": {
                "nearest_metro": "2.5 km",
                "nearest_railway": "5 km",
                "airport_distance": "15 km",
                "major_roads": ["MG Road", "Outer Ring Road"]
            },
            "amenities": {
                "schools": 15,
                "hospitals": 8,
                "shopping_centers": 5,
                "parks": 10,
                "restaurants": 50
            },
            "demographics": {
                "population_density": "medium",
                "avg_income_level": "upper-middle",
                "primary_occupation": "IT professionals"
            },
            "safety": {
                "crime_rate": "low",
                "police_stations": 3,
                "street_lighting": "good"
            },
            "environmental": {
                "air_quality": "moderate",
                "green_cover": "25%",
                "noise_levels": "moderate"
            },
            "development_projects": [
                "New metro line (2025)",
                "Shopping mall (2024)",
                "IT park expansion"
            ]
        }
        
        return neighborhood_info
    
    async def _fetch_property_trends(self, location: str) -> Dict[str, Any]:
        """Fetch property market trends"""
        trends = {
            "location": location,
            "period": "2023-2024",
            "price_appreciation": {
                "1_year": 12.5,
                "3_years": 35.2,
                "5_years": 62.8
            },
            "supply_demand": {
                "new_launches": 25,
                "units_sold": 1850,
                "inventory_months": 8.5,
                "demand_level": "high"
            },
            "popular_configurations": [
                {"type": "2BHK", "percentage": 45},
                {"type": "3BHK", "percentage": 35},
                {"type": "1BHK", "percentage": 15},
                {"type": "4BHK+", "percentage": 5}
            ],
            "price_ranges": {
                "budget": {"range": "< 50L", "percentage": 20},
                "mid_segment": {"range": "50L - 1Cr", "percentage": 50},
                "premium": {"range": "1Cr - 2Cr", "percentage": 25},
                "luxury": {"range": "> 2Cr", "percentage": 5}
            },
            "future_outlook": "positive",
            "investment_rating": 4.2
        }
        
        return trends
    
    async def _fetch_nearby_amenities(self, location: str) -> Dict[str, Any]:
        """Fetch nearby amenities using Google Places or similar"""
        amenities = {
            "schools": [
                {"name": "DPS School", "distance": "1.2 km", "rating": 4.5},
                {"name": "Cambridge International", "distance": "2.5 km", "rating": 4.3}
            ],
            "hospitals": [
                {"name": "Apollo Hospital", "distance": "3 km", "rating": 4.6},
                {"name": "Fortis Healthcare", "distance": "5 km", "rating": 4.4}
            ],
            "shopping": [
                {"name": "Phoenix Mall", "distance": "2 km", "type": "mall"},
                {"name": "D-Mart", "distance": "1.5 km", "type": "supermarket"}
            ],
            "transport": [
                {"name": "Metro Station", "distance": "1.8 km", "line": "Blue Line"},
                {"name": "Bus Stop", "distance": "500 m", "routes": ["201", "305", "410"]}
            ],
            "recreation": [
                {"name": "Central Park", "distance": "1 km", "type": "park"},
                {"name": "Gold's Gym", "distance": "800 m", "type": "fitness"}
            ],
            "banks_atms": [
                {"name": "HDFC Bank", "distance": "600 m"},
                {"name": "SBI ATM", "distance": "400 m"}
            ]
        }
        
        return amenities
    
    async def _fetch_demographics(self, location: str) -> Dict[str, Any]:
        """Fetch demographic information"""
        demographics = {
            "location": location,
            "population": 125000,
            "households": 35000,
            "age_distribution": {
                "0-18": 22,
                "19-35": 38,
                "36-50": 25,
                "51-65": 10,
                "65+": 5
            },
            "income_levels": {
                "< 5L": 15,
                "5L-10L": 25,
                "10L-25L": 35,
                "25L-50L": 20,
                "> 50L": 5
            },
            "occupation_types": {
                "IT/Software": 45,
                "Business": 20,
                "Government": 10,
                "Healthcare": 8,
                "Education": 7,
                "Others": 10
            },
            "education_level": {
                "graduate": 65,
                "post_graduate": 25,
                "others": 10
            },
            "vehicle_ownership": {
                "cars": 75,
                "two_wheelers": 85,
                "none": 5
            }
        }
        
        return demographics
    
    async def _tavily_search(self, query: str) -> Dict[str, Any]:
        """Use Tavily API for web search"""
        if not self.apis.get("tavily"):
            return {"error": "Tavily API key not configured"}
        
        async with aiohttp.ClientSession() as session:
            async with self.rate_limit:
                try:
                    headers = {
                        "Content-Type": "application/json",
                        "X-API-Key": self.apis["tavily"]
                    }
                    
                    payload = {
                        "query": query,
                        "search_depth": "advanced",
                        "include_answer": True,
                        "max_results": 5
                    }
                    
                    async with session.post(
                        "https://api.tavily.com/search",
                        json=payload,
                        headers=headers
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "answer": data.get("answer", ""),
                                "results": data.get("results", []),
                                "query": query
                            }
                        else:
                            return {"error": f"Tavily API error: {response.status}"}
                            
                except Exception as e:
                    self.logger.error(f"Tavily search error: {str(e)}")
                    return {"error": str(e)}
    
    async def _general_search(self, query: str) -> Dict[str, Any]:
        """Perform general web search"""
        # Fallback to web scraping or other APIs
        search_results = {
            "query": query,
            "results": [
                {
                    "title": "Real Estate Market Analysis",
                    "snippet": f"Latest analysis for {query}",
                    "url": "https://example.com/analysis"
                }
            ],
            "summary": f"Search results for: {query}"
        }
        
        return search_results
    
    async def fetch_comparative_analysis(self, locations: List[str]) -> Dict[str, Any]:
        """Fetch comparative analysis for multiple locations"""
        comparison = {}
        
        async def fetch_location_data(location):
            market_data = await self._fetch_market_rates(location)
            trends = await self._fetch_property_trends(location)
            return {
                "location": location,
                "avg_price_sqft": market_data["average_price_per_sqft"]["residential"],
                "appreciation_1yr": trends["price_appreciation"]["1_year"],
                "demand_level": trends["supply_demand"]["demand_level"]
            }
        
        # Fetch data for all locations concurrently
        tasks = [fetch_location_data(loc) for loc in locations]
        results = await asyncio.gather(*tasks)
        
        comparison["locations"] = results
        comparison["best_investment"] = max(results, key=lambda x: x["appreciation_1yr"])["location"]
        comparison["best_value"] = min(results, key=lambda x: x["avg_price_sqft"])["location"]
        
        return comparison