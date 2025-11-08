# agents/renovation_estimation.py - Renovation Estimation Agent

from typing import Dict, Any, Optional, List
from agents.base import BaseAgent, AgentResponse, AgentType
from datetime import datetime, timedelta

class RenovationEstimationAgent(BaseAgent):
    """
    Renovation Estimation Agent - Calculates renovation costs
    Provides detailed cost breakdown based on property size and requirements
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("RenovationEstimation", AgentType.RENOVATION_ESTIMATION, config)
        
        # Cost rates per sq.ft (in INR)
        self.cost_rates = {
            "basic": {
                "flooring": 80,
                "painting": 30,
                "electrical": 50,
                "plumbing": 40,
                "false_ceiling": 60,
                "woodwork": 150,
                "bathroom": 300,
                "kitchen": 400,
                "windows": 100,
                "doors": 120
            },
            "premium": {
                "flooring": 150,
                "painting": 50,
                "electrical": 80,
                "plumbing": 70,
                "false_ceiling": 100,
                "woodwork": 300,
                "bathroom": 500,
                "kitchen": 700,
                "windows": 200,
                "doors": 250
            },
            "luxury": {
                "flooring": 300,
                "painting": 80,
                "electrical": 120,
                "plumbing": 100,
                "false_ceiling": 150,
                "woodwork": 500,
                "bathroom": 800,
                "kitchen": 1200,
                "windows": 350,
                "doors": 400
            }
        }
        
        # Material quality multipliers
        self.quality_multipliers = {
            "economy": 0.8,
            "standard": 1.0,
            "premium": 1.5,
            "luxury": 2.0
        }
        
        # Timeline estimates (in weeks)
        self.timeline_estimates = {
            "small": {  # < 1000 sqft
                "minimal": 3,
                "moderate": 6,
                "extensive": 10
            },
            "medium": {  # 1000-2000 sqft
                "minimal": 4,
                "moderate": 8,
                "extensive": 14
            },
            "large": {  # > 2000 sqft
                "minimal": 6,
                "moderate": 12,
                "extensive": 20
            }
        }
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Calculate renovation costs based on property details and requirements
        """
        try:
            # Extract input parameters
            area_sqft = input_data.get("area_sqft", 1000)
            rooms = input_data.get("rooms", 2)
            bathrooms = input_data.get("bathrooms", 1)
            renovation_type = input_data.get("renovation_type", "moderate")
            quality_level = input_data.get("quality_level", "standard")
            custom_requirements = input_data.get("custom_requirements", [])
            
            # Calculate costs
            cost_breakdown = self._calculate_cost_breakdown(
                area_sqft, rooms, bathrooms, renovation_type, quality_level, custom_requirements
            )
            
            # Calculate timeline
            timeline = self._estimate_timeline(area_sqft, renovation_type)
            
            # Get material recommendations
            materials = self._recommend_materials(quality_level)
            
            # Calculate payment schedule
            payment_schedule = self._create_payment_schedule(
                cost_breakdown["total_cost"], timeline
            )
            
            # Create detailed estimate
            estimate = {
                "property_details": {
                    "area_sqft": area_sqft,
                    "rooms": rooms,
                    "bathrooms": bathrooms
                },
                "renovation_scope": renovation_type,
                "quality_level": quality_level,
                "cost_breakdown": cost_breakdown,
                "total_cost": cost_breakdown["total_cost"],
                "cost_per_sqft": cost_breakdown["total_cost"] / area_sqft,
                "timeline_weeks": timeline,
                "estimated_completion": (datetime.now() + timedelta(weeks=timeline)).strftime("%Y-%m-%d"),
                "payment_schedule": payment_schedule,
                "material_recommendations": materials,
                "contractor_recommendations": self._get_contractor_recommendations(renovation_type),
                "warranty_info": self._get_warranty_info(quality_level)
            }
            
            return AgentResponse(
                success=True,
                data=estimate,
                message=f"Renovation estimate prepared: ₹{cost_breakdown['total_cost']:,}",
                metadata={
                    "confidence_level": self._calculate_confidence(input_data),
                    "price_range": self._calculate_price_range(cost_breakdown["total_cost"])
                }
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Renovation estimation failed: {str(e)}"
            )
    
    def _calculate_cost_breakdown(
        self, 
        area_sqft: int, 
        rooms: int, 
        bathrooms: int, 
        renovation_type: str, 
        quality_level: str,
        custom_requirements: List[str]
    ) -> Dict[str, Any]:
        """Calculate detailed cost breakdown"""
        
        # Select rate category based on quality level
        if quality_level == "economy":
            rates = self.cost_rates["basic"]
        elif quality_level in ["standard", "premium"]:
            rates = self.cost_rates["premium"]
        else:
            rates = self.cost_rates["luxury"]
        
        # Quality multiplier
        multiplier = self.quality_multipliers.get(quality_level, 1.0)
        
        breakdown = {}
        
        # Flooring
        if renovation_type in ["moderate", "extensive"]:
            breakdown["flooring"] = int(area_sqft * rates["flooring"] * multiplier)
        
        # Painting
        wall_area = area_sqft * 3  # Approximate wall area
        breakdown["painting"] = int(wall_area * rates["painting"] * multiplier)
        
        # Electrical
        if renovation_type in ["moderate", "extensive"]:
            breakdown["electrical"] = int(area_sqft * rates["electrical"] * multiplier)
        
        # Plumbing
        if renovation_type == "extensive":
            breakdown["plumbing"] = int(area_sqft * rates["plumbing"] * multiplier)
        
        # False Ceiling
        if renovation_type in ["moderate", "extensive"] or "false_ceiling" in custom_requirements:
            ceiling_area = area_sqft * 0.8  # Not all areas need false ceiling
            breakdown["false_ceiling"] = int(ceiling_area * rates["false_ceiling"] * multiplier)
        
        # Woodwork
        if renovation_type in ["moderate", "extensive"]:
            woodwork_area = area_sqft * 0.3  # Approximate woodwork area
            breakdown["woodwork"] = int(woodwork_area * rates["woodwork"] * multiplier)
        
        # Bathroom renovation
        if renovation_type in ["moderate", "extensive"]:
            bathroom_area = bathrooms * 50  # Average bathroom size
            breakdown["bathroom_renovation"] = int(bathroom_area * rates["bathroom"] * multiplier * bathrooms)
        
        # Kitchen renovation
        if renovation_type in ["moderate", "extensive"] or "kitchen" in custom_requirements:
            kitchen_area = 150  # Average kitchen size
            breakdown["kitchen_renovation"] = int(kitchen_area * rates["kitchen"] * multiplier)
        
        # Windows and doors
        if renovation_type == "extensive":
            num_windows = rooms * 2
            num_doors = rooms + 2
            breakdown["windows"] = int(num_windows * 5000 * multiplier)  # Per window cost
            breakdown["doors"] = int(num_doors * 8000 * multiplier)  # Per door cost
        
        # Custom requirements
        for requirement in custom_requirements:
            if requirement == "balcony":
                breakdown["balcony_renovation"] = int(15000 * multiplier)
            elif requirement == "terrace":
                breakdown["terrace_work"] = int(30000 * multiplier)
            elif requirement == "garden":
                breakdown["garden_landscaping"] = int(50000 * multiplier)
            elif requirement == "smart_home":
                breakdown["smart_home_automation"] = int(100000 * multiplier)
            elif requirement == "solar":
                breakdown["solar_installation"] = int(200000 * multiplier)
        
        # Labor costs (typically 20-30% of material cost)
        material_cost = sum(breakdown.values())
        breakdown["labor_cost"] = int(material_cost * 0.25)
        
        # Miscellaneous (5% for unexpected expenses)
        breakdown["miscellaneous"] = int(material_cost * 0.05)
        
        # Calculate total
        total_cost = sum(breakdown.values())
        
        return {
            "breakdown": breakdown,
            "material_cost": material_cost,
            "labor_cost": breakdown["labor_cost"],
            "total_cost": total_cost
        }
    
    def _estimate_timeline(self, area_sqft: int, renovation_type: str) -> int:
        """Estimate renovation timeline in weeks"""
        # Determine property size category
        if area_sqft < 1000:
            size_category = "small"
        elif area_sqft <= 2000:
            size_category = "medium"
        else:
            size_category = "large"
        
        # Map renovation type to timeline category
        timeline_map = {
            "minimal": "minimal",
            "moderate": "moderate",
            "extensive": "extensive"
        }
        
        timeline_category = timeline_map.get(renovation_type, "moderate")
        
        return self.timeline_estimates[size_category][timeline_category]
    
    def _recommend_materials(self, quality_level: str) -> Dict[str, List[str]]:
        """Recommend materials based on quality level"""
        materials = {
            "economy": {
                "flooring": ["Ceramic tiles", "Laminate flooring", "Basic vinyl"],
                "paint": ["Distemper", "Basic emulsion"],
                "sanitaryware": ["Local brands", "Basic fittings"],
                "electrical": ["Standard switches", "Basic LED lights"],
                "woodwork": ["Commercial plywood", "Laminate finish"]
            },
            "standard": {
                "flooring": ["Vitrified tiles", "Engineered wood", "Good quality vinyl"],
                "paint": ["Premium emulsion", "Washable paint"],
                "sanitaryware": ["Jaquar", "Hindware", "Cera"],
                "electrical": ["Modular switches", "Designer LED lights"],
                "woodwork": ["BWR plywood", "Veneer finish"]
            },
            "premium": {
                "flooring": ["Italian marble", "Hardwood", "Premium tiles"],
                "paint": ["Luxury emulsion", "Textured paint"],
                "sanitaryware": ["Kohler", "Grohe", "Roca"],
                "electrical": ["Smart switches", "Automated lighting"],
                "woodwork": ["Marine plywood", "PU finish"]
            },
            "luxury": {
                "flooring": ["Imported marble", "Exotic hardwood", "Designer tiles"],
                "paint": ["Imported paints", "Special finishes"],
                "sanitaryware": ["Duravit", "Villeroy & Boch", "Premium imports"],
                "electrical": ["Full automation", "Designer fixtures"],
                "woodwork": ["Solid wood", "Custom designs"]
            }
        }
        
        return materials.get(quality_level, materials["standard"])
    
    def _create_payment_schedule(self, total_cost: int, timeline_weeks: int) -> List[Dict]:
        """Create payment schedule for renovation"""
        schedule = []
        
        # Advance (20%)
        schedule.append({
            "milestone": "Project initiation",
            "percentage": 20,
            "amount": int(total_cost * 0.20),
            "week": 0
        })
        
        # Material procurement (30%)
        schedule.append({
            "milestone": "Material procurement",
            "percentage": 30,
            "amount": int(total_cost * 0.30),
            "week": 1
        })
        
        # Mid-work payment (25%)
        schedule.append({
            "milestone": "50% work completion",
            "percentage": 25,
            "amount": int(total_cost * 0.25),
            "week": timeline_weeks // 2
        })
        
        # Near completion (20%)
        schedule.append({
            "milestone": "90% work completion",
            "percentage": 20,
            "amount": int(total_cost * 0.20),
            "week": timeline_weeks - 1
        })
        
        # Final payment (5%)
        schedule.append({
            "milestone": "Project handover",
            "percentage": 5,
            "amount": int(total_cost * 0.05),
            "week": timeline_weeks
        })
        
        return schedule
    
    def _get_contractor_recommendations(self, renovation_type: str) -> List[Dict]:
        """Get contractor recommendations based on renovation type"""
        contractors = {
            "minimal": [
                {"name": "Quick Fix Solutions", "rating": 4.2, "projects": 150},
                {"name": "Home Makeover Express", "rating": 4.0, "projects": 200}
            ],
            "moderate": [
                {"name": "Dream Home Builders", "rating": 4.5, "projects": 300},
                {"name": "Urban Living Designs", "rating": 4.6, "projects": 250}
            ],
            "extensive": [
                {"name": "Premium Interiors Ltd", "rating": 4.8, "projects": 500},
                {"name": "Luxury Living Spaces", "rating": 4.7, "projects": 400}
            ]
        }
        
        return contractors.get(renovation_type, contractors["moderate"])
    
    def _get_warranty_info(self, quality_level: str) -> Dict[str, str]:
        """Get warranty information based on quality level"""
        warranty = {
            "economy": {
                "workmanship": "6 months",
                "materials": "As per manufacturer",
                "waterproofing": "1 year"
            },
            "standard": {
                "workmanship": "1 year",
                "materials": "As per manufacturer",
                "waterproofing": "3 years"
            },
            "premium": {
                "workmanship": "2 years",
                "materials": "Extended manufacturer warranty",
                "waterproofing": "5 years"
            },
            "luxury": {
                "workmanship": "3 years",
                "materials": "Premium warranty coverage",
                "waterproofing": "10 years"
            }
        }
        
        return warranty.get(quality_level, warranty["standard"])
    
    def _calculate_confidence(self, input_data: Dict) -> float:
        """Calculate confidence level of the estimate"""
        confidence = 0.5  # Base confidence
        
        # More detailed input increases confidence
        if input_data.get("area_sqft"):
            confidence += 0.2
        if input_data.get("rooms"):
            confidence += 0.1
        if input_data.get("bathrooms"):
            confidence += 0.1
        if input_data.get("custom_requirements"):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_price_range(self, estimated_cost: int) -> Dict[str, int]:
        """Calculate price range for the estimate"""
        # Typically ±15% variation
        return {
            "min": int(estimated_cost * 0.85),
            "max": int(estimated_cost * 1.15)
        }
    
    def estimate_roi_from_renovation(self, property_value: int, renovation_cost: int, renovation_type: str) -> Dict[str, Any]:
        """Estimate ROI from renovation"""
        # Value increase percentages based on renovation type
        value_increase_pct = {
            "minimal": 0.05,  # 5% increase
            "moderate": 0.12,  # 12% increase
            "extensive": 0.20  # 20% increase
        }
        
        increase_pct = value_increase_pct.get(renovation_type, 0.10)
        value_increase = property_value * increase_pct
        
        roi_percentage = ((value_increase - renovation_cost) / renovation_cost) * 100
        
        return {
            "property_value_before": property_value,
            "renovation_cost": renovation_cost,
            "estimated_value_after": property_value + value_increase,
            "value_increase": value_increase,
            "roi_percentage": round(roi_percentage, 2),
            "payback_period_years": round(renovation_cost / (value_increase / 5), 1),  # Assuming 5-year hold
            "recommendation": "Recommended" if roi_percentage > 20 else "Consider carefully"
        }