# orchestrator.py - Main orchestrator for agent coordination

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from agents import (
    AgentType,
    QueryRouterAgent,
    PlannerAgent,
    StructuredDataAgent,
    RAGAgent,
    WebResearchAgent,
    ReportGenerationAgent,
    RenovationEstimationAgent,
    MemoryAgent
)

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Main orchestrator that coordinates all agents
    Handles routing, planning, and execution of complex queries
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize all agents with configuration"""
        self.config = config
        
        # Initialize agents
        self.agents = {
            AgentType.QUERY_ROUTER: QueryRouterAgent(config.get("query_router")),
            AgentType.PLANNER: PlannerAgent(config.get("planner")),
            AgentType.STRUCTURED_DATA: StructuredDataAgent(config.get("structured_data")),
            AgentType.RAG: RAGAgent(config.get("rag")),
            AgentType.WEB_RESEARCH: WebResearchAgent(config.get("web_research")),
            AgentType.REPORT_GENERATION: ReportGenerationAgent(config.get("report_generation")),
            AgentType.RENOVATION_ESTIMATION: RenovationEstimationAgent(config.get("renovation")),
            AgentType.MEMORY: MemoryAgent(config.get("memory"))
        }
        
        # Execution history
        self.execution_history = []
        
    async def process_query(self, query: str, user_id: str = "default", session_id: str = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Get user context from memory
            user_context = await self._get_user_context(user_id)
            
            # Step 2: Route the query
            routing_result = await self.agents[AgentType.QUERY_ROUTER].execute({
                "query": query
            }, user_context)
            
            if not routing_result.success:
                return self._create_error_response("Failed to route query", routing_result.message)
            
            routing_data = routing_result.data
            
            # Step 3: Check if planning is needed
            if routing_data.get("requires_planning"):
                # Use planner for complex queries
                plan_result = await self.agents[AgentType.PLANNER].execute({
                    "query": query,
                    "intent_data": routing_data
                }, user_context)
                
                if not plan_result.success:
                    return self._create_error_response("Failed to create plan", plan_result.message)
                
                # Execute planned tasks
                execution_results = await self._execute_plan(plan_result.data, user_context)
            else:
                # Execute single or simple multi-agent flow
                execution_results = await self._execute_simple_flow(
                    routing_data["target_agents"],
                    routing_data,
                    query,
                    user_context
                )
            
            # Step 4: Update memory with interaction
            await self._update_memory(user_id, query, routing_data, execution_results)
            
            # Step 5: Compile final response
            response = await self._compile_response(
                query,
                routing_data,
                execution_results,
                user_context
            )
            
            # Log execution
            execution_time = (datetime.now() - start_time).total_seconds()
            self._log_execution(user_id, query, response, execution_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Orchestrator error: {str(e)}")
            return self._create_error_response("Processing failed", str(e))
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get user context from memory agent"""
        memory_result = await self.agents[AgentType.MEMORY].execute({
            "operation": "retrieve",
            "memory_type": "preferences",
            "user_id": user_id
        })
        
        context = {
            "user_id": user_id,
            "preferences": memory_result.data if memory_result.success else {}
        }
        
        return context
    
    async def _execute_plan(self, plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a planned sequence of tasks"""
        tasks = plan.get("tasks", [])
        results = {}
        
        for task in tasks:
            # Check dependencies
            deps_satisfied = all(
                dep_id in results and results[dep_id].get("success")
                for dep_id in task.dependencies or []
            )
            
            if not deps_satisfied:
                results[task.id] = {"success": False, "message": "Dependencies not satisfied"}
                continue
            
            # Execute task
            agent = self.agents.get(task.type)
            if agent:
                # Prepare input data with dependency results
                task_input = task.params.copy()
                for dep_id in task.dependencies or []:
                    if dep_id in results:
                        task_input[f"dep_{dep_id}"] = results[dep_id].get("data")
                
                result = await agent.execute(task_input, context)
                results[task.id] = {
                    "success": result.success,
                    "data": result.data,
                    "message": result.message,
                    "agent": task.type.value
                }
            else:
                results[task.id] = {"success": False, "message": f"Agent {task.type} not found"}
        
        return results
    
    async def _execute_simple_flow(
        self, 
        target_agents: List[AgentType], 
        routing_data: Dict,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute simple agent flow without planning"""
        results = {}
        
        for agent_type in target_agents:
            if agent_type == AgentType.MEMORY:
                continue  # Memory is handled separately
            
            agent = self.agents.get(agent_type)
            if not agent:
                continue
            
            # Prepare input based on agent type
            agent_input = self._prepare_agent_input(agent_type, routing_data, query)
            
            # Execute agent
            result = await agent.execute(agent_input, context)
            
            results[agent_type.value] = {
                "success": result.success,
                "data": result.data,
                "message": result.message,
                "citations": result.citations
            }
        
        return results
    
    def _prepare_agent_input(self, agent_type: AgentType, routing_data: Dict, query: str) -> Dict[str, Any]:
        """Prepare input data for specific agent type"""
        slots = routing_data.get("slots", {})
        
        if agent_type == AgentType.STRUCTURED_DATA:
            return {
                "query_type": "search",
                "filters": slots,
                "limit": 10
            }
        
        elif agent_type == AgentType.RAG:
            return {
                "query": query,
                "top_k": 5,
                "filters": slots,
                "synthesis": True
            }
        
        elif agent_type == AgentType.WEB_RESEARCH:
            return {
                "research_type": "market_rates",
                "location": slots.get("location", ""),
                "query": query
            }
        
        elif agent_type == AgentType.RENOVATION_ESTIMATION:
            return {
                "area_sqft": slots.get("area_sqft", 1000),
                "rooms": slots.get("rooms", slots.get("bhk", 2)),
                "bathrooms": slots.get("bathrooms", 1),
                "renovation_type": "moderate",
                "quality_level": "standard"
            }
        
        elif agent_type == AgentType.REPORT_GENERATION:
            return {
                "report_type": "property_analysis",
                "data": {},
                "format": "pdf"
            }
        
        return {"query": query}
    
    async def _update_memory(self, user_id: str, query: str, routing_data: Dict, results: Dict):
        """Update memory with interaction data"""
        # Update search history
        await self.agents[AgentType.MEMORY].execute({
            "operation": "update",
            "memory_type": "search_history",
            "user_id": user_id,
            "data": [{
                "query": query,
                "intent": routing_data.get("intent"),
                "slots": routing_data.get("slots"),
                "timestamp": datetime.now().isoformat()
            }]
        })
        
        # Learn from interaction if property was selected
        if "structured_data" in results:
            properties = results["structured_data"].get("data", [])
            if properties and len(properties) > 0:
                await self.agents[AgentType.MEMORY].execute({
                    "operation": "update",
                    "memory_type": "saved_properties",
                    "user_id": user_id,
                    "data": properties[:3]  # Save top 3 properties
                })
    
    async def _compile_response(
        self, 
        query: str,
        routing_data: Dict,
        execution_results: Dict,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile final response from all agent results"""
        response = {
            "success": True,
            "query": query,
            "intent": routing_data.get("intent"),
            "response_type": "multi_agent",
            "data": {},
            "message": "",
            "citations": []
        }
        
        # Compile data from different agents
        if "structured_data" in execution_results:
            result = execution_results["structured_data"]
            if result.get("success"):
                response["data"]["properties"] = result.get("data", [])
                response["message"] += f"Found {len(result.get('data', []))} properties. "
        
        if "rag" in execution_results:
            result = execution_results["rag"]
            if result.get("success"):
                response["data"]["document_insights"] = result.get("data", {}).get("answer")
                response["citations"] = result.get("citations", [])
        
        if "web_research" in execution_results:
            result = execution_results["web_research"]
            if result.get("success"):
                response["data"]["market_data"] = result.get("data")
                response["message"] += "Retrieved current market data. "
        
        if "renovation_estimation" in execution_results:
            result = execution_results["renovation_estimation"]
            if result.get("success"):
                response["data"]["renovation_estimate"] = result.get("data")
                response["message"] += f"Renovation estimate: ₹{result.get('data', {}).get('total_cost', 0):,}. "
        
        if "report_generation" in execution_results:
            result = execution_results["report_generation"]
            if result.get("success"):
                response["data"]["report"] = result.get("data")
                response["message"] += "Report generated successfully. "
        
        # If no message generated, create a default one
        if not response["message"]:
            response["message"] = "Query processed successfully."
        
        return response
    
    def _create_error_response(self, error_type: str, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error_type": error_type,
            "message": error_message,
            "data": {}
        }
    
    def _log_execution(self, user_id: str, query: str, response: Dict, execution_time: float):
        """Log execution for debugging and analytics"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "query": query,
            "success": response.get("success"),
            "intent": response.get("intent"),
            "execution_time": execution_time
        }
        
        self.execution_history.append(log_entry)
        
        # Keep only last 100 entries
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        logger.info(f"Query processed in {execution_time:.2f}s for user {user_id}")
    
    async def generate_report_from_data(
        self, 
        report_type: str, 
        data: Dict[str, Any], 
        user_id: str = "default"
    ) -> Dict[str, Any]:
        """Generate a report from provided data"""
        result = await self.agents[AgentType.REPORT_GENERATION].execute({
            "report_type": report_type,
            "data": data,
            "format": "pdf"
        })
        
        return {
            "success": result.success,
            "report": result.data if result.success else None,
            "message": result.message
        }
    
    async def get_property_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized property recommendations"""
        # Get user preferences
        context = await self._get_user_context(user_id)
        preferences = context.get("preferences", {}).get("data", {})
        
        # Search based on preferences
        properties_result = await self.agents[AgentType.STRUCTURED_DATA].execute({
            "query_type": "search",
            "filters": {
                "location": preferences.get("locations", [""])[0] if preferences.get("locations") else None,
                "budget": preferences.get("budget_range", [0, 100000000])[1] if preferences.get("budget_range") else None,
                "bhk": preferences.get("bhk", [2])[0] if preferences.get("bhk") else None
            },
            "limit": 5
        }, context)
        
        if properties_result.success:
            # Get similar properties using RAG
            rag_result = await self.agents[AgentType.RAG].execute({
                "query": f"Properties similar to {preferences}",
                "top_k": 3,
                "synthesis": False
            }, context)
            
            return {
                "success": True,
                "recommendations": properties_result.data,
                "similar_properties": rag_result.data if rag_result.success else []
            }
        
        return {
            "success": False,
            "recommendations": [],
            "message": "Could not generate recommendations"
        }