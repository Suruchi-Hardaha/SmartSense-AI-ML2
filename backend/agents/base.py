# agents/base.py - Base agent class and utilities

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AgentType(Enum):
    QUERY_ROUTER = "query_router"
    PLANNER = "planner"
    STRUCTURED_DATA = "structured_data"
    RAG = "rag"
    WEB_RESEARCH = "web_research"
    REPORT_GENERATION = "report_generation"
    RENOVATION_ESTIMATION = "renovation_estimation"
    MEMORY = "memory"

@dataclass
class AgentResponse:
    """Standard response format for all agents"""
    success: bool
    data: Any
    message: str
    metadata: Optional[Dict] = None
    citations: Optional[List[Dict]] = None

@dataclass
class Task:
    """Task representation for the planner agent"""
    id: str
    type: AgentType
    description: str
    params: Dict[str, Any]
    dependencies: List[str] = None
    result: Optional[Any] = None

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, agent_type: AgentType, config: Dict[str, Any] = None):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """Execute the agent's main task"""
        pass
    
    def validate_input(self, input_data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate that required fields are present in input"""
        missing = [field for field in required_fields if field not in input_data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return True
    
    def log_execution(self, input_data: Dict[str, Any], response: AgentResponse):
        """Log agent execution for debugging"""
        self.logger.info(f"Agent {self.name} executed. Success: {response.success}")
        if not response.success:
            self.logger.error(f"Agent {self.name} error: {response.message}")