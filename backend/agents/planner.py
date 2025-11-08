# agents/planner.py - Planner/Task Decomposer Agent

from typing import Dict, Any, Optional, List
from agents.base import BaseAgent, AgentResponse, AgentType, Task
import uuid
import networkx as nx

class PlannerAgent(BaseAgent):
    """
    Planner Agent - Decomposes complex queries into ordered tasks
    Creates execution plans with dependencies
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Planner", AgentType.PLANNER, config)
        
        # Task templates for common scenarios
        self.task_templates = {
            "property_search_and_report": [
                {"type": AgentType.STRUCTURED_DATA, "desc": "Search properties in database"},
                {"type": AgentType.RAG, "desc": "Get additional property details from documents"},
                {"type": AgentType.WEB_RESEARCH, "desc": "Fetch market rates and neighborhood info"},
                {"type": AgentType.REPORT_GENERATION, "desc": "Generate comprehensive report", "deps": [0, 1, 2]}
            ],
            "renovation_and_comparison": [
                {"type": AgentType.STRUCTURED_DATA, "desc": "Fetch property details"},
                {"type": AgentType.RENOVATION_ESTIMATION, "desc": "Estimate renovation costs", "deps": [0]},
                {"type": AgentType.WEB_RESEARCH, "desc": "Get market renovation rates"},
                {"type": AgentType.REPORT_GENERATION, "desc": "Create comparison report", "deps": [1, 2]}
            ],
            "market_analysis": [
                {"type": AgentType.WEB_RESEARCH, "desc": "Gather market trends"},
                {"type": AgentType.STRUCTURED_DATA, "desc": "Get historical property data"},
                {"type": AgentType.RAG, "desc": "Analyze market documents", "deps": [0]},
                {"type": AgentType.REPORT_GENERATION, "desc": "Generate market analysis", "deps": [0, 1, 2]}
            ]
        }
    
    async def execute(self, input_data: Dict[str, Any], context: Optional[Dict] = None) -> AgentResponse:
        """
        Create an execution plan for complex queries
        """
        try:
            query = input_data.get("query", "")
            intent_data = input_data.get("intent_data", {})
            
            # Analyze query complexity
            complexity = self._analyze_complexity(query, intent_data)
            
            # Create task list
            tasks = self._create_task_list(query, intent_data, complexity)
            
            # Order tasks based on dependencies
            ordered_tasks = self._order_tasks(tasks)
            
            # Create execution plan
            execution_plan = {
                "tasks": ordered_tasks,
                "complexity": complexity,
                "estimated_time": self._estimate_execution_time(ordered_tasks),
                "parallelizable": self._identify_parallel_tasks(ordered_tasks)
            }
            
            return AgentResponse(
                success=True,
                data=execution_plan,
                message=f"Created execution plan with {len(ordered_tasks)} tasks",
                metadata={"complexity_score": complexity}
            )
            
        except Exception as e:
            return AgentResponse(
                success=False,
                data=None,
                message=f"Planning failed: {str(e)}"
            )
    
    def _analyze_complexity(self, query: str, intent_data: Dict) -> float:
        """Analyze query complexity"""
        complexity_score = 0.0
        
        # Check for multiple intents
        if "and" in query.lower() or "then" in query.lower():
            complexity_score += 0.3
        
        # Check for multiple entities
        slots = intent_data.get("slots", {})
        complexity_score += len(slots) * 0.1
        
        # Check for specific complex keywords
        complex_keywords = ["compare", "analyze", "evaluate", "assess", "detailed"]
        for keyword in complex_keywords:
            if keyword in query.lower():
                complexity_score += 0.2
        
        # Check query length
        word_count = len(query.split())
        if word_count > 20:
            complexity_score += 0.2
        
        return min(complexity_score, 1.0)
    
    def _create_task_list(self, query: str, intent_data: Dict, complexity: float) -> List[Task]:
        """Create list of tasks based on query analysis"""
        tasks = []
        query_lower = query.lower()
        
        # Determine which template to use
        template_key = self._select_template(query_lower, intent_data)
        
        if template_key in self.task_templates:
            template_tasks = self.task_templates[template_key]
            
            for i, task_template in enumerate(template_tasks):
                task = Task(
                    id=str(uuid.uuid4())[:8],
                    type=task_template["type"],
                    description=task_template["desc"],
                    params=self._extract_task_params(task_template["type"], intent_data),
                    dependencies=[tasks[dep].id for dep in task_template.get("deps", [])]
                )
                tasks.append(task)
        else:
            # Create custom task list for non-template queries
            tasks = self._create_custom_tasks(query, intent_data)
        
        return tasks
    
    def _select_template(self, query: str, intent_data: Dict) -> str:
        """Select appropriate task template"""
        intent = intent_data.get("intent", "")
        
        if "report" in query and "search" in query:
            return "property_search_and_report"
        elif "renovation" in query and ("compare" in query or "comparison" in query):
            return "renovation_and_comparison"
        elif "market" in query and ("analysis" in query or "trend" in query):
            return "market_analysis"
        
        return ""
    
    def _create_custom_tasks(self, query: str, intent_data: Dict) -> List[Task]:
        """Create custom task list for non-template queries"""
        tasks = []
        target_agents = intent_data.get("target_agents", [])
        
        # Always start with memory check
        tasks.append(Task(
            id=str(uuid.uuid4())[:8],
            type=AgentType.MEMORY,
            description="Check user preferences and history",
            params={"action": "retrieve"},
            dependencies=[]
        ))
        
        # Add tasks based on target agents
        for agent_type in target_agents:
            if agent_type == AgentType.MEMORY:
                continue  # Already added
            
            task = Task(
                id=str(uuid.uuid4())[:8],
                type=agent_type,
                description=self._get_task_description(agent_type),
                params=self._extract_task_params(agent_type, intent_data),
                dependencies=[tasks[0].id] if tasks else []
            )
            tasks.append(task)
        
        # Add report generation if multiple data sources
        if len(tasks) > 2:
            tasks.append(Task(
                id=str(uuid.uuid4())[:8],
                type=AgentType.REPORT_GENERATION,
                description="Compile results into report",
                params={"format": "pdf"},
                dependencies=[t.id for t in tasks[1:]]
            ))
        
        return tasks
    
    def _get_task_description(self, agent_type: AgentType) -> str:
        """Get default description for agent type"""
        descriptions = {
            AgentType.STRUCTURED_DATA: "Query database for property information",
            AgentType.RAG: "Search and retrieve relevant documents",
            AgentType.WEB_RESEARCH: "Fetch external market data",
            AgentType.RENOVATION_ESTIMATION: "Calculate renovation costs",
            AgentType.REPORT_GENERATION: "Generate detailed report",
            AgentType.MEMORY: "Access user preferences and history"
        }
        return descriptions.get(agent_type, "Execute task")
    
    def _extract_task_params(self, agent_type: AgentType, intent_data: Dict) -> Dict[str, Any]:
        """Extract relevant parameters for each task"""
        slots = intent_data.get("slots", {})
        params = {}
        
        if agent_type == AgentType.STRUCTURED_DATA:
            params = {
                "filters": {
                    "location": slots.get("location"),
                    "bhk": slots.get("bhk"),
                    "budget": slots.get("budget"),
                    "property_type": slots.get("property_type")
                }
            }
        elif agent_type == AgentType.RENOVATION_ESTIMATION:
            params = {
                "area_sqft": slots.get("area_sqft"),
                "rooms": slots.get("rooms", slots.get("bhk"))
            }
        elif agent_type == AgentType.WEB_RESEARCH:
            params = {
                "location": slots.get("location"),
                "query_type": "market_rates"
            }
        
        return params
    
    def _order_tasks(self, tasks: List[Task]) -> List[Task]:
        """Order tasks based on dependencies using topological sort"""
        if not tasks:
            return []
        
        # Create dependency graph
        G = nx.DiGraph()
        
        for task in tasks:
            G.add_node(task.id, task=task)
            for dep_id in task.dependencies or []:
                G.add_edge(dep_id, task.id)
        
        # Topological sort
        try:
            ordered_ids = list(nx.topological_sort(G))
            ordered_tasks = [G.nodes[task_id]["task"] for task_id in ordered_ids]
            return ordered_tasks
        except nx.NetworkXError:
            # If there's a cycle, return tasks as-is
            self.logger.warning("Circular dependency detected in tasks")
            return tasks
    
    def _estimate_execution_time(self, tasks: List[Task]) -> float:
        """Estimate total execution time in seconds"""
        time_estimates = {
            AgentType.STRUCTURED_DATA: 0.5,
            AgentType.RAG: 2.0,
            AgentType.WEB_RESEARCH: 3.0,
            AgentType.RENOVATION_ESTIMATION: 1.0,
            AgentType.REPORT_GENERATION: 2.5,
            AgentType.MEMORY: 0.2
        }
        
        total_time = sum(time_estimates.get(task.type, 1.0) for task in tasks)
        return total_time
    
    def _identify_parallel_tasks(self, tasks: List[Task]) -> List[List[str]]:
        """Identify tasks that can be executed in parallel"""
        parallel_groups = []
        
        # Group tasks by dependency level
        level_groups = {}
        for task in tasks:
            level = len(task.dependencies or [])
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(task.id)
        
        # Tasks at the same dependency level can run in parallel
        for level in sorted(level_groups.keys()):
            if len(level_groups[level]) > 1:
                parallel_groups.append(level_groups[level])
        
        return parallel_groups