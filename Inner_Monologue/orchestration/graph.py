"""LangGraph orchestration graph for Inner Monologue agent system."""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from agents.planning_agent import PlanningAgent
from tools import (
    RecognizeObjectsTool,
    LowLevelPolicyTool,
    DetectSuccessTool,
    DescribeSceneTool
)

import logging
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent orchestration graph."""
    messages: Annotated[list, add_messages]
    task_goal: str
    scene_context: dict
    iteration: int
    max_iterations: int
    success: bool
    reflection: str
    last_action: dict  # Track last executed action for success detection
    action_success: bool  # Result of success detection
    should_replan: bool  # Whether to replan based on feedback
    human_interaction_pending: bool  # Whether waiting for human answer


class InnerMonologueGraph:
    """
    LangGraph-based orchestration for Inner Monologue agent system.
    
    This graph coordinates:
    1. Scene observation and description
    2. Object recognition
    3. Planning with Gemini
    4. Action execution
    5. Success detection
    6. Reflection and adaptation
    """
    
    def __init__(self, planning_agent: PlanningAgent, environment=None):
        """
        Initialize the orchestration graph.
        
        Args:
            planning_agent: Planning agent instance
            environment: PyBullet environment instance
        """
        print("[DEBUG] InnerMonologueGraph.__init__() starting...")
        
        self.planning_agent = planning_agent
        self.environment = environment
        print(f"[DEBUG] Planning agent: {type(planning_agent).__name__}")
        print(f"[DEBUG] Environment: {type(environment).__name__ if environment else 'None'}")
        
        # Initialize tools
        print("[DEBUG] Initializing tools...")
        self.base_tools = {
            "recognize_objects": RecognizeObjectsTool(),
            "low_level_policy": LowLevelPolicyTool(),
            "detect_success": DetectSuccessTool(environment=environment),
            "describe_scene": DescribeSceneTool(environment=environment),
        }
        print(f"[DEBUG] Base tools initialized: {list(self.base_tools.keys())}")
        
        # Set environment references
        if environment:
            print("[DEBUG] Setting environment references for tools...")
            self.base_tools["detect_success"].set_environment(environment)
            self.base_tools["describe_scene"].set_environment(environment)
        
        # Create LangChain tool wrappers
        print("[DEBUG] Creating LangChain tool wrappers...")
        self.langchain_tools = self._create_langchain_tools()
        print(f"[DEBUG] LangChain tools created: {len(self.langchain_tools)}")
        
        # Create tool node - temporarily skip ToolNode to avoid version issues
        # self.tool_node = ToolNode(list(self.langchain_tools.values()))
        self.tool_node = None  # We'll handle tools manually in nodes
        print("[DEBUG] Tool node: None (manual handling)")
        
        # Build graph
        print("[DEBUG] Building graph...")
        self.graph = self._build_graph()
        print("[DEBUG] Graph built successfully")
        print("[DEBUG] InnerMonologueGraph initialization complete!")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("observe", self._observe_node)
        workflow.add_node("plan", self._plan_node)
        workflow.add_node("human_interact", self._human_interact_node)  # New: human interaction
        workflow.add_node("execute", self._execute_node)
        workflow.add_node("check_success", self._check_success_node)  # After each action
        workflow.add_node("reflect", self._reflect_node)
        
        # Set entry point
        workflow.set_entry_point("observe")
        
        # Add edges
        workflow.add_edge("observe", "plan")
        workflow.add_edge("plan", "human_interact")  # Always check for human interaction after plan
        workflow.add_conditional_edges(
            "human_interact",
            self._should_replan_from_human,
            {
                "replan": "plan",
                "continue": "execute"
            }
        )
        workflow.add_edge("execute", "check_success")  # Always check success after execute
        workflow.add_conditional_edges(
            "check_success",
            self._should_continue_after_success,
            {
                "success": END,
                "replan": "plan",  # Replan if action failed
                "reflect": "reflect",
                "continue": "observe"
            }
        )
        workflow.add_edge("reflect", "plan")
        
        return workflow.compile()
    
    def _create_langchain_tools(self) -> dict:
        """Create LangChain tool wrappers for our custom tools."""
        langchain_tools = {}
        
        # For now, we'll skip the tool wrapper creation to avoid version issues
        # and handle tools manually in the nodes
        
        return langchain_tools
    
    def _observe_node(self, state: AgentState) -> AgentState:
        """Observe the scene and describe it."""
        logger.info("Observing scene...")
        
        # Use describe_scene tool
        scene_result = self.base_tools["describe_scene"].execute()
        
        # Use recognize_objects tool
        objects_result = self.base_tools["recognize_objects"].execute()
        
        # Update state
        state["scene_context"] = {
            "description": scene_result.data.get("description", ""),
            "objects": objects_result.data.get("objects", [])
        }
        
        # Add observation message
        observation_text = f"Scene: {state['scene_context']['description']}\n"
        observation_text += f"Objects: {len(state['scene_context']['objects'])} detected"
        
        state["messages"].append(HumanMessage(content=observation_text))
        
        return state
    
    def _plan_node(self, state: AgentState) -> AgentState:
        """Generate a plan using the planning agent."""
        print(f"\n[DEBUG] _plan_node() called")
        print(f"[DEBUG] Current iteration: {state.get('iteration', 0)}")
        print(f"[DEBUG] Max iterations: {state.get('max_iterations', 0)}")
        print(f"[DEBUG] Task goal: {state.get('task_goal', 'None')}")
        
        logger.info("Planning...")
        
        # Get plan from planning agent
        print("[DEBUG] Calling planning_agent.plan()...")
        plan_result = self.planning_agent.plan(
            task_goal=state["task_goal"],
            scene_context=state["scene_context"]
        )
        print(f"[DEBUG] Plan result received: {type(plan_result)}")
        print(f"[DEBUG] Plan result keys: {list(plan_result.keys()) if isinstance(plan_result, dict) else 'N/A'}")
        
        # Add plan message
        plan_content = plan_result.get('plan', 'No plan generated')
        reasoning = plan_result.get('reasoning', '')
        error = plan_result.get('error')
        
        if error:
            print(f"[ERROR] Plan generation failed: {error}")
            plan_text = f"Plan generation failed: {error}\n"
        else:
            plan_text = f"Plan: {plan_content}\n"
            
        plan_text += f"Reasoning: {reasoning}"
        
        print(f"[DEBUG] Adding plan message (length: {len(plan_text)})")
        state["messages"].append(AIMessage(content=plan_text))
        print(f"[DEBUG] Plan node complete, total messages: {len(state['messages'])}")
        
        return state
    
    def _should_use_tools(self, state: AgentState) -> Literal["use_tools", "continue"]:
        """Decide if tools should be used."""
        # Check if last message contains tool calls
        last_message = state["messages"][-1] if state["messages"] else None
        
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls"):
            if last_message.tool_calls:
                return "use_tools"
        
        return "continue"
    
    def _execute_node(self, state: AgentState) -> AgentState:
        """Execute planned actions."""
        logger.info("Executing actions...")
        
        # Increment iteration
        state["iteration"] += 1
        
        # TODO: Parse plan and execute low-level actions
        # For now, extract action info from last plan message
        # In real implementation, this would parse the plan and execute via low_level_policy tool
        
        # Store last action info for success detection
        # This should be extracted from the actual executed action
        state["last_action"] = {
            "action_type": "pick_and_place",  # Placeholder
            "picked_object": None,
            "place_target": None,
            "place_location": None
        }
        
        execution_message = "Action execution - placeholder (implement low_level_policy integration)"
        state["messages"].append(HumanMessage(content=execution_message))
        
        return state
    
    def _check_success_node(self, state: AgentState) -> AgentState:
        """Check if last action was successful (runs after each action)."""
        logger.info("Checking action success...")
        
        # Use detect_success tool with last action
        success_result = self.base_tools["detect_success"].execute(
            last_action=state.get("last_action"),
            action_type=state.get("last_action", {}).get("action_type"),
            picked_object=state.get("last_action", {}).get("picked_object"),
            place_target=state.get("last_action", {}).get("place_target"),
            place_location=state.get("last_action", {}).get("place_location")
        )
        
        # Store action success result
        state["action_success"] = success_result.data.get("action_success", False)
        
        # Check overall task goal (all objects in basket)
        if self.environment:
            goal_achieved, goal_feedback = self.environment.check_goal()
            state["success"] = goal_achieved
        else:
            state["success"] = False
        
        # Add success check message
        action_feedback = success_result.data.get("feedback", "")
        success_text = f"Action Success: {state['action_success']}\n"
        success_text += f"Action Feedback: {action_feedback}\n"
        success_text += f"Overall Goal: {state['success']}"
        
        state["messages"].append(HumanMessage(content=success_text))
        
        return state
    
    def _human_interact_node(self, state: AgentState) -> AgentState:
        """Handle human interaction after planning (runs after each plan step)."""
        logger.info("Checking for human interaction...")
        
        # Use describe_scene tool in interaction mode
        # First, get LLM's choice (this should come from the plan message)
        # For now, we'll prompt the LLM to choose
        
        # Check if there's a pending question waiting for answer
        scene_tool = self.base_tools["describe_scene"]
        
        if scene_tool.pending_question and not state.get("human_answer"):
            # Waiting for human answer
            state["human_interaction_pending"] = True
            state["messages"].append(HumanMessage(
                content=f"Pending question: {scene_tool.pending_question}\nWaiting for human answer..."
            ))
            return state
        
        # Get human answer from state if provided
        human_answer = state.get("human_answer")
        
        # Check interaction result
        interact_result = scene_tool.execute(
            mode="interact",
            llm_choice=state.get("llm_choice", "continue"),
            question=state.get("question"),
            human_answer=human_answer
        )
        
        # Check if should replan based on human feedback
        state["should_replan"] = interact_result.data.get("should_replan", False)
        state["human_interaction_pending"] = interact_result.data.get("waiting_for_answer", False)
        
        if interact_result.data.get("question"):
            state["messages"].append(HumanMessage(
                content=f"LLM Question: {interact_result.data['question']}"
            ))
        
        if interact_result.data.get("human_answer"):
            state["messages"].append(HumanMessage(
                content=f"Human Answer: {interact_result.data['human_answer']}"
            ))
        
        return state
    
    def _should_replan_from_human(self, state: AgentState) -> Literal["replan", "continue"]:
        """Decide if should replan based on human interaction."""
        if state.get("human_interaction_pending", False):
            # Still waiting for answer, don't proceed yet
            return "continue"  # Wait, don't replan yet
        
        if state.get("should_replan", False):
            logger.info("Replanning due to human feedback")
            return "replan"
        
        return "continue"
    
    def _should_continue_after_success(self, state: AgentState) -> Literal["success", "replan", "reflect", "continue"]:
        """Decide next step based on action success, overall success, and iterations."""
        # If overall goal achieved, end
        if state.get("success", False):
            return "success"
        
        # If last action failed, replan
        if not state.get("action_success", True):
            logger.info("Action failed, replanning...")
            return "replan"
        
        # If max iterations reached, reflect
        if state["iteration"] >= state["max_iterations"]:
            return "reflect"
        
        # Continue to next action
        return "continue"
    
    def _reflect_node(self, state: AgentState) -> AgentState:
        """Reflect on progress and adapt."""
        logger.info("Reflecting...")
        
        # Get reflection from planning agent
        last_plan = state["messages"][-2].content if len(state["messages"]) >= 2 else ""
        last_observation = state["messages"][-1].content if state["messages"] else ""
        
        reflection = self.planning_agent.reflect(
            observation=last_observation,
            previous_plan=last_plan
        )
        
        state["reflection"] = reflection
        state["messages"].append(AIMessage(content=f"Reflection: {reflection}"))
        
        return state
    
    def run(self, task_goal: str, max_iterations: int = 50) -> dict:
        """
        Run the agent orchestration.
        
        Args:
            task_goal: Task goal to achieve
            max_iterations: Maximum number of iterations
            
        Returns:
            Final state
        """
        print(f"\n[DEBUG] InnerMonologueGraph.run() called")
        print(f"[DEBUG] Task goal: {task_goal}")
        print(f"[DEBUG] Max iterations: {max_iterations}")
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=f"Task Goal: {task_goal}")],
            "task_goal": task_goal,
            "scene_context": {},
            "iteration": 0,
            "max_iterations": max_iterations,
            "success": False,
            "reflection": "",
            "last_action": {},
            "action_success": True,
            "should_replan": False,
            "human_interaction_pending": False
        }
        
        print(f"[DEBUG] Initial state created")
        print(f"[DEBUG] Starting graph execution...")
        
        try:
            final_state = self.graph.invoke(initial_state)
            print(f"[DEBUG] Graph execution completed successfully")
            print(f"[DEBUG] Final state type: {type(final_state)}")
            print(f"[DEBUG] Final iteration: {final_state.get('iteration', 'N/A')}")
            print(f"[DEBUG] Final success: {final_state.get('success', 'N/A')}")
            return final_state
        except Exception as e:
            print(f"[ERROR] Graph execution failed: {type(e).__name__}: {e}")
            raise

