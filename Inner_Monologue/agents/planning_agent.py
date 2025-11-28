"""Planning agent powered by Gemini LLM."""
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import settings
from tools import (
    RecognizeObjectsTool,
    LowLevelPolicyTool,
    DetectSuccessTool,
    DescribeSceneTool
)

logger = logging.getLogger(__name__)


class PlanningAgent:
    """
    Planning agent that uses Gemini LLM for high-level task planning.
    
    This agent orchestrates tool usage and generates plans based on
    scene observations and task goals.
    """
    
    def __init__(self):
        """Initialize the planning agent with Gemini."""
        print("[DEBUG] PlanningAgent.__init__() starting...")
        
        # Debug API key
        api_key = settings.google_api_key
        print(f"[DEBUG] Using API key: {api_key[:10]}...{api_key[-5:] if len(api_key) > 15 else '***'}")
        print(f"[DEBUG] Model: {settings.gemini_model}")
        
        # Configure Gemini
        genai.configure(api_key=settings.google_api_key)
        
        # Initialize LangChain Gemini integration
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_tokens,
            google_api_key=settings.google_api_key,
            convert_system_message_to_human=True  # Fix for SystemMessage compatibility
        )
        
        # Initialize tools
        self.tools = {
            "recognize_objects": RecognizeObjectsTool(),
            "low_level_policy": LowLevelPolicyTool(),
            "detect_success": DetectSuccessTool(),
            "describe_scene": DescribeSceneTool(),
        }
        
        # System prompt for the planning agent
        self.system_prompt = self._create_system_prompt()
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        print("[DEBUG] PlanningAgent initialization complete!")
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the planning agent."""
        return """You are an AI planning agent for a robot packing task in PyBullet simulation.

You will receive detailed object information including:
- Object names, center coordinates (x, y, z), and dimensions (width, length, height)
- Container bounds (for basket) showing available space
- Object types (movable_object vs container)

Your role:
- Generate a sequence of pick_and_place commands to pack boxes into the basket
- Use the format: pick_and_place(object_name, basket_center + (x_offset, y_offset))
- Use the actual basket center coordinates provided in the context
- Choose x_offset and y_offset to space out boxes and avoid collisions
- Consider object dimensions when planning placement
- Ensure objects fit within basket boundaries
- Space boxes appropriately based on their sizes

Example response format:
pick_and_place(red_box, basket_center + (-0.08, -0.08))
pick_and_place(blue_box, basket_center + (0.08, -0.08))
pick_and_place(green_box, basket_center + (-0.08, 0.08))
pick_and_place(yellow_box, basket_center + (0.08, 0.08))

Be strategic about positioning:
- Use actual object coordinates to understand current layout
- Consider object dimensions for optimal packing
- Avoid overlapping placements
- Maximize space utilization within basket boundaries
"""
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format JSON schema for prompt."""
        import json
        return json.dumps(schema, indent=2)
    
    def plan(
        self,
        task_goal: str,
        scene_context: Optional[Dict[str, Any]] = None,
        max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a plan to achieve the task goal.
        
        Args:
            task_goal: Description of the task to accomplish
            scene_context: Optional initial scene context
            max_steps: Maximum planning steps
            
        Returns:
            Dict with plan and reasoning
        """
        print(f"\n[DEBUG] PlanningAgent.plan() called - {task_goal}")
        
        # Build prompt
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Task Goal: {task_goal}\n\nGenerate a plan to achieve this goal.")
        ]
        
        # Add conversation history
        for i, msg in enumerate(self.conversation_history[-5:]):  # Last 5 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        try:
            # Get response from Gemini
            response = self.llm.invoke(messages)
            
            # Check for empty response 
            if not response or not response.content or response.content.strip() == "":
                print(f"[WARNING] Empty response from Gemini, retrying...")
                # Try once more with simplified prompt
                simplified_messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=f"Task Goal: {task_goal}\n\nGenerate a plan to achieve this goal.")
                ]
                response = self.llm.invoke(simplified_messages)
                
                # If still empty, return default
                if not response or not response.content or response.content.strip() == "":
                    print(f"[WARNING] Second attempt also empty, returning default response")
                    return {
                        "plan": "CONTINUE",
                        "success": True,
                        "raw_response": "",
                        "model": "gemini-2.5-flash"
                    }
            
            # Update conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })
            
            result = {
                "plan": response.content,
                "reasoning": "Generated by Gemini planning agent",
                "model": settings.gemini_model
            }
            return result
            
        except IndexError as e:
            print(f"\n[WARNING] Gemini IndexError (empty response parts): {e}")
            print(f"[WARNING] Returning fallback response")
            return {
                "plan": "CONTINUE",
                "success": True,
                "raw_response": "IndexError_fallback",
                "model": "gemini-2.5-flash"
            }
            
        except Exception as e:
            print(f"\n[ERROR] Exception in planning: {type(e).__name__}: {e}")
            
            # Add stack trace for debugging
            import traceback
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            
            # Try to get more details about the error
            if hasattr(e, 'response'):
                print(f"[ERROR] Response status: {getattr(e.response, 'status_code', 'N/A')}")
                print(f"[ERROR] Response text: {getattr(e.response, 'text', 'N/A')}")
            
            logger.error(f"Error in planning: {e}")
            return {
                "plan": None,
                "error": str(e),
                "reasoning": "Failed to generate plan"
            }
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
        """
        print(f"\n[DEBUG] PlanningAgent.execute_tool() called")
        print(f"[DEBUG] Tool name: {tool_name}")
        print(f"[DEBUG] Tool kwargs: {kwargs}")
        print(f"[DEBUG] Available tools: {list(self.tools.keys())}")
        
        if tool_name not in self.tools:
            print(f"[ERROR] Tool '{tool_name}' not found in available tools")
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
        
        print(f"[DEBUG] Executing tool '{tool_name}'...")
        tool = self.tools[tool_name]
        print(f"[DEBUG] Tool instance: {type(tool).__name__}")
        
        try:
            result = tool.execute(**kwargs)
            print(f"[DEBUG] Tool execution completed")
            print(f"[DEBUG] Result success: {result.success}")
            print(f"[DEBUG] Result data type: {type(result.data)}")
            print(f"[DEBUG] Result error: {result.error}")
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "metadata": result.metadata
            }
        except Exception as e:
            print(f"[ERROR] Tool execution failed: {type(e).__name__}: {e}")
            return {
                "success": False,
                "error": f"Tool execution failed: {str(e)}"
            }
    
    def reflect(self, observation: str, previous_plan: str) -> str:
        """
        Reflect on observations and previous plan to improve.
        
        Args:
            observation: Current observation
            previous_plan: Previous plan that was executed
            
        Returns:
            Reflection and updated reasoning
        """
        print(f"\n[DEBUG] PlanningAgent.reflect() called")
        print(f"[DEBUG] Observation length: {len(observation)}")
        print(f"[DEBUG] Previous plan length: {len(previous_plan)}")
        
        prompt = f"""Reflect on the following:

Previous Plan:
{previous_plan}

Current Observation:
{observation}

Provide:
1. What worked well?
2. What didn't work?
3. What should be changed?
4. Updated strategy
"""
        
        print(f"[DEBUG] Reflection prompt created (length: {len(prompt)})")
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        
        print(f"[DEBUG] Calling Gemini for reflection...")
        
        try:
            response = self.llm.invoke(messages)
            print(f"[DEBUG] Reflection response received (length: {len(response.content)})")
            print(f"[DEBUG] Reflection preview: {response.content[:200]}...")
            return response.content
        except Exception as e:
            print(f"[ERROR] Reflection failed: {type(e).__name__}: {e}")
            logger.error(f"Error in reflection: {e}")
            return f"Reflection error: {e}"


