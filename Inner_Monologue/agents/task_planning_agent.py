"""Task Planning Agent with Gemini LLM."""
import logging
import json
import os
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Load .env from Inner_Monologue directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# Get API key from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class TaskPlanningAgent:
    """
    Planning agent that uses Gemini to reason through robotic manipulation tasks.
    Decides which tools to call and with what parameters based on task context.
    """
    
    def __init__(self, tools: Dict[str, Any]):
        """Initialize with available tools."""
        self.tools = tools
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Configure generation settings with higher output token limit
        generation_config = {
            "temperature": 0.7,
            "max_output_tokens": 8192,  # Increased to handle complex responses
        }
        
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            generation_config=generation_config
        )
        self.chat = None
        self.system_prompt = self._create_system_prompt()
        
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt explaining all tools and workflow."""
        return """You are an intelligent robotic planning agent controlling a robot arm in a simulation.

Your task is to pack boxes into a basket by making strategic tool calls.

AVAILABLE TOOLS:

1. recognize_objects(fov: float)   
   - Returns list of all object names in the scene
   - Call this FIRST to understand what objects exist
   - if you think the results are incomplete, try increasing fov (field of view) to see more maximum fov value is 140

2. detect(object_name: str)
   - Returns 3D center coordinates (x, y, z) of specified object in meters
   - Use before picking to know where the object is
   - Use for basket to know destination center
   - CRITICAL: Be very descriptive with object_name you pass to the detection (e.g., "a small red box", "a rectangular brown basket")
   - CRITICAL: When detecting the basket, ALWAYS use The following name for "A brown rectangular basket on a light brown table"
   - CRITICAL: When detecting any box, ALWAYS use The following name for "A small {color} box" where {color} is the color of the box (red, blue, green, yellow)

3. pick(object_name: str, x: float, y: float, z: float)
   - Moves gripper to (x,y,z) and picks up the object
   - Coordinates MUST come from detect() call
   - Only one object can be held at a time

4. place(x: float, y: float, theta: float)
   - Places currently held object at (x, y) with rotation theta (radians)
   - You must predict good placement coordinates
   - theta=0.0 for no rotation, theta=1.57 for 90 degrees

5. detect_success(object_name: str, target_x: float, target_y: float, target_theta: float, context: dict)
   - Validates if object was placed correctly
   - Returns success (within threshold) or failure
   - Call AFTER every place operation

6. describe_scene(task_context: str, last_action: str, context: dict)
   - ASKS A HUMAN to describe current scene state
   - Human will tell you where objects are placed, space remaining, any issues
   - Use this feedback to decide next action
   - Call after detect_success to understand scene state

CRITICAL NOTE:
    - Never call describe scene instead of recognize objects!
    - describe_scene is ONLY for getting human feedback on scene state after actions

WORKFLOW FOR EACH BOX:

1. detect(box_name) -> get box 3D position
2. detect("A rectangular brown basket") -> get basket center position
3. pick(box_name, x, y, z) -> pick at detected coordinates
4. place(x, y, theta) -> predict placement inside basket
   - Use offsets from basket center to avoid collisions
   - Leave space for remaining boxes
   - Start with small offsets like ±0.04 or ±0.05 from center
5. detect_success(...) -> check if placement succeeded
6. describe_scene(...) -> get human feedback on scene state

ON FAILURE - REPLAN CLEVERLY:
- If placement fails (collision), THINK about why
- Adjust coordinates: try different offset direction or larger offset
- Try rotation if space seems tight
- Use feedback from describe_scene to understand available space
- Keep trying with adjusted params until success or give up after 3-4 attempts

PLACEMENT STRATEGY:
- NEVER place exactly at basket center - leave room for other boxes
- Use offsets from basket center (e.g., center_x ± 0.05, center_y ± 0.04)
- First box: one corner/edge area
- Subsequent boxes: different areas to spread them out
- If collision: move AWAY from placed boxes

RESPONSE FORMAT (JSON only):

IMPORTANT: Use actual computed numbers, NOT arithmetic expressions!
- WRONG: "x": 0.7 + 0.05
- CORRECT: "x": 0.75

{
  "reasoning": "Your thought process - what you're doing and why",
  "tool_call": {
    "tool": "tool_name",
    "params": {"x": 0.75, "y": 0.04, "theta": 0.0}
  },
  "strategy": "Brief strategy note"
}

When task complete:
{
  "reasoning": "All boxes successfully packed",
  "task_complete": true
}

When giving up:
{
  "reasoning": "Explanation of why giving up after multiple failures",
  "task_failed": true
}"""

    def start_conversation(self, task: str):
        """Start a new conversation with the agent."""
        self.chat = self.model.start_chat(history=[])
        initial_message = f"{self.system_prompt}\n\nTASK: {task}\n\nBegin by calling recognize_objects() to see what's in the scene."
        return self.get_next_action(initial_message)
    
    def _fix_json_arithmetic(self, text: str) -> str:
        """Fix arithmetic expressions in JSON (e.g., 0.7 + 0.05 -> 0.75)."""
        import re
        # Find patterns like: number + number or number - number
        pattern = r'(\d+\.?\d*)\s*([+\-])\s*(\d+\.?\d*)'
        
        def eval_match(match):
            num1 = float(match.group(1))
            op = match.group(2)
            num2 = float(match.group(3))
            if op == '+':
                return str(num1 + num2)
            else:
                return str(num1 - num2)
        
        # Keep replacing until no more matches
        prev_text = ""
        while prev_text != text:
            prev_text = text
            text = re.sub(pattern, eval_match, text)
        return text
    
    def get_next_action(self, context: str) -> Dict[str, Any]:
        """Get next action from the agent based on current context."""
        try:
            response = self.chat.send_message(context)
            response_text = response.text.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {response_text[:500]}...")
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Try to parse JSON response
            try:
                action = json.loads(response_text)
                return action
            except json.JSONDecodeError as e:
                # Try fixing arithmetic expressions and parse again
                fixed_text = self._fix_json_arithmetic(response_text)
                try:
                    action = json.loads(fixed_text)
                    return action
                except json.JSONDecodeError as e2:
                    # Log the problematic text for debugging
                    logger.error(f"JSON parse error at position {e2.pos}: {e2.msg}")
                    logger.error(f"Problematic text (first 1000 chars): {response_text[:1000]}")
                    
                    # If still not valid JSON, return as reasoning
                    return {
                        "reasoning": response_text[:500],  # Truncate to avoid huge error messages
                        "error": f"Response was not valid JSON: {e2.msg} at position {e2.pos}"
                    }
                
        except Exception as e:
            logger.error(f"Error getting next action: {e}")
            return {
                "error": str(e),
                "reasoning": "Failed to get response from agent"
            }
    
    def provide_feedback(self, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Provide tool execution result as feedback to agent."""
        feedback_message = f"""Tool '{tool_name}' executed.

Result: {json.dumps(result, indent=2)}

Based on this result, what should be the next action? Remember to follow the workflow and validate each step."""
        
        return self.get_next_action(feedback_message)

