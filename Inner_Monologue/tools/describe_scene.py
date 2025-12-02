"""Scene description tool - prompts user to describe the scene state."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DescribeSceneTool:
    """
    Tool that prompts a human user to describe the current scene state.
    
    Returns human observations about:
    - Where objects were placed in the destination
    - Space remaining for upcoming objects
    - Any issues or observations
    """
    
    def __init__(self, environment, llm_client=None):
        """Initialize scene description tool with environment."""
        self.environment = environment
        self.llm_client = llm_client
        self.wrapper = None
        self.name = "describe_scene"
        self.description = "Ask human to describe current scene state"
    
    def set_wrapper(self, wrapper):
        """Set the environment wrapper for getting scene info."""
        self.wrapper = wrapper
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "describe_scene",
                "description": "Ask a human observer to describe the current scene: object placements, space remaining, any issues.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_context": {
                            "type": "string",
                            "description": "Brief description of current task"
                        },
                        "last_action": {
                            "type": "string", 
                            "description": "The last action performed"
                        },
                        "context": {
                            "type": "object",
                            "description": "Scene context (optional)"
                        }
                    },
                    "required": ["task_context", "last_action"]
                }
            }
        }
    
    def execute(self, task_context: str, last_action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prompt user to describe the scene state.
        
        Waits for user input and returns their description to the agent.
        """
        print(f"\n{'='*60}")
        print("🔍 SCENE DESCRIPTION REQUEST")
        print(f"{'='*60}")
        print(f"Task: {task_context}")
        print(f"Last Action: {last_action}")
        print(f"{'='*60}")
        print("\nPlease describe the current scene:")
        print("  - Where was the object placed in the basket?")
        print("  - How much space is left for remaining boxes?")
        print("  - Any collisions, issues, or observations?")
        print(f"{'='*60}\n")
        
        try:
            # Get user input - wait indefinitely
            user_description = input("👤 Your description: ").strip()
            
            if not user_description:
                user_description = "No description provided"
            
            print(f"\n✅ Description received: {user_description[:50]}...")
            print(f"{'='*60}\n")
            
            return {
                "success": True,
                "feedback": user_description,
                "description": user_description,
                "task_context": task_context,
                "last_action": last_action
            }
            
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            return {
                "success": False,
                "feedback": "User interrupted the description",
                "description": "interrupted"
            }
        except Exception as e:
            logger.error(f"Error getting scene description: {e}")
            return {
                "success": False,
                "feedback": f"Error: {str(e)}",
                "description": f"error: {str(e)}"
            }
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(
            task_context=kwargs.get("task_context", "packing boxes"),
            last_action=kwargs.get("last_action", "unknown"),
            context=kwargs.get("context", {})
        )
