"""Place tool for LLM to directly call."""
import logging
import time
from typing import Dict, Any
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class PlaceTool(BaseTool):
    """Tool for LLM to place objects directly."""
    
    def __init__(self, environment):
        """Initialize place tool with environment."""
        self.environment = environment
        self.name = "place"
        self.description = "Place a held object at specified coordinates"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "place",
                "description": "Place the currently held object at specified coordinates. X,Y coordinates are in meters, theta is rotation in radians.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "X coordinate to place the object (in meters)"
                        },
                        "y": {
                            "type": "number", 
                            "description": "Y coordinate to place the object (in meters)"
                        },
                        "theta": {
                            "type": "number",
                            "description": "Rotation angle in radians (0 = no rotation, 3.14159 = 180 degrees)"
                        }
                    },
                    "required": ["x", "y", "theta"]
                }
            }
        }
    
    def execute(self, x: float, y: float, theta: float) -> Dict[str, Any]:
        """Execute the place action using LLM-TAMP's original logic."""
        print(f"📍 Placing at ({x:.2f}, {y:.2f}, θ={theta:.2f})...")
        
        try:
            # Import LLM-TAMP action classes
            import sys
            from pathlib import Path
            llm_tamp_path = Path(__file__).parent.parent.parent.parent / "LLM-TAMP"
            if str(llm_tamp_path) not in sys.path:
                sys.path.insert(0, str(llm_tamp_path))
            
            from utils.tamp_util import Action, PrimitiveAction
            
            # Validate environment
            if not hasattr(self.environment, 'env'):
                return {
                    "success": False,
                    "feedback": "Environment not initialized",
                    "coordinates": {"x": x, "y": y, "theta": theta}
                }
            
            # Find which object is currently being held
            # Check if robot has any attachments
            if not hasattr(self.environment.env.robot, 'attachments_robot') or len(self.environment.env.robot.attachments_robot) == 0:
                return {
                    "success": False,
                    "feedback": "No object is currently being held by the robot",
                    "coordinates": {"x": x, "y": y, "theta": theta}
                }
            
            # Find the held object name
            held_object_name = None
            attachment = self.environment.env.robot.attachments_robot[0]  # Should be exactly one attachment
            
            # Search through objects to find which one matches the attachment
            for obj_name, obj_id in self.environment.env.objects.items():
                if obj_id == attachment.child:
                    held_object_name = obj_name
                    break
            
            if not held_object_name:
                return {
                    "success": False,
                    "feedback": "Could not identify the held object",
                    "coordinates": {"x": x, "y": y, "theta": theta}
                }
            
            # Create place action using the environment's existing primitive action
            place_primitive = self.environment.env._primitive_actions["place"]
            place_action = Action(
                primitive=place_primitive,
                obj_args=[held_object_name],
                param_args={"x": x, "y": y, "theta": theta}
            )
            
            success, feedback = self.environment.env.apply_action(place_action)
            
            # Add delay for better viewing of the motion
            time.sleep(0.8)
            
            result = {
                "success": success,
                "feedback": feedback,
                "coordinates": {"x": x, "y": y, "theta": theta},
                "placed_object": held_object_name
            }
            
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {status}: {feedback}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "feedback": f"Exception: {str(e)}",
                "coordinates": {"x": x, "y": y, "theta": theta}
            }
            print(f"   ❌ Error: {str(e)}")
            return error_result
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(
            x=kwargs.get("x", 0.0),
            y=kwargs.get("y", 0.0), 
            theta=kwargs.get("theta", 0.0)
        )