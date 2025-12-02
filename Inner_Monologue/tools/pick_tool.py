"""Pick tool for LLM to directly call."""
import logging
import time
from typing import Dict, Any
from .base_tool import BaseTool

logger = logging.getLogger(__name__)


class PickTool(BaseTool):
    """Tool for LLM to pick objects directly."""
    
    def __init__(self, environment):
        """Initialize pick tool with environment."""
        self.environment = environment
        self.name = "pick"
        self.description = "Pick up an object from the environment"
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "pick",
                "description": "Pick up an object from the environment at specified 3D coordinates. The robot will move its gripper to the given position and grasp the object.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name of the object to pick up (e.g., 'red_box', 'blue_box')"
                        },
                        "x": {
                            "type": "number",
                            "description": "X coordinate of object center in meters"
                        },
                        "y": {
                            "type": "number",
                            "description": "Y coordinate of object center in meters"
                        },
                        "z": {
                            "type": "number",
                            "description": "Z coordinate of object center in meters"
                        }
                    },
                    "required": ["object_name", "x", "y", "z"]
                }
            }
        }
    
    def execute(self, object_name: str, x: float, y: float, z: float) -> Dict[str, Any]:
        """Execute the pick action using LLM-TAMP's original logic with 3D coordinates."""
        print(f"🤏 Picking {object_name} at ({x:.3f}, {y:.3f}, {z:.3f})...")
        
        try:
            # Import LLM-TAMP action classes
            import sys
            from pathlib import Path
            llm_tamp_path = Path(__file__).parent.parent.parent.parent / "LLM-TAMP"
            if str(llm_tamp_path) not in sys.path:
                sys.path.insert(0, str(llm_tamp_path))
            
            from utils.tamp_util import Action, PrimitiveAction
            
            # Validate object exists
            if not hasattr(self.environment, 'env') or not hasattr(self.environment.env, 'objects'):
                return {
                    "success": False,
                    "feedback": "Environment not initialized",
                    "object_name": object_name
                }
            
            if object_name not in self.environment.env.objects:
                available_objects = list(self.environment.env.objects.keys())
                return {
                    "success": False, 
                    "feedback": f"Object '{object_name}' not found. Available: {available_objects}",
                    "object_name": object_name
                }
            
            # Check if robot is already holding something
            if hasattr(self.environment.env.robot, 'attachments_robot') and len(self.environment.env.robot.attachments_robot) > 0:
                # Find what the robot is holding
                held_object_name = None
                attachment = self.environment.env.robot.attachments_robot[0]
                for obj_name, obj_id in self.environment.env.objects.items():
                    if obj_id == attachment.child:
                        held_object_name = obj_name
                        break
                
                return {
                    "success": False,
                    "feedback": f"Robot is already holding {held_object_name}. Must place it first before picking {object_name}",
                    "object_name": object_name
                }
            
            # Create pick action using the environment's existing primitive action
            pick_primitive = self.environment.env._primitive_actions["pick"]
            pick_action = Action(
                primitive=pick_primitive,
                obj_args=[object_name],
                param_args={}
            )
            
            success, feedback = self.environment.env.apply_action(pick_action)
            
            # Add delay for better viewing of the motion
            time.sleep(0.8)
            
            result = {
                "success": success,
                "feedback": feedback,
                "object_name": object_name,
                "coordinates": {"x": x, "y": y, "z": z}
            }
            
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {status}: {feedback}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "feedback": f"Exception: {str(e)}",
                "object_name": object_name,
                "coordinates": {"x": x, "y": y, "z": z}
            }
            print(f"   ❌ Error: {str(e)}")
            return error_result
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(
            object_name=kwargs.get("object_name", ""),
            x=kwargs.get("x", 0.0),
            y=kwargs.get("y", 0.0),
            z=kwargs.get("z", 0.0)
        )