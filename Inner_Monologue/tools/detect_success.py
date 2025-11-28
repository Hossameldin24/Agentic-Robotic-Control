"""Success detection tool for validating object placements."""
import logging
import math
from typing import Dict, Any

logger = logging.getLogger(__name__)


class DetectSuccessTool:
    """Tool to validate if an object was successfully placed at target coordinates."""
    
    def __init__(self, environment):
        """Initialize success detection tool with environment."""
        self.environment = environment
        self.name = "detect_success"
        self.description = "Validate if an object was successfully placed at target coordinates"
        self.threshold = 0.04  # 4 cm threshold as requested
        
        # Import tool wrapper for accurate position detection
        from orchestration.tool_wrapper import EnvironmentToolWrapper
        self.wrapper = EnvironmentToolWrapper(environment)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "detect_success",
                "description": "Validate if an object was successfully placed at the target coordinates within 4cm threshold and return updated context with actual object positions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name of the object that was placed (e.g., 'red_box', 'blue_box')"
                        },
                        "target_x": {
                            "type": "number",
                            "description": "Target X coordinate that was passed to place command"
                        },
                        "target_y": {
                            "type": "number", 
                            "description": "Target Y coordinate that was passed to place command"
                        },
                        "target_theta": {
                            "type": "number",
                            "description": "Target theta orientation that was passed to place command"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current LLM context containing all object coordinates and dimensions from scene_info",
                            "properties": {
                                "objects": {
                                    "type": "array",
                                    "description": "List of all objects with their coordinates and dimensions"
                                },
                                "scene_description": {
                                    "type": "string",
                                    "description": "Description of the current scene"
                                }
                            }
                        }
                    },
                    "required": ["object_name", "target_x", "target_y", "target_theta", "context"]
                }
            }
        }
    
    def execute(self, object_name: str, target_x: float, target_y: float, target_theta: float, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate object placement by comparing actual vs target coordinates and return updated context.
        
        Args:
            object_name: Name of the placed object
            target_x: Target X coordinate from place command
            target_y: Target Y coordinate from place command  
            target_theta: Target theta from place command
            context: Current LLM context with all object positions and dimensions
            
        Returns:
            Dict with success status, detailed feedback, and updated context with actual object positions
        """
        print(f"🔍 Validating placement of {object_name} at target ({target_x:.3f}, {target_y:.3f}, θ={target_theta:.2f})...")
        
        try:
            # Get fresh object positions using the wrapper that works correctly
            scene_info = self.wrapper.get_scene_info()
            
            # Find the object in the scene
            target_object = None
            for obj in scene_info.get('objects', []):
                if obj.get('name') == object_name:
                    target_object = obj
                    break
            
            if target_object is None:
                return {
                    "success": False,
                    "object_name": object_name,
                    "error": f"Object {object_name} not found in scene",
                    "distance": None,
                    "updated_context": context  # Return original context on error
                }
            
            # Get actual object position from scene info
            center = target_object.get('center', {})
            actual_x = center.get('x', 0)
            actual_y = center.get('y', 0)
            actual_z = center.get('z', 0)
            
            # Calculate distance from target
            distance = math.sqrt((actual_x - target_x)**2 + (actual_y - target_y)**2)
            
            print(f"📍 Actual position: ({actual_x:.3f}, {actual_y:.3f})")
            print(f"🎯 Target position: ({target_x:.3f}, {target_y:.3f})")
            print(f"📏 Distance: {distance:.3f}m (threshold: {self.threshold:.3f}m)")
            
            # Check if within threshold
            success = distance <= self.threshold
            
            if success:
                feedback = f"✅ SUCCESS: {object_name} placed within {distance*100:.1f}cm of target (threshold: {self.threshold*100:.0f}cm)"
                print(f"   {feedback}")
            else:
                feedback = f"❌ FAILED: {object_name} is {distance*100:.1f}cm from target (threshold: {self.threshold*100:.0f}cm)"
                print(f"   {feedback}")
            
            # Update context with actual object position
            updated_context = self._update_context_with_actual_position(
                context, object_name, actual_x, actual_y, actual_z
            )
            
            return {
                "success": success,
                "object_name": object_name,
                "target_position": {"x": target_x, "y": target_y, "theta": target_theta},
                "actual_position": {"x": actual_x, "y": actual_y, "z": actual_z},
                "distance": distance,
                "threshold": self.threshold,
                "feedback": feedback,
                "updated_context": updated_context
            }
            
        except Exception as e:
            error_result = {
                "success": False,
                "object_name": object_name,
                "error": f"Exception during validation: {str(e)}",
                "distance": None,
                "updated_context": context  # Return original context on error
            }
            print(f"   ❌ Error: {str(e)}")
            return error_result
    
    def _update_context_with_actual_position(self, context: Dict[str, Any], object_name: str, actual_x: float, actual_y: float, actual_z: float) -> Dict[str, Any]:
        """
        Update the context with the actual position of the placed object.
        
        Args:
            context: Original context with object information
            object_name: Name of the object that was placed
            actual_x, actual_y, actual_z: Actual position from environment
            
        Returns:
            Updated context with corrected object position
        """
        updated_context = {
            "objects": [],
            "scene_description": context.get("scene_description", "Scene with objects")
        }
        
        # Copy all objects and update the placed object's position
        for obj in context.get("objects", []):
            if obj.get("name") == object_name:
                # Update this object with actual position
                updated_obj = obj.copy()
                updated_obj["center"] = {
                    "x": actual_x,
                    "y": actual_y, 
                    "z": actual_z
                }
                
                # Also update bounds if they exist
                if "dimensions" in updated_obj:
                    dims = updated_obj["dimensions"]
                    width = dims.get("width", 0.035)
                    height = dims.get("height", 0.035)
                    depth = dims.get("depth", 0.035)
                    
                    updated_obj["bounds"] = {
                        "min": {
                            "x": actual_x - width/2,
                            "y": actual_y - height/2,
                            "z": actual_z - depth/2
                        },
                        "max": {
                            "x": actual_x + width/2,
                            "y": actual_y + height/2,
                            "z": actual_z + depth/2
                        }
                    }
                
                updated_context["objects"].append(updated_obj)
                print(f"🔄 Updated {object_name} position in context: ({actual_x:.3f}, {actual_y:.3f}, {actual_z:.3f})")
            else:
                # Keep other objects unchanged
                updated_context["objects"].append(obj.copy())
        
        return updated_context
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(
            object_name=kwargs.get("object_name", ""),
            target_x=kwargs.get("target_x", 0.0),
            target_y=kwargs.get("target_y", 0.0),
            target_theta=kwargs.get("target_theta", 0.0),
            context=kwargs.get("context", {"objects": [], "scene_description": "Empty scene"})
        )