"""Environment-aware tool wrappers for orchestration."""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class EnvironmentToolWrapper:
    """Wrapper to make tools work with environment."""
    
    def __init__(self, environment):
        """Initialize with environment."""
        self.environment = environment
    
    def get_scene_info(self) -> Dict[str, Any]:
        """Get comprehensive scene information."""
        try:
            # Use the environment's recognize_objects_dummy method if available
            if hasattr(self.environment, 'recognize_objects_dummy'):
                objects = self.environment.recognize_objects_dummy()
                description = self.environment.get_scene_description() if hasattr(self.environment, 'get_scene_description') else "Scene with objects"
            else:
                # Fallback to basic object info from the LLM-TAMP environment
                objects = self._get_basic_object_info()
                description = "Scene containing boxes and basket"
            
            return {
                "objects": objects,
                "scene_description": description
            }
        except Exception as e:
            logger.error(f"Error getting scene info: {e}")
            return {
                "objects": [],
                "scene_description": "Error retrieving scene information"
            }
    
    def _get_basic_object_info(self) -> list:
        """Get basic object information from LLM-TAMP environment."""
        objects = []
        
        try:
            if hasattr(self.environment, 'env') and hasattr(self.environment.env, 'objects'):
                import pybullet as p
                
                for obj_name, obj_id in self.environment.env.objects.items():
                    try:
                        # Get position and orientation
                        pos, orn = p.getBasePositionAndOrientation(obj_id)
                        
                        # Get bounding box
                        aabb = p.getAABB(obj_id)
                        min_pos, max_pos = aabb
                        
                        # Calculate dimensions
                        width = max_pos[0] - min_pos[0]
                        height = max_pos[1] - min_pos[1]
                        depth = max_pos[2] - min_pos[2]
                        
                        obj_info = {
                            "name": obj_name,
                            "center": {"x": pos[0], "y": pos[1], "z": pos[2]},
                            "dimensions": {"width": width, "height": height, "depth": depth},
                            "bounds": {
                                "min": {"x": min_pos[0], "y": min_pos[1], "z": min_pos[2]},
                                "max": {"x": max_pos[0], "y": max_pos[1], "z": max_pos[2]}
                            }
                        }
                        
                        objects.append(obj_info)
                    except Exception as e:
                        logger.warning(f"Could not get info for object {obj_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error accessing environment objects: {e}")
        
        return objects
    
    def check_success(self) -> Dict[str, Any]:
        """Check if goal has been achieved using LLM-TAMP environment."""
        try:
            # Use the LLM-TAMP environment's check_goal method
            if hasattr(self.environment, 'check_goal'):
                is_goal, feedback = self.environment.check_goal()
                return {
                    "success": is_goal,
                    "explanation": feedback if feedback else ("Goal achieved!" if is_goal else "Goal not yet achieved")
                }
            # Try the inner environment if it's wrapped
            elif hasattr(self.environment, 'env') and hasattr(self.environment.env, 'check_goal'):
                is_goal, feedback = self.environment.env.check_goal()
                return {
                    "success": is_goal,
                    "explanation": feedback if feedback else ("Goal achieved!" if is_goal else "Goal not yet achieved")
                }
            else:
                # Basic fallback - check if boxes are in basket area
                objects = self._get_basic_object_info()
                boxes_in_basket = []
                for obj in objects:
                    if 'box' in obj['name'].lower() and obj['center']['x'] > 0.6:  # Rough basket area
                        boxes_in_basket.append(obj['name'])
                
                total_boxes = len([obj for obj in objects if 'box' in obj['name'].lower()])
                success = len(boxes_in_basket) == total_boxes and total_boxes > 0
                
                return {
                    "success": success,
                    "explanation": f"{len(boxes_in_basket)}/{total_boxes} boxes in basket: {boxes_in_basket}"
                }
        except Exception as e:
            logger.error(f"Error checking success: {e}")
            return {
                "success": False,
                "explanation": f"Error checking success: {str(e)}"
            }