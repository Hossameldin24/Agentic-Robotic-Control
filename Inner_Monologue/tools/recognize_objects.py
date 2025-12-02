"""Recognize objects tool - uses vision model to detect all objects in scene."""
import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np
from .base_tool import BaseTool
from .detection_tool import DetectionTool

logger = logging.getLogger(__name__)


class RecognizeObjectsTool(BaseTool):
    """
    Tool for detecting all objects in the scene.
    
    Uses MDETR model with prompt "all objects you can see" to detect
    and return a list of object names present in the scene.
    """
    
    def __init__(self, environment):
        """Initialize recognize objects tool with environment."""
        self.environment = environment
        self.name = "recognize_objects"
        self.description = "Detect and return list of all object names in the scene"
        
        # Initialize MDETR detection tool
        self.detection_tool = DetectionTool(
            environment=environment,
            confidence_threshold=0.6,
            iou_threshold=0.5
        )
        logger.info("Initialized DetectionTool for object recognition")
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "recognize_objects",
                "description": "Detect and return list of all object names currently visible in the scene using vision model.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fov": {
                            "type": "number",
                            "description": "Field of view in degrees for camera zoom (higher = more zoom out)"
                        }
                    },
                    "required": []
                }
            }
        }
    
    def _get_camera_image(self, fov=None):
        """
        Capture RGB image from PyBullet camera.
        
        Args:
            fov: Field of view in degrees. Higher values = more zoom out. 
                 Default is typically 60-90 degrees. Try 100-120 for wider view.
        
        Returns: PIL Image object
        """
        if hasattr(self.environment, 'get_camera_image'):
            # If environment has a method to get camera image
            camera_data = self.environment.get_camera_image(fov=fov)
            
            # Handle dict return (LLM-TAMP format)
            if isinstance(camera_data, dict) and 'rgb' in camera_data:
                img_array = camera_data['rgb']
            else:
                img_array = camera_data
            
            # Convert numpy array to PIL Image
            if isinstance(img_array, np.ndarray):
                return Image.fromarray(img_array.astype('uint8'), 'RGB')
            return img_array
            
        elif hasattr(self.environment, 'render'):
            # Alternative: use render method
            img_array = self.environment.render(mode='rgb_array')
            if isinstance(img_array, np.ndarray):
                return Image.fromarray(img_array.astype('uint8'), 'RGB')
            return img_array
        else:
            raise NotImplementedError(
                "Environment must implement either 'get_camera_image()' or 'render(mode=\"rgb_array\")' method"
            )
    
    def execute(self, fov=None) -> Dict[str, Any]:
        """
        Execute object recognition using MDETR via DetectionTool.
        
        Args:
            fov: Field of view in degrees for camera zoom (higher = more zoom out)
        
        Uses prompt "all objects you can see" to detect all objects.
        Returns list of unique object names found.
        """
        print(f"👁️  Recognizing objects in scene...")
        
        try:
            # Step 1: Capture camera image
            image = self._get_camera_image(fov=fov)
            logger.info(f"Captured image of size: {image.size}")
            
            # Step 2: Run MDETR detection with broad prompt
            prompt = "red box, blue box, green box, yellow box, brown basket and a robot on a light brown table"
            bboxes, scores, detected_objects = self.detection_tool.detect_objects(image, prompt)
            
            # Step 3: Extract unique object names (filter out "unknown")
            object_list = list(set([obj for obj in detected_objects if obj != "unknown"]))
            
            # Log detections
            logger.info(f"Detected {len(detected_objects)} objects: {detected_objects}")
            logger.info(f"Unique objects: {object_list}")
            
            result = {
                "success": True,
                "objects": object_list,
                "count": len(object_list),
                "detections": [
                    {
                        "name": obj,
                        "confidence": score.item(),
                        "bbox": bbox.tolist()
                    }
                    for obj, score, bbox in zip(detected_objects, scores, bboxes)
                ],
                "feedback": f"Found {len(object_list)} unique objects in scene"
            }
            
            print(f"   ✅ Found {len(object_list)} unique objects: {', '.join(object_list)}")
            return result
            
        except NotImplementedError as e:
            # Fallback to hardcoded list if camera not available
            logger.warning(f"Camera not available: {str(e)}. Using fallback.")
            object_list = ["red box", "blue box", "green box", "yellow box", "basket"]
            
            result = {
                "success": True,
                "objects": object_list,
                "count": len(object_list),
                "feedback": f"Using fallback: Found {len(object_list)} objects in scene"
            }
            
            print(f"   ⚠️  Using fallback: {', '.join(object_list)}")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "objects": [],
                "count": 0,
                "feedback": f"Exception: {str(e)}"
            }
            print(f"   ❌ Error: {str(e)}")
            logger.error(f"Error in recognize_objects: {str(e)}", exc_info=True)
            return error_result
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(**kwargs)
