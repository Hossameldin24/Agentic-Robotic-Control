"""Recognize objects tool - uses vision model to detect all objects in scene."""
import logging
from typing import Dict, Any, List
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
            confidence_threshold=0.9,
            iou_threshold=0.3
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
            camera_data = self.environment.get_camera_image()
            
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
    
    def _get_camera_depth(self, fov=None):
        """
        Capture Depth image from PyBullet camera.
        
        Args:
            fov: Field of view in degrees. Higher values = more zoom out. 
                 Default is typically 60-90 degrees. Try 100-120 for wider view.
        
        Returns: PIL Image object
        """
        if hasattr(self.environment, 'get_camera_image'):
            # If environment has a method to get camera image
            camera_data = self.environment.get_camera_image(fov=fov)
            
            # Handle dict return (LLM-TAMP format)
            if isinstance(camera_data, dict) and 'depth' in camera_data:
                img_array = camera_data['depth']
            else:
                raise NotImplementedError("Environment's get_camera_image() does not return depth data")
            
            # # Convert numpy array to PIL Image
            # if isinstance(img_array, np.ndarray):
            #     return Image.fromarray(img_array.astype('uint8'), 'RGB')
            return img_array
            
        # elif hasattr(self.environment, 'render'):
        #     # Alternative: use render method
        #     img_array = self.environment.render(mode='rgb_array')
        #     if isinstance(img_array, np.ndarray):
        #         return Image.fromarray(img_array.astype('uint8'), 'RGB')
        #     return img_array
        else:
            raise NotImplementedError(
                "Environment must implement either 'get_camera_image()' or 'render(mode=\"rgb_array\")' method"
            )
    
    def _plot_recognized_objects(self, image, detections_list):
        """
        Plot the image with all detected objects, bounding boxes, and labels.
        
        Args:
            image: PIL Image object
            detections_list: List of detection dictionaries with 'bbox', 'name', 'confidence'
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            ax.imshow(image)
            ax.set_title(f"Recognized Objects ({len(detections_list)} detections)")
            
            # Different colors for different objects
            colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
            
            # Plot each detection
            for i, detection in enumerate(detections_list):
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                name = detection['name']
                confidence = detection.get('confidence', 0.0)
                
                # Create rectangle patch for bounding box
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Assign color
                color = colors[i % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Calculate center point
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Draw center point
                ax.plot(center_x, center_y, 'o', color=color, markersize=6, 
                       markeredgecolor='white', markeredgewidth=1.5)
                
                # Add label with object name and confidence
                label_text = f"{name}\n{confidence:.2f}"
                ax.text(x1, y1-5, label_text, color=color, fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                print(f"   📦 Detection {i+1}: {name} (conf: {confidence:.2f}) at bbox [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
            
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️  Could not plot recognition visualization: {str(e)}")
            logger.warning(f"Plotting failed: {str(e)}")
    
    def execute(self,fov=None) -> Dict[str, Any]:
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
            image = self._get_camera_image()
            logger.info(f"Captured image of size: {image.size}")
            
            # Step 2: Run MDETR detection with broad prompt
            prompt = "yellow box, green box, blue box, red box, brown basket and a robot on a light brown table"
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
            
            # Plot all detected objects
            if len(result["detections"]) > 0:
                self._plot_recognized_objects(image, result["detections"])
            
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