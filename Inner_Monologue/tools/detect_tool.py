"""Detect tool - uses vision model to get 3D coordinates of specific object."""
import logging
from typing import Dict, Any, Tuple

from .detection_tool import DetectionTool
from .base_tool import BaseTool
import numpy as np

from .recognize_objects import RecognizeObjectsTool
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pybullet as p

from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class DetectTool(BaseTool):
    """
    Tool for detecting a specific object and getting its 3D coordinates.
    
    Uses MDETR model to detect the object by name, then converts 
    the 2D bounding box center to 3D world coordinates using depth info.
    """
    
    def __init__(self, environment, fov=140):
        """Initialize detect tool with environment."""
        self.environment = environment
        self.name = "detect"
        self.description = "Detect object and return its 3D center coordinates"

        self.recognize_tool = RecognizeObjectsTool(environment=environment)
        self.fov = fov
        self.detection_tool = DetectionTool(
            environment=environment,
            confidence_threshold=0.6,
            iou_threshold=0.5
        )
        
        # TODO: Initialize MDETR model here
        # self.mdetr_model = load_mdetr_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Fallback to default values if environment doesn't provide them
        self.camera_width = 640
        self.camera_height = 480
        self.camera_near = 0.02
        self.camera_far = 5.0
        print(f"   📷 Using default camera params: {self.camera_width}x{self.camera_height}, near={self.camera_near}, far={self.camera_far}")
    
    def _get_camera_parameters(self) -> Tuple[int, int, float, float]:
        """
        Get current camera parameters, refreshing from environment if possible.
        
        Returns:
            Tuple of (width, height, near, far)
        """
        # Try to get fresh parameters from environment
        if hasattr(self.environment, 'env') and hasattr(self.environment.env, 'robot'):
            robot = self.environment.env.robot
            if hasattr(robot, 'camera_width'):
                return (robot.camera_width, robot.camera_height, robot.camera_near, robot.camera_far)
        
        # Fall back to cached values
        return (self.camera_width, self.camera_height, self.camera_near, self.camera_far)
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "detect",
                "description": "Detect a specific object by name and return its 3D center coordinates (x, y, z) in meters. Use this before pick() or place() operations.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name of the object to detect (e.g., 'red_box', 'blue_box', 'basket')"
                        }
                    },
                    "required": ["object_name"]
                }
            }
        }
    
    def _get_camera_image_and_depth(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Capture RGB image and depth map from PyBullet camera with explicit parameters.
        
        Returns:
            Tuple of (rgb_image, depth_map, camera_params) where camera_params contains:
                - width: actual image width used
                - height: actual image height used
                - fov: actual field of view used
                - near: actual near clipping plane
                - far: actual far clipping plane
        """
        
        # Capture image directly from robot with explicit parameters
        camera_data = self.environment.env.robot.get_camera_image(fov = self.fov)
        return camera_data
    
    def _run_mdetr_detection(self, image, object_name: str) -> Dict[str, Any]:
        """
        Run MDETR model to detect specific object.
        
        TODO: Implement MDETR inference for specific object
        
        Args:
            object_name: Name of object to detect
            
        Returns:
            Detection dict with 'bbox', 'center_2d', 'score' or None if not found
        """
        # TODO: Run MDETR inference
        # prompt = f"a {object_name}"  # or just object_name
        # inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        # 
        # with torch.no_grad():
        #     outputs = self.mdetr_model(**inputs)
        # 
        # # Post-process
        # target_sizes = torch.tensor([image.shape[:2]])
        # results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        # 
        # # Find best match for object_name
        # best_detection = None
        # best_score = 0
        # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        #     if score > best_score and object_name.lower() in str(label).lower():
        #         best_score = score.item()
        #         x1, y1, x2, y2 = box.tolist()
        #         center_x = (x1 + x2) / 2
        #         center_y = (y1 + y2) / 2
        #         best_detection = {
        #             'bbox': [x1, y1, x2, y2],
        #             'center_2d': (center_x, center_y),
        #             'score': best_score
        #         }
        # return best_detection
        try:
            # Step 1: Capture camera image
            logger.info(f"Captured image of size: {image.size}")
            
            # Step 2: Run MDETR detection with broad prompt
            prompt = f"{object_name}"
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
                        'center_2d': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
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
    
    def _plot_detection(self, image, detections_list, object_name):
        """
        Plot the image with detected bounding boxes and center points.
        
        Args:
            image: PIL Image object
            detections_list: List of detection dictionaries with 'bbox', 'center_2d', 'name'
            object_name: Name of the target object being detected
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            ax.set_title(f"Detection Results for '{object_name}'")
            
            # Plot each detection
            for i, detection in enumerate(detections_list):
                bbox = detection['bbox']  # [x1, y1, x2, y2]
                center_2d = detection['center_2d']  # (center_x, center_y)
                name = detection['name']
                confidence = detection.get('confidence', 0.0)
                
                # Create rectangle patch for bounding box
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                
                # Different colors for different detections
                colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
                color = colors[i % len(colors)]
                
                # Draw bounding box
                rect = patches.Rectangle((x1, y1), width, height, 
                                       linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                # Draw center point
                center_x, center_y = center_2d
                ax.plot(center_x, center_y, 'o', color=color, markersize=8, markeredgecolor='white', markeredgewidth=2)
                
                # Add label with object name and confidence
                label_text = f"{name}\n{confidence:.2f}"
                ax.text(x1, y1-5, label_text, color=color, fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                
                print(f"   📍 Plotted detection {i+1}: {name} at center ({center_x:.1f}, {center_y:.1f})")
            
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"   ⚠️  Could not plot detection visualization: {str(e)}")
            logger.warning(f"Plotting failed: {str(e)}")
    
    def _convert_2d_to_3d(self, center_2d: Tuple[float, float], camera_data) -> Tuple[float, float, float]:
        u, v = center_2d
        
        depth_map = camera_data['depth']
        camera_orientation = camera_data['camera_orientation']
        camera_position = camera_data['camera_position']
        
        img_width = camera_data['img_width']
        img_height = camera_data['img_height']
        near = camera_data['near']
        far = camera_data['far']
        
        # Get depth at pixel (u, v)
        depth_buffer = depth_map[int(v), int(u)]
        
        # Convert normalized depth buffer to real depth (linearize)
        real_depth = near * far / (far - (far - near) * depth_buffer)
        
        # Compute intrinsic parameters from FOV
        fov_rad = np.deg2rad(self.fov)
        cx = img_width / 2
        cy = img_height / 2
        fx = (img_width / 2) / np.tan(fov_rad / 2)
        fy = (img_height / 2) / np.tan(fov_rad / 2)
        
        # Flip y-coordinate since pixel origin is top-left (y increases downward)
        # but camera coordinates have y increasing upward
        v_flipped = img_height - v
        
        # Convert pixel coordinates to camera coordinates
        x_cam = (u - cx) * real_depth / fx
        y_cam = (v_flipped - cy) * real_depth / fy
        z_cam = real_depth
        
        # Camera coordinates (OpenGL convention: +Z forward, Y up, X right)
        camera_coords = np.array([x_cam, y_cam, z_cam])
        
        # Transform to world coordinates
        rot = R.from_quat(camera_orientation)
        R_mat = rot.as_matrix()
        
        # Correct transformation: rotate camera coords then translate
        world_coords = R_mat @ camera_coords + camera_position
        
        return world_coords[0], world_coords[1], world_coords[2]
    
    def _add_visual_marker(self, position: Tuple[float, float, float], object_name: str):
        """
        Add a visual marker in the PyBullet environment at the detected 3D position.
        
        Args:
            position: (x, y, z) 3D coordinates in world frame
            object_name: Name of the detected object (for color coding)
        """
        try:
            x, y, z = position
            
            # Color coding based on object name
            color_map = {
                'red': [1, 0, 0],
                'blue': [0, 0, 1],
                'green': [0, 1, 0],
                'yellow': [1, 1, 0],
                'basket': [0.6, 0.4, 0.2],
                'brown': [0.6, 0.4, 0.2],
            }
            
            # Find matching color or use default cyan
            marker_color = [0, 1, 1]  # Default cyan
            for key, color in color_map.items():
                if key in object_name.lower():
                    marker_color = color
                    break
            
            # Create a visual marker sphere at the detected position
            marker_size = 0.02  # 2cm radius sphere
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=marker_size,
                rgbaColor=marker_color + [0.8]  # Add alpha
            )
            
            # Create the marker body
            marker_body_id = p.createMultiBody(
                baseMass=0,  # Static marker
                baseVisualShapeIndex=visual_shape_id,
                basePosition=[x, y, z]
            )
            
            # Also add a vertical line from marker to table for better visualization
            line_start = [x, y, 0.0]  # Table level
            line_end = [x, y, z]
            p.addUserDebugLine(
                lineFromXYZ=line_start,
                lineToXYZ=line_end,
                lineColorRGB=marker_color,
                lineWidth=2,
                lifeTime=30  # Marker lasts 30 seconds
            )
            
            # Add text label above the marker
            text_position = [x, y, z + 0.05]  # 5cm above the marker
            p.addUserDebugText(
                text=f"{object_name}\n({x:.2f}, {y:.2f}, {z:.2f})",
                textPosition=text_position,
                textColorRGB=marker_color,
                textSize=1.0,
                lifeTime=30  # Text lasts 30 seconds
            )
            
            print(f"   📍 Added visual marker at ({x:.3f}, {y:.3f}, {z:.3f}) with color {marker_color}")
            
        except Exception as e:
            print(f"   ⚠️  Could not add visual marker: {str(e)}")
            logger.warning(f"Visual marker creation failed: {str(e)}")
    
    def execute(self, object_name: str) -> Dict[str, Any]:
        """
        Execute detection to get object's 3D center position.
        
        Uses MDETR to detect object, then converts 2D center to 3D coordinates.
        """
        print(f"🎯 Detecting position of {object_name}...")
        
        try:
            # TODO: Replace this with MDETR-based detection
            # ============================================================
            # MDETR INTEGRATION POINT
            # ============================================================
            #
            # Step 1: Capture camera image and depth
            # image, depth_map = self._get_camera_image_and_depth()
            #
            # Step 2: Run MDETR detection for specific object
            # detection = self._run_mdetr_detection(image, object_name)
            # 
            # if detection is None:
            #     return {
            #         "success": False,
            #         "feedback": f"Object '{object_name}' not detected in scene",
            #         "object_name": object_name
            #     }
            #
            # Step 3: Convert 2D center to 3D world coordinates
            # x, y, z = self._convert_2d_to_3d(detection['center_2d'], depth_map)
            #
            # ============================================================
            
            # PLACEHOLDER: Get position from PyBullet environment directly
            # This will be replaced by MDETR + depth projection
            
            # Validate environment
            if not hasattr(self.environment, 'env'):
                return {
                    "success": False,
                    "feedback": "Environment not initialized",
                    "object_name": object_name
                }
            
            
            camera_data = self._get_camera_image_and_depth()
            print(f"   ✅ Captured image and depth for detection.")
            print(f"   📷 Camera params: {camera_data['img_width']}x{camera_data['img_height']}")
            detection_result = self._run_mdetr_detection(object_name = object_name, image=camera_data['rgb'])
            
            # Check if detection was successful
            if not detection_result.get("success", False) or len(detection_result.get("detections", [])) == 0:
                return {
                    "success": False,
                    "feedback": f"Object '{object_name}' not detected in scene",
                    "object_name": object_name
                }
            
            # Get the first detection
            # Find the detection with the highest confidence
            detection = max(detection_result["detections"], key=lambda d: d['confidence'])
            
            # Plot the detection results for visualization
            self._plot_detection(camera_data['rgb'], detection_result["detections"], object_name)
            
            x, y, z = self._convert_2d_to_3d(detection['center_2d'], camera_data)
            print(f"   ✅ Detected '{object_name}' at 3D position ({x:.3f}, {y:.3f}, {z:.3f})")
            
            # Add visual marker in the PyBullet environment
            self._add_visual_marker((x, y, z), object_name)
            
            result = {
                "success": True,
                "object_name": object_name,
                "position": {
                    "x": x,
                    "y": y,
                    "z": z
                },
                "feedback": f"Detected '{object_name}' at position ({x:.3f}, {y:.3f}, {z:.3f})"
            }
            print(f"   ✅ Position: ({x:.3f}, {y:.3f}, {z:.3f})")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "feedback": f"Exception: {str(e)}",
                "object_name": object_name
            }
            print(f"   ❌ Error: {str(e)}")
            return error_result
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(kwargs.get("object_name", ""))