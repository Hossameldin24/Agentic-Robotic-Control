"""Detect tool - uses vision model to get 3D coordinates of specific object."""
import logging
from typing import Dict, Any, Tuple
from .base_tool import BaseTool
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from .recognize_objects import RecognizeObjectsTool

logger = logging.getLogger(__name__)


class DetectTool(BaseTool):
    """
    Tool for detecting a specific object and getting its 3D coordinates.
    
    Uses MDETR model to detect the object by name, then converts 
    the 2D bounding box center to 3D world coordinates using depth info.
    """
    
    def __init__(self, environment, fov=60):
        """Initialize detect tool with environment."""
        self.environment = environment
        self.name = "detect"
        self.description = "Detect object and return its 3D center coordinates"

        self.recognize_tool = RecognizeObjectsTool(environment=environment)
        self.fov = fov
        
        # Camera parameters (default values from robot)
        self.camera_width = 640
        self.camera_height = 480
        self.camera_near = 0.02
        self.camera_far = 5.0
        
        # TODO: Initialize MDETR model here
        # self.mdetr_model = load_mdetr_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    def _get_camera_image_and_depth(self) -> Tuple[Any, Any]:
        """
        Capture RGB image and depth map from PyBullet camera.
        
        TODO: Implement camera capture from PyBullet environment
        
        Returns:
            Tuple of (rgb_image, depth_map)
        """
        return self.recognize_tool._get_camera_image(fov=self.fov), self.recognize_tool._get_camera_depth(fov=self.fov)
    
    def _run_mdetr_detection(self, object_name: str) -> Dict[str, Any]:
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

        results = self.recognize_tool.execute(object_name, fov=self.fov)
        detections = [{
            'bbox': det['bbox'],
            'center_2d': ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2),
            'score': det['confidence']
        } for det in results.get('detections', [])]

        return detections
    
    def _convert_2d_to_3d(self, center_2d: Tuple[float, float], depth_map, fov=None) -> Tuple[float, float, float]:
        """
        Convert 2D pixel coordinates to 3D world coordinates using depth.
        
        Args:
            center_2d: (u, v) pixel coordinates
            depth_map: Depth buffer from PyBullet camera (needs conversion to real depth)
            fov: Field of view in degrees (uses self.fov if None)
            
        Returns:
            (x, y, z) 3D coordinates in world frame
        """
        u, v = center_2d
        fov = fov or self.fov
        
        # Get depth buffer value at pixel
        depth_buffer = depth_map[int(v), int(u)]
        
        # Convert depth buffer to real depth using near/far planes
        # Formula from PyBullet: depth = far * near / (far - (far - near) * depth_buffer)
        z_cam = self.camera_far * self.camera_near / (
            self.camera_far - (self.camera_far - self.camera_near) * depth_buffer
        )
        
        # Calculate focal length from FOV
        # focal_length = height / (2 * tan(fov/2))
        fov_rad = np.radians(fov)
        focal_length = self.camera_height / (2.0 * np.tan(fov_rad / 2.0))
        
        # Calculate camera intrinsic matrix parameters
        fx = fy = focal_length
        cx = (self.camera_width - 1) / 2.0
        cy = (self.camera_height - 1) / 2.0
        
        # Back-project to 3D camera frame using pinhole camera model
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        
        # Get camera pose in world frame
        camera_position, camera_orientation = self.environment.env.robot.get_camera_pose()
        
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(camera_orientation)
        camera_rot_matrix = rotation.as_matrix()
        
        # Create homogeneous transformation matrix from camera to world
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = camera_rot_matrix
        camera_to_world[:3, 3] = camera_position
        
        # Transform point from camera frame to world frame
        point_camera = np.array([x_cam, y_cam, z_cam, 1.0])
        point_world = camera_to_world @ point_camera
        
        x_world, y_world, z_world = point_world[:3]
        
        return (float(x_world), float(y_world), float(z_world))
    
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
            
            
            image, depth = self._get_camera_image_and_depth()
            detections = self._run_mdetr_detection(object_name)
            if len(detections) == 0:
                return {
                    "success": False,
                    "feedback": f"Object '{object_name}' not detected in scene",
                    "object_name": object_name
                }
            detection = detections[0]  # Take first detection
            x, y, z = self._convert_2d_to_3d(detection['center_2d'], depth)
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