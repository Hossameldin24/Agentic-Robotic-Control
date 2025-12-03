"""Detect success tool - validates object placement using vision model."""
import logging
import math
from typing import Dict, Any, Tuple
from .base_tool import BaseTool
import numpy as np
from scipy.spatial.transform import Rotation
from .recognize_objects import RecognizeObjectsTool

logger = logging.getLogger(__name__)


class DetectSuccessTool(BaseTool):
    """
    Tool for validating if an object was placed correctly.
    
    Uses MDETR to detect the object, converts its center to 3D,
    calculates distance loss from target position, and returns
    success/failure based on threshold.
    """
    
    def __init__(self, environment, fov = 60):
        """Initialize detect success tool with environment."""
        self.environment = environment
        self.wrapper = None  # Set externally
        self.name = "detect_success"
        self.description = "Validate if object placement was successful"
        self.recognize_tool = RecognizeObjectsTool(environment=environment) 
        
        self.fov = fov
        self.camera_width = 640
        self.camera_height = 480
        self.camera_near = 0.02
        self.camera_far = 5.0
        # Success threshold in meters
        self.distance_threshold = 0.04  # 4cm tolerance
        
        # TODO: Initialize MDETR model here
        # self.mdetr_model = load_mdetr_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Camera intrinsics - TODO: Get from actual camera setup
        # self.fx = 525.0
        # self.fy = 525.0
        # self.cx = 320.0
        # self.cy = 240.0
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "detect_success",
                "description": "Validate if an object was placed at the target position. Returns success/failure based on distance threshold.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name of the object to validate"
                        },
                        "target_x": {
                            "type": "number",
                            "description": "Target X coordinate in meters"
                        },
                        "target_y": {
                            "type": "number",
                            "description": "Target Y coordinate in meters"
                        },
                        "target_theta": {
                            "type": "number",
                            "description": "Target rotation in radians"
                        },
                        "context": {
                            "type": "object",
                            "description": "Scene context (optional)"
                        }
                    },
                    "required": ["object_name", "target_x", "target_y", "target_theta"]
                }
            }
        }
    
    def _get_camera_image_and_depth(self) -> Tuple[Any, Any]:
        """
        Capture RGB image and depth map from PyBullet camera.
        
        TODO: Implement camera capture from PyBullet environment
        """
        # TODO: Same as in detect_tool.py
        return self.recognize_tool._get_camera_image(fov=self.fov), self.recognize_tool._get_depth_map(fov=self.fov)
    
    def _run_mdetr_detection(self,object_name: str) -> Dict[str, Any]:
        """
        Run MDETR model to detect specific object.
        
        TODO: Implement MDETR inference
        """
        # TODO: Same as in detect_tool.py
        results = self.recognize_tool.execute(object_name=object_name, fov=self.fov)
        if results['detections']:
            detections = [{
                'bbox': det['bbox'],
                'center_2d': ((det['bbox'][0] + det['bbox'][2]) / 2, (det['bbox'][1] + det['bbox'][3]) / 2),
                'score': det['score']
            } for det in results['detections']]
        return detections
        
    
    def _convert_2d_to_3d(self, center_2d: Tuple[float, float], depth_map) -> Tuple[float, float, float]:
        """
        Convert 2D pixel coordinates to 3D world coordinates.
        
        TODO: Implement 2D to 3D projection
        """
        # TODO: Same as in detect_tool.py
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
    
    def _calculate_placement_loss(
        self, 
        actual_pos: Tuple[float, float, float],
        target_pos: Tuple[float, float, float]
    ) -> float:
        """
        Calculate Euclidean distance loss between actual and target positions.
        
        Args:
            actual_pos: (x, y, z) actual position from detection
            target_pos: (x, y, z) target position
            
        Returns:
            Distance in meters
        """
        dx = actual_pos[0] - target_pos[0]
        dy = actual_pos[1] - target_pos[1]
        # Note: We primarily care about x,y placement, z should be on surface
        distance = math.sqrt(dx**2 + dy**2)
        return distance
    
    def execute(
        self, 
        object_name: str, 
        target_x: float, 
        target_y: float, 
        target_theta: float,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Validate object placement success.
        
        Uses MDETR to detect object, converts to 3D, calculates loss,
        returns true/false based on distance threshold.
        """
        print(f"🔍 Validating placement of {object_name} at target ({target_x:.3f}, {target_y:.3f}, θ={target_theta:.2f})...")
        
        try:
            # TODO: Replace this with MDETR-based detection and validation
            # ============================================================
            # MDETR INTEGRATION POINT
            # ============================================================
            #
            # Step 1: Capture camera image and depth
            # image, depth_map = self._get_camera_image_and_depth()
            #
            # Step 2: Run MDETR to detect the placed object
            # detection = self._run_mdetr_detection(image, object_name)
            #
            # if detection is None:
            #     return {
            #         "success": False,
            #         "feedback": f"Could not detect {object_name} - placement may have failed",
            #         "object_name": object_name
            #     }
            #
            # Step 3: Convert detected 2D center to 3D world coordinates
            # actual_x, actual_y, actual_z = self._convert_2d_to_3d(
            #     detection['center_2d'], depth_map
            # )
            #
            # Step 4: Calculate placement loss (distance from target)
            # distance = self._calculate_placement_loss(
            #     (actual_x, actual_y, actual_z),
            #     (target_x, target_y, 0.0)  # z target is surface level
            # )
            #
            # Step 5: Determine success based on threshold
            # success = distance <= self.distance_threshold
            #
            # ============================================================
            
            # PLACEHOLDER: Get actual position from PyBullet environment
            # This will be replaced by MDETR detection
            
            if not hasattr(self.environment, 'env'):
                return {
                    "success": False,
                    "feedback": "Environment not initialized",
                    "object_name": object_name
                }
            
            if object_name not in self.environment.env.objects:
                return {
                    "success": False,
                    "feedback": f"Object '{object_name}' not found in environment",
                    "object_name": object_name
                }
            
            # PLACEHOLDER: Get position from PyBullet (to be replaced by MDETR)
            position = self.environment.env.get_position(object_name)
            actual_x, actual_y, actual_z = position[0], position[1], position[2]
            
            print(f"📍 Actual position: ({actual_x:.3f}, {actual_y:.3f})")
            print(f"🎯 Target position: ({target_x:.3f}, {target_y:.3f})")
            
            # Calculate distance loss
            distance = self._calculate_placement_loss(
                (actual_x, actual_y, actual_z),
                (target_x, target_y, 0.0)
            )
            
            print(f"📏 Distance: {distance:.3f}m (threshold: {self.distance_threshold:.3f}m)")
            
            # Determine success
            success = distance <= self.distance_threshold
            
            if success:
                feedback = f"✅ SUCCESS: {object_name} placed within {distance*100:.1f}cm of target (threshold: {self.distance_threshold*100:.0f}cm)"
                print(f"   {feedback}")
            else:
                feedback = f"❌ FAILED: {object_name} is {distance*100:.1f}cm away from target (threshold: {self.distance_threshold*100:.0f}cm)"
                print(f"   {feedback}")
            
            # Update context with actual position
            updated_context = context.copy() if context else {}
            if 'objects' in updated_context:
                for obj in updated_context['objects']:
                    if obj.get('name') == object_name:
                        obj['center'] = {'x': actual_x, 'y': actual_y, 'z': actual_z}
                        break
            
            print(f"🔄 Updated {object_name} position in context: ({actual_x:.3f}, {actual_y:.3f}, {actual_z:.3f})")
            
            return {
                "success": success,
                "feedback": feedback,
                "object_name": object_name,
                "target_position": {"x": target_x, "y": target_y, "theta": target_theta},
                "actual_position": {"x": actual_x, "y": actual_y, "z": actual_z},
                "distance": distance,
                "threshold": self.distance_threshold,
                "updated_context": updated_context
            }
            
        except Exception as e:
            error_result = {
                "success": False,
                "feedback": f"Validation error: {str(e)}",
                "object_name": object_name
            }
            print(f"   ❌ Error: {str(e)}")
            return error_result
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(
            object_name=kwargs.get("object_name", ""),
            target_x=kwargs.get("target_x", 0.0),
            target_y=kwargs.get("target_y", 0.0),
            target_theta=kwargs.get("target_theta", 0.0),
            context=kwargs.get("context", {})
        )
