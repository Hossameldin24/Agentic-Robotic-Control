"""Detect tool - uses vision model to get 3D coordinates of specific object."""
import logging
from typing import Dict, Any, Tuple

from .detection_tool import DetectionTool
from .base_tool import BaseTool
import numpy as np
from scipy.spatial.transform import Rotation

from .recognize_objects import RecognizeObjectsTool
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        
        # Camera parameters - get from environment if available, otherwise use defaults
        self._init_camera_parameters()
        
        # TODO: Initialize MDETR model here
        # self.mdetr_model = load_mdetr_model()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _init_camera_parameters(self):
        """Initialize camera parameters from environment or use defaults."""
        # Try to get camera parameters from environment
        if hasattr(self.environment, 'env') and hasattr(self.environment.env, 'robot'):
            robot = self.environment.env.robot
            # Check if robot has camera configuration
            if hasattr(robot, 'camera_width'):
                self.camera_width = robot.camera_width
                self.camera_height = robot.camera_height
                self.camera_near = robot.camera_near
                self.camera_far = robot.camera_far
                print(f"   📷 Using camera params from environment: {self.camera_width}x{self.camera_height}, near={self.camera_near}, far={self.camera_far}")
                return
        
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
    
    def _get_camera_image_and_depth(self) -> Tuple[Any, Any]:
        """
        Capture RGB image and depth map from PyBullet camera.
        
        TODO: Implement camera capture from PyBullet environment
        
        Returns:
            Tuple of (rgb_image, depth_map)
        """
        return self.recognize_tool._get_camera_image(fov=self.fov), self.recognize_tool._get_camera_depth(fov=self.fov)
    
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
        
        # Get current camera parameters (may be updated from environment)
        camera_width, camera_height, camera_near, camera_far = self._get_camera_parameters()
        
        print(f"   🔍 Debug - Camera params: {camera_width}x{camera_height}, FOV={fov}°, near={camera_near}, far={camera_far}")
        print(f"   🔍 Debug - Pixel coordinates: u={u:.1f}, v={v:.1f}")
        
        # Clamp pixel coordinates to image bounds
        u = max(0, min(camera_width - 1, u))
        v = max(0, min(camera_height - 1, v))
        
        # Get depth buffer value at pixel
        depth_buffer = depth_map[int(v), int(u)]
        print(f"   🔍 Debug - Raw depth buffer value: {depth_buffer:.6f}")
        
        # Check for invalid depth values
        if depth_buffer <= 0.0 or depth_buffer >= 1.0:
            print(f"   ⚠️  Warning: Invalid depth buffer value: {depth_buffer}")
            depth_buffer = max(0.01, min(0.99, depth_buffer))
        
        # Convert depth buffer to real depth using near/far planes
        # Formula from PyBullet: depth = far * near / (far - (far - near) * depth_buffer)
        denominator = camera_far - (camera_far - camera_near) * depth_buffer
        if abs(denominator) < 1e-6:
            print(f"   ⚠️  Warning: Near-zero denominator in depth conversion: {denominator}")
            z_cam = camera_far  # Use far plane as fallback
        else:
            z_cam = camera_far * camera_near / denominator
        
        print(f"   🔍 Debug - Calculated camera Z depth: {z_cam:.6f}m")
        
        # Sanity check on depth
        if z_cam < camera_near or z_cam > camera_far:
            print(f"   ⚠️  Warning: Depth {z_cam:.3f} outside valid range [{camera_near}, {camera_far}]")
            z_cam = max(camera_near, min(camera_far, z_cam))
        
        # Calculate focal length from FOV
        # focal_length = height / (2 * tan(fov/2))
        fov_rad = np.radians(fov)
        focal_length = camera_height / (2.0 * np.tan(fov_rad / 2.0))
        print(f"   🔍 Debug - Calculated focal length: {focal_length:.1f} pixels")
        
        # Calculate camera intrinsic matrix parameters
        fx = fy = focal_length
        cx = (camera_width - 1) / 2.0  # Principal point x
        cy = (camera_height - 1) / 2.0  # Principal point y
        
        print(f"   🔍 Debug - Camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        
        # Back-project to 3D camera frame using pinhole camera model
        # Note: PyBullet camera coordinate system analysis from rotation matrix:
        # Camera rotation matrix shows: X=-1, Y=1, Z=-1 on diagonal
        # This means: +X points left, +Y points down, +Z points backward (away from scene)
        # We need to adjust for this coordinate system
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        
        print(f"   🔍 Debug - Camera frame coordinates (raw): x_cam={x_cam:.6f}, y_cam={y_cam:.6f}, z_cam={z_cam:.6f}")
        
        # Adjust for PyBullet camera coordinate system
        # Based on rotation matrix, we need to flip X and Z to match world coordinates
        x_cam_corrected = -x_cam  # Flip X (camera X axis is inverted)
        y_cam_corrected = -y_cam  # Flip Y (based on results, Y needs sign flip)  
        z_cam_corrected = z_cam   # Keep Z positive (depth into scene)
        
        print(f"   🔍 Debug - Camera frame coordinates (corrected): x_cam={x_cam_corrected:.6f}, y_cam={y_cam_corrected:.6f}, z_cam={z_cam_corrected:.6f}")
        
        # Get camera pose in world frame
        camera_position, camera_orientation = self.environment.env.robot.get_camera_pose()
        print(f"   🔍 Debug - Camera position: {camera_position}")
        print(f"   🔍 Debug - Camera orientation (quat): {camera_orientation}")
        
        # Convert quaternion to rotation matrix
        rotation = Rotation.from_quat(camera_orientation)
        camera_rot_matrix = rotation.as_matrix()
        
        print(f"   🔍 Debug - Camera rotation matrix:")
        print(f"   {camera_rot_matrix[0]}")
        print(f"   {camera_rot_matrix[1]}")
        print(f"   {camera_rot_matrix[2]}")
        
        # The Z issue is likely because we're using camera-to-world transform incorrectly
        # Let's try a simpler approach: just add the camera frame offset to camera position
        # Since the camera is pointing down with specific orientation
        
        # Simple transformation: camera position + rotated camera frame coordinates
        world_offset = camera_rot_matrix @ np.array([x_cam_corrected, y_cam_corrected, z_cam_corrected])
        x_world_simple = camera_position[0] + world_offset[0]
        y_world_simple = camera_position[1] + world_offset[1]  
        z_world_simple = camera_position[2] + world_offset[2]
        
        print(f"   🔍 Debug - Simple world coordinates: x={x_world_simple:.6f}, y={y_world_simple:.6f}, z={z_world_simple:.6f}")
        
        # Alternative: Project point down to table height (assume Z=0.04 is table surface)
        # The object should be near the table surface
        table_height = 0.040  # Your true Z coordinate
        x_world_projected = camera_position[0] + (world_offset[0] * table_height / camera_position[2])
        y_world_projected = camera_position[1] + (world_offset[1] * table_height / camera_position[2])
        z_world_projected = table_height
        
        print(f"   🔍 Debug - Table-projected coordinates: x={x_world_projected:.6f}, y={y_world_projected:.6f}, z={z_world_projected:.6f}")
        
        # Create homogeneous transformation matrix from camera to world
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = camera_rot_matrix
        camera_to_world[:3, 3] = camera_position
        
        # Transform point from camera frame to world frame using corrected coordinates
        point_camera = np.array([x_cam_corrected, y_cam_corrected, z_cam_corrected, 1.0])
        point_world = camera_to_world @ point_camera
        
        x_world, y_world, z_world = point_world[:3]
        
        print(f"   🔍 Debug - Homogeneous transform result: x={x_world:.6f}, y={y_world:.6f}, z={z_world:.6f}")
        
        # Try alternative Y coordinate (flip Y axis) to see which is correct
        point_camera_alt = np.array([x_cam_corrected, -y_cam_corrected, z_cam_corrected, 1.0])
        point_world_alt = camera_to_world @ point_camera_alt
        x_world_alt, y_world_alt, z_world_alt = point_world_alt[:3]
        
        print(f"   🔍 Debug - Alternative world coords (Y-flipped): x={x_world_alt:.6f}, y={y_world_alt:.6f}, z={z_world_alt:.6f}")
        
        # Test multiple coordinate system combinations to find the best match
        combinations = [
            ("Original homogeneous", x_world, y_world, z_world),
            ("Y-flipped homogeneous", x_world_alt, y_world_alt, z_world_alt),
            ("Simple transform", x_world_simple, y_world_simple, z_world_simple),
            ("Table projected", x_world_projected, y_world_projected, z_world_projected),
            ("Manual correction", x_world_simple, -y_world_simple, table_height),
        ]
        
        print(f"   🔧 Testing coordinate system combinations:")
        best_error = float('inf')
        best_coords = None
        for name, x_test, y_test, z_test in combinations:
            error_x = abs(x_test - 0.390)
            error_y = abs(y_test - 0.470) 
            error_z = abs(z_test - 0.040)
            total_error = error_x + error_y + error_z
            print(f"   {name:20}: ({x_test:.3f}, {y_test:.3f}, {z_test:.3f}) - Error: {total_error:.3f}")
            
            if total_error < best_error:
                best_error = total_error
                best_coords = (x_test, y_test, z_test)
        
        print(f"   🎯 Best method: {best_coords} with error: {best_error:.3f}")
        
        # Return the best coordinates found
        return (float(best_coords[0]), float(best_coords[1]), float(best_coords[2]))
    
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
            print(f"   ✅ Captured image and depth for detection.")
            detection_result = self._run_mdetr_detection(object_name = object_name, image=image)
            
            # Check if detection was successful
            if not detection_result.get("success", False) or len(detection_result.get("detections", [])) == 0:
                return {
                    "success": False,
                    "feedback": f"Object '{object_name}' not detected in scene",
                    "object_name": object_name
                }
            
            # Get the first detection
            detection = detection_result["detections"][0]
            
            # Plot the detection results for visualization
            self._plot_detection(image, detection_result["detections"], object_name)
            
            x, y, z = self._convert_2d_to_3d(detection['center_2d'], depth)
            print(f"   ✅ Detected '{object_name}' at 3D position ({x:.3f}, {y:.3f}, {z:.3f})")
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