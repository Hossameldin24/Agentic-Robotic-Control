"""Test camera integration with RecognizeObjectsTool."""
import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from Inner_Monologue.environment.pybullet_env import InnerMonologueEnv
from Inner_Monologue.tools.recognize_objects import RecognizeObjectsTool


def test_camera_integration():
    """Test that camera image capture works with DetectionTool."""
    print("=" * 80)
    print("Testing Camera Integration with RecognizeObjectsTool")
    print("=" * 80)
    
    # Initialize environment
    print("\n1. Initializing PyBullet environment...")
    env = InnerMonologueEnv(use_gui=True)
    
    print("\n2. Creating RecognizeObjectsTool...")
    recognize_tool = RecognizeObjectsTool(environment=env)
    
    print("\n3. Testing camera image capture with zoom out...")
    try:
        # Try with wider field of view (zoom out)
        # Default FOV is typically 60-90 degrees
        # Higher values = more zoom out (try 100-120 for wider view)
        fov = 130  # Wider field of view to see more of the scene
        image = recognize_tool._get_camera_image(fov=fov)
        print(f"   ✅ Camera image captured successfully with FOV={fov}°!")
        print(f"   Image size: {image.size}")
        print(f"   Image mode: {image.mode}")
    except Exception as e:
        print(f"   ❌ Failed to capture camera image: {e}")
        return False
    
    print("\n4. Running full object recognition with zoom out...")
    result = recognize_tool.execute(fov=fov)
    
    print("\n5. Results:")
    print(f"   Success: {result['success']}")
    print(f"   Objects found: {result['objects']}")
    print(f"   Count: {result['count']}")
    if 'detections' in result:
        print(f"   Number of detections: {len(result['detections'])}")
        for i, det in enumerate(result['detections'][:]):  # Show all
            print(f"      {i+1}. {det['name']} (confidence: {det['confidence']:.2f})")
    
    print("\n6. Visualizing detections...")
    if 'detections' in result and len(result['detections']) > 0:
        import torch
        
        # Extract bounding boxes, scores, and labels from detections
        bboxes = torch.tensor([det['bbox'] for det in result['detections']])
        scores = torch.tensor([det['confidence'] for det in result['detections']])
        labels = [det['name'] for det in result['detections']]
        
        # Visualize using the detection tool
        recognize_tool.detection_tool.visualize_detections(image, bboxes, scores, labels)
        print("   ✅ Visualization displayed!")
    else:
        print("   ⚠️ No detections to visualize")
    
    print("\n7. Cleaning up...")
    env.destroy()
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_camera_integration()
    sys.exit(0 if success else 1)
