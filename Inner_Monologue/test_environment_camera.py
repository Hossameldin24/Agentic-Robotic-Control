"""Test the new environment camera view."""
import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from Inner_Monologue.environment.pybullet_env import InnerMonologueEnv
import matplotlib.pyplot as plt
import numpy as np


def test_environment_camera():
    """Test the environment camera view (third-person view)."""
    print("=" * 80)
    print("Testing Environment Camera View")
    print("=" * 80)
    
    # Initialize environment
    print("\n1. Initializing PyBullet environment with GUI...")
    env = InnerMonologueEnv(use_gui=True)
    
    print("\n2. Capturing images from different camera views...")
    
    # Capture from environment camera (default - third-person view)
    print("\n   a) Environment camera (third-person view)...")
    env_camera_data = env.get_camera_image(
        width=1280,
        height=720,
        use_robot_camera=False,  # This is the default
        camera_distance=1.5,
        camera_yaw=50,
        camera_pitch=-35
    )
    
    # Capture from robot camera for comparison
    print("   b) Robot end-effector camera (first-person view)...")
    robot_camera_data = env.get_camera_image(
        width=1280,
        height=720,
        use_robot_camera=True
    )
    
    print("\n3. Camera data:")
    print(f"   Environment camera RGB shape: {env_camera_data['rgb'].shape}")
    print(f"   Environment camera target: {env_camera_data.get('camera_target')}")
    print(f"   Environment camera distance: {env_camera_data.get('camera_distance')}")
    print(f"   Environment camera yaw: {env_camera_data.get('camera_yaw')}°")
    print(f"   Environment camera pitch: {env_camera_data.get('camera_pitch')}°")
    print(f"\n   Robot camera RGB shape: {robot_camera_data['rgb'].shape}")
    print(f"   Robot camera position: {robot_camera_data.get('camera_position')}")
    
    print("\n4. Visualizing both camera views...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Environment camera RGB
    axes[0, 0].imshow(env_camera_data['rgb'])
    axes[0, 0].set_title('Environment Camera - RGB\n(Third-person view like GUI)', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Environment camera depth
    axes[0, 1].imshow(env_camera_data['depth'], cmap='viridis')
    axes[0, 1].set_title('Environment Camera - Depth', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Robot camera RGB
    axes[1, 0].imshow(robot_camera_data['rgb'])
    axes[1, 0].set_title('Robot End-Effector Camera - RGB\n(First-person view from robot)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Robot camera depth
    axes[1, 1].imshow(robot_camera_data['depth'], cmap='viridis')
    axes[1, 1].set_title('Robot End-Effector Camera - Depth', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    print("   ✅ Displaying visualization... (close window to continue)")
    plt.show()
    
    print("\n5. Testing different camera angles for environment view...")
    
    # Different angles to try
    camera_configs = [
        {"name": "Top-down view", "distance": 2.0, "yaw": 0, "pitch": -90},
        {"name": "Side view", "distance": 1.5, "yaw": 90, "pitch": -20},
        {"name": "Front view", "distance": 1.5, "yaw": 0, "pitch": -30},
        {"name": "Angled view (default)", "distance": 1.5, "yaw": 30, "pitch": -35},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, config in enumerate(camera_configs):
        print(f"   {chr(97+i)}) Capturing {config['name']}...")
        image_data = env.get_camera_image(
            width=640,
            height=480,
            use_robot_camera=False,
            camera_distance=config['distance'],
            camera_yaw=config['yaw'],
            camera_pitch=config['pitch']
        )
        
        axes[i].imshow(image_data['rgb'])
        axes[i].set_title(f"{config['name']}\n(dist={config['distance']}, yaw={config['yaw']}°, pitch={config['pitch']}°)", 
                         fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    plt.tight_layout()
    print("\n   ✅ Displaying different camera angles... (close window to continue)")
    plt.show()
    
    print("\n6. Cleaning up...")
    env.destroy()
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("\n💡 Summary:")
    print("   - By default, get_camera_image() now returns environment view (third-person)")
    print("   - Set use_robot_camera=True to get the robot's end-effector camera view")
    print("   - Adjust camera_distance, camera_yaw, and camera_pitch for different angles")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = test_environment_camera()
    sys.exit(0 if success else 1)
