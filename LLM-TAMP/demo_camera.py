"""
Demo script showing how to use the camera mounted on the robot's end-effector.
This script demonstrates:
1. Getting the camera pose
2. Capturing images from the camera
3. Visualizing camera data
"""

import numpy as np
import matplotlib.pyplot as plt
from envs.pack_compact_env import PackCompactEnv


def main():
    # Create environment
    env = PackCompactEnv()
    
    # Reset environment with GUI
    basket = {
        "x": 0.7,
        "y": 0.0,
        "w": 0.32,
        "l": 0.21
    }
    
    boxes = {
        0: {
            "name": "red_box",
            "color": [1, 0, 0, 1],
            "w": 0.035,
            "l": 0.035,
            "h": 0.07,
            "x": 0.39,
            "y": -0.47,
            "z": 0.06
        },
        1: {
            "name": "blue_box",
            "color": [0, 0, 1, 1],
            "w": 0.07,
            "l": 0.07,
            "h": 0.07,
            "x": 0.39,
            "y": -0.01,
            "z": 0.06
        }
    }
    
    print("Initializing environment...")
    obs, obs_text = env.reset(basket=basket, boxes=boxes, use_gui=True)
    print(f"Environment initialized with {len(env.objects)} objects")
    
    # Access the robot
    robot = env.robot
    
    # Get camera pose at initial position
    print("\n" + "="*60)
    print("CAMERA POSE AT INITIAL POSITION")
    print("="*60)
    camera_position, camera_orientation = robot.get_camera_pose()
    print(f"Camera Position (x, y, z): {camera_position}")
    print(f"Camera Orientation (quaternion): {camera_orientation}")
    
    # Capture image from camera
    print("\n" + "="*60)
    print("CAPTURING IMAGE FROM CAMERA")
    print("="*60)
    image_data = robot.get_camera_image(width=1920, height=1080)
    
    rgb_image = image_data['rgb']
    depth_image = image_data['depth']
    segmentation_image = image_data['segmentation']
    
    print(f"RGB Image shape: {rgb_image.shape}")
    print(f"Depth Image shape: {depth_image.shape}")
    print(f"Segmentation Image shape: {segmentation_image.shape}")
    print(f"Camera Position: {image_data['camera_position']}")
    print(f"Camera Orientation: {image_data['camera_orientation']}")
    
    # Optional: Set a custom camera offset
    # For example, move camera 5cm forward along tool link's local Z-axis
    print("\n" + "="*60)
    print("SETTING CUSTOM CAMERA OFFSET")
    print("="*60)
    robot.set_camera_offset(position=(0.0, 0.0, 0.05), euler=(0, 0, 0))
    camera_position_new, camera_orientation_new = robot.get_camera_pose()
    print(f"New Camera Position: {camera_position_new}")
    print(f"Distance moved: {np.linalg.norm(np.array(camera_position_new) - np.array(camera_position)):.4f}m")
    
    # Visualize the captured images
    print("\n" + "="*60)
    print("VISUALIZING IMAGES")
    print("="*60)
    print("Creating visualization... Close the plot window to continue.")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    # Depth image
    depth_normalized = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    axes[1].imshow(depth_normalized, cmap='gray')
    axes[1].set_title('Depth Image')
    axes[1].axis('off')
    
    # Segmentation image
    axes[2].imshow(segmentation_image, cmap='tab20')
    axes[2].set_title('Segmentation Image')
    axes[2].axis('off')
    
    plt.suptitle(f'Camera View from Position: ({camera_position[0]:.2f}, {camera_position[1]:.2f}, {camera_position[2]:.2f})')
    plt.tight_layout()
    plt.savefig('camera_demo_output.png', dpi=150)
    print("Saved visualization to 'camera_demo_output.png'")
    plt.show()
    
    print("\n" + "="*60)
    print("DEMONSTRATION COMPLETE")
    print("="*60)
    print("Summary:")
    print(f"  - Camera successfully mounted on robot tool link")
    print(f"  - Camera pose retrieved: position {camera_position}")
    print(f"  - Image captured: {rgb_image.shape[1]}x{rgb_image.shape[0]} pixels")
    print(f"  - Image saved to: camera_demo_output.png")
    
    # Clean up
    input("\nPress Enter to close the simulation...")
    env.destroy()
    print("Done!")

    plt.imsave('rgb_image.png', rgb_image)
    plt.imsave('depth_image.png', depth_normalized, cmap='gray')


if __name__ == "__main__":
    main()
