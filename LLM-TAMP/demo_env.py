"""
Simple demo script to visualize the PyBullet environment without LLM
Just loads the environment and shows the box-packing task setup
"""
import numpy as np
import time
from envs.pack_compact_env import PackCompactEnv

def demo_environment():
    """Demo the environment with a simple task setup"""
    
    # Create environment
    env = PackCompactEnv()
    
    # Define a simple task configuration
    basket_config = {
        "x": 0.6,
        "y": 0.0,
        "w": 0.3,
        "l": 0.5
    }
    
    # Define some boxes to pack
    boxes_config = {
        0: {
            "name": "red_box",
            "color": [1, 0, 0, 1],  # Red
            "w": 0.08,
            "l": 0.08,
            "h": 0.08,
            "x": 0.4,
            "y": -0.3,
            "z": 0.06
        },
        1: {
            "name": "blue_box",
            "color": [0, 0, 1, 1],  # Blue
            "w": 0.08,
            "l": 0.08,
            "h": 0.08,
            "x": 0.4,
            "y": -0.1,
            "z": 0.06
        },
        2: {
            "name": "green_box",
            "color": [0, 1, 0, 1],  # Green
            "w": 0.08,
            "l": 0.08,
            "h": 0.08,
            "x": 0.4,
            "y": 0.1,
            "z": 0.06
        },
        3: {
            "name": "yellow_box",
            "color": [1, 1, 0, 1],  # Yellow
            "w": 0.08,
            "l": 0.08,
            "h": 0.08,
            "x": 0.4,
            "y": 0.3,
            "z": 0.06
        }
    }
    
    task_config = {
        "basket": basket_config,
        "boxes": boxes_config,
        "use_gui": True
    }
    
    print("=" * 50)
    print("PyBullet Box Packing Environment Demo")
    print("=" * 50)
    print("\nInitializing environment with GUI...")
    print("Task: Pack boxes into the basket")
    print(f"Basket position: x={basket_config['x']}, y={basket_config['y']}")
    print(f"Number of boxes: {len(boxes_config)}")
    print("\nClose the PyBullet window to exit.")
    print("=" * 50)
    
    # Reset environment with GUI
    obs, obs_text = env.reset(**task_config)
    
    # Print observation
    print("\nEnvironment Observation:")
    print(obs_text)
    
    # Check initial goal status
    is_goal, goal_feedback = env.check_goal()
    print(f"\nInitial Goal Status: {'Achieved' if is_goal else 'Not Achieved'}")
    if not is_goal:
        print(f"Goal Feedback: {goal_feedback}")
    
    # Keep the simulation running
    print("\nSimulation is running. The robot and boxes are visible.")
    print("Press Ctrl+C to exit...")
    
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        env.destroy()
        print("Environment closed.")

if __name__ == "__main__":
    demo_environment()
