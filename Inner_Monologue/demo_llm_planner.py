"""Simple demo to run the LLM planning agent."""
import logging
import time
from pathlib import Path
import sys

# Add to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from environment.pybullet_env import InnerMonologueEnv
from agents.task_planning_agent import TaskPlanningAgent
from tools import (
    PickTool,
    PlaceTool,
    DetectSuccessTool,
    DescribeSceneTool,
    RecognizeObjectsTool,
    DetectTool
)
from orchestration.tool_wrapper import EnvironmentToolWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_agent_demo():
    """Run the LLM planning agent on the packing task."""
    print("🤖 Inner Monologue: LLM-Based Planning Agent\n")
    
    # Initialize environment
    print("[ENV] Starting PyBullet environment...")
    env = InnerMonologueEnv()
    wrapper = EnvironmentToolWrapper(env)
    
    # Initialize all tools
    tools = {
        "recognize_objects": RecognizeObjectsTool(env),
        "detect": DetectTool(env),
        "pick": PickTool(env),
        "place": PlaceTool(env),
        "detect_success": DetectSuccessTool(env),
        "describe_scene": DescribeSceneTool(env)
    }
    
    # Set up tool dependencies
    tools["detect_success"].wrapper = wrapper
    tools["describe_scene"].set_wrapper(wrapper)
    
    # Initialize planning agent
    print("[AGENT] Initializing LLM planning agent...")
    agent = TaskPlanningAgent(tools)
    
    # Define the task
    task = "Pack all colored boxes (red_box, blue_box, green_box, yellow_box) into the basket as compactly as possible."
    
    print(f"\n📋 TASK: {task}\n")
    print("🚀 Starting agent...\n")
    
    # Start conversation
    action = agent.start_conversation(task)
    current_context = {}
    
    # Agent loop
    max_steps = 50
    step = 0
    
    while step < max_steps:
        step += 1
        print(f"\n{'='*60}")
        print(f"STEP {step}")
        print(f"{'='*60}\n")
        
        # Check if task complete
        if action.get("task_complete"):
            print("✅ Agent reports task complete!")
            print(f"Reasoning: {action.get('reasoning', 'N/A')}")
            break
        
        # Check for errors
        if action.get("error"):
            print(f"❌ Error: {action['error']}")
            print(f"Reasoning: {action.get('reasoning', 'N/A')}")
            break
        
        # Show agent's reasoning
        if action.get("reasoning"):
            print(f"🧠 Agent Reasoning:")
            print(f"   {action['reasoning']}\n")
        
        if action.get("strategy"):
            print(f"💡 Strategy: {action['strategy']}\n")
        
        # Get tool call
        tool_call = action.get("tool_call")
        if not tool_call:
            print("⚠️  No tool call in response. Asking agent to continue...")
            action = agent.provide_feedback("none", {"message": "Please specify which tool to call next"})
            continue
        
        tool_name = tool_call.get("tool")
        params = tool_call.get("params", {})
        
        if not tool_name or tool_name not in tools:
            print(f"⚠️  Unknown tool: {tool_name}")
            action = agent.provide_feedback(tool_name, {
                "success": False,
                "error": f"Unknown tool '{tool_name}'. Available tools: {list(tools.keys())}"
            })
            continue
        
        # Execute tool
        print(f"⚡ Executing: {tool_name}({params})")
        tool = tools[tool_name]
        
        try:
            # Handle different tool signatures
            if tool_name == "detect_success":
                # Need context for detect_success
                if not current_context:
                    current_context = wrapper.get_scene_info()
                params["context"] = current_context
            elif tool_name == "describe_scene":
                # Need context for describe_scene
                if not current_context:
                    current_context = wrapper.get_scene_info()
                params["context"] = current_context
            
            result = tool(**params)
            
            # Update context if tool returned it
            if isinstance(result, dict):
                if "updated_context" in result:
                    current_context = result["updated_context"]
                print(f"   Result: {result.get('feedback', result.get('success', 'Done'))}")
            
            # Provide feedback to agent
            action = agent.provide_feedback(tool_name, result)
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            print(f"   ❌ Error: {e}")
            action = agent.provide_feedback(tool_name, {
                "success": False,
                "error": str(e)
            })
    
    if step >= max_steps:
        print(f"\n⚠️  Reached maximum steps ({max_steps})")
    
    # Final validation
    print(f"\n{'='*60}")
    print("FINAL VALIDATION")
    print(f"{'='*60}\n")
    
    final_success = wrapper.check_success()
    final_scene = wrapper.get_scene_info()
    
    print(f"🎯 Task Success: {final_success}")
    print(f"📊 Final Scene: {len(final_scene.get('objects', []))} objects")
    
    boxes_in_basket = []
    for obj in final_scene.get('objects', []):
        if 'box' in obj.get('name', '').lower():
            center = obj.get('center', {})
            if center.get('x', 0) > 0.6:  # Rough basket check
                boxes_in_basket.append(obj['name'])
                print(f"   ✅ {obj['name']}: ({center['x']:.2f}, {center['y']:.2f}, {center['z']:.2f})")
    
    print(f"\n🏆 RESULT: {len(boxes_in_basket)}/4 boxes in basket")
    
    # Keep environment open
    print(f"\nKeeping environment open for 30 seconds...")
    try:
        for i in range(30, 0, -1):
            print(f"\rTime remaining: {i:2d}s (Ctrl+C to exit)", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nDemo finished!")
    print()


if __name__ == "__main__":
    run_agent_demo()

