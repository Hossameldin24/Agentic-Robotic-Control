"""Demo of LLM planner with simulated responses to show the architecture."""
import logging
import sys
from pathlib import Path
import time

# Add the current directory to Python path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from environment.pybullet_env import InnerMonologueEnv
from orchestration.tool_wrapper import EnvironmentToolWrapper
from tools.pick_tool import PickTool
from tools.place_tool import PlaceTool
from tools.placement_planner import PlacementPlanner
from tools.detect_success import DetectSuccessTool
from tools.describe_scene import DescribeSceneTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_llm_planner():
    """Demo the LLM planner with simulated intelligent responses."""
    print("🤖 Inner Monologue: LLM Tool-Based Planning DEMO")
    
    # Initialize environment
    print("\n[ENV] Starting PyBullet (GUI=True)...")
    env = InnerMonologueEnv()
    
    # Initialize tools
    wrapper = EnvironmentToolWrapper(env)
    pick_tool = PickTool(env)
    place_tool = PlaceTool(env)
    detect_success_tool = DetectSuccessTool(env)
    describe_scene_tool = DescribeSceneTool(env)
    
    # Set up tool dependencies
    detect_success_tool.wrapper = wrapper
    describe_scene_tool.set_wrapper(wrapper)
    
    print("\n🚀 Tool-Based LLM Planner")
    print("Goal: Pack all boxes into the basket as compactly as possible")
    
    print("\n🤖 LLM Planner starting...")
    
    # Step 1: Get scene information
    print(f"\n--- Step 1 ---")
    print("🧠 LLM: I need to understand the current scene. Let me get scene information.")
    print("⚡ Executing: get_scene_info({})")
    
    scene_info = wrapper.get_scene_info()
    current_context = scene_info  # Track current LLM context
    print(f"📊 Scene: Found {len(current_context['objects'])} objects")
    
    # Show available objects (excluding table and basket)
    boxes = [obj for obj in current_context['objects'] if 'box' in obj.get('name', '').lower()]
    basket = next((obj for obj in current_context['objects'] if 'basket' in obj.get('name', '').lower()), None)
    
    print(f"📦 Boxes to pack: {[box['name'] for box in boxes]}")
    # Initialize placement planner with basket dimensions
    if basket:
        basket_info = {
            'x': basket['center']['x'],
            'y': basket['center']['y'], 
            'w': 0.32,  # LLM-TAMP basket width
            'l': 0.21   # LLM-TAMP basket length
        }
    else:
        basket_info = {'x': 0.7, 'y': 0.0, 'w': 0.32, 'l': 0.21}
    
    planner = PlacementPlanner(basket_info)
    placed_boxes = []  # Track successfully placed boxes
    
    # Step 2: Pick first box
    if boxes:
        print(f"\n--- Step 2 ---")
        first_box = boxes[0]['name']
        print(f"🧠 LLM: I'll start by picking the {first_box} and placing it in the basket.")
        print(f"⚡ Executing: pick({{\"object_name\": \"{first_box}\"}})")
        
        pick_result = pick_tool.execute(first_box)
        
        # Step 3: Intelligent placement with replanning
        if pick_result.get('success', False):
            print(f"\n--- Step 3 ---")
            print(f"🧠 LLM: Now I'll use spatial reasoning to find optimal placement for {first_box}.")
            
            # Get box dimensions for spatial reasoning
            first_box_info = next((box for box in boxes if box['name'] == first_box), {})
            
            # Try placement with up to 3 attempts using different strategies
            placement_successful = False
            for attempt in range(3):
                if attempt > 0:
                    print(f"\n🔄 Replanning attempt {attempt + 1} for {first_box}...")
                    
                # Calculate optimal placement using spatial reasoning
                place_x, place_y = planner.calculate_optimal_placement(
                    first_box_info, placed_boxes, attempt
                )
                
                placement_info = planner.get_placement_info(place_x, place_y, first_box_info)
                print(f"🎯 Calculated placement: ({place_x:.3f}, {place_y:.3f})")
                print(f"📏 Margins - Left:{placement_info['margins']['left']:.3f}, Right:{placement_info['margins']['right']:.3f}")
                
                print(f"⚡ Executing: place({{\"x\": {place_x:.2f}, \"y\": {place_y:.2f}, \"theta\": 0.0}})")
                place_result = place_tool.execute(place_x, place_y, 0.0)
                
                # Validate placement with detect_success tool
                print(f"⚡ Executing: detect_success({{\"object_name\": \"{first_box}\", \"target_x\": {place_x:.2f}, \"target_y\": {place_y:.2f}, \"target_theta\": 0.0, \"context\": {{scene_info}}}})")
                success_result = detect_success_tool.execute(first_box, place_x, place_y, 0.0, current_context)
                
                if success_result.get('success', False):
                    print(f"   ✅ Placement Success: {success_result.get('feedback', 'Success')}")
                    # Update current context with actual object position
                    current_context = success_result.get('updated_context', current_context)
                    placement_successful = True
                    
                    # Get strategic scene analysis after successful placement
                    print(f"⚡ Executing: describe_scene({{\"task_context\": \"packing boxes into basket\", \"last_action\": \"placed {first_box} at ({place_x:.2f}, {place_y:.2f})\", \"context\": {{updated_scene_info}}}})")
                    scene_analysis = describe_scene_tool.execute(
                        task_context="packing boxes into basket",
                        last_action=f"placed {first_box} at ({place_x:.2f}, {place_y:.2f})",
                        context=current_context
                    )
                    
                    # Add to placed boxes for future collision avoidance using actual position
                    actual_pos = success_result.get('actual_position', {'x': place_x, 'y': place_y})
                    placed_boxes.append({
                        'name': first_box,
                        'center': {'x': actual_pos['x'], 'y': actual_pos['y']},
                        'w': first_box_info.get('w', 0.035),
                        'l': first_box_info.get('l', 0.035)
                    })
                    break
                else:
                    print(f"   ❌ Placement Failed: {success_result.get('feedback', 'Unknown error')}")
                    
                    # Get scene analysis for failed placement to understand why
                    print(f"⚡ Executing: describe_scene({{\"task_context\": \"packing boxes into basket\", \"last_action\": \"failed to place {first_box} at ({place_x:.2f}, {place_y:.2f})\", \"context\": {{scene_info}}}})")
                    scene_analysis = describe_scene_tool.execute(
                        task_context="packing boxes into basket", 
                        last_action=f"failed to place {first_box} at ({place_x:.2f}, {place_y:.2f})",
                        context=current_context
                    )
                    
                    # Use scene analysis insights for replanning
                    insights = scene_analysis.get('insights', {})
                    if insights.get('recommended_action') == 'replan':
                        print(f"🧠 LLM: Scene analysis recommends replanning. Trying different approach...")
                    else:
                        print(f"🧠 LLM: Scene analysis suggests continuing. Trying alternative coordinates...")
            
            if not placement_successful:
                print(f"❌ Could not place {first_box} after 3 attempts")
                
            place_result = {'success': placement_successful}
            
            # Validate placement with success detection and scene analysis
            if place_result.get('success', False):
                print(f"   ✅ Success: {place_result.get('message', 'Success')}")
                
                print(f"\n🔍 Validating placement...")
                print("⚡ Executing: check_success({})")
                success_result = wrapper.check_success()
                print(f"🎯 Success Check: {success_result}")
                
                print("⚡ Executing: get_scene_info({}) for validation")
                validation_scene = wrapper.get_scene_info()
                print(f"📊 Updated Scene: Found {len(validation_scene['objects'])} objects")
                
                # Check if the box is now in the basket
                for obj in validation_scene['objects']:
                    if obj['name'] == first_box:
                        print(f"📍 {first_box} position: ({obj['center']['x']:.2f}, {obj['center']['y']:.2f}, {obj['center']['z']:.2f})")
                        break
            else:
                print(f"   ❌ Failed: {place_result.get('message', 'Unknown error')}")
    
    # Continue with remaining boxes
    remaining_boxes = boxes[1:] if len(boxes) > 1 else []
    step_counter = 4
    
    for i, box in enumerate(remaining_boxes):
        if step_counter > 15:  # Safety limit
            break
            
        box_name = box['name']
        print(f"\n--- Step {step_counter} ---")
        print(f"🧠 LLM: Now I'll pick the {box_name} and use spatial reasoning for optimal placement.")
        print(f"⚡ Executing: pick({{\"object_name\": \"{box_name}\"}})")
        
        pick_result = pick_tool.execute(box_name)
        step_counter += 1
        
        if pick_result.get('success', False):
            print(f"   ✅ Success: {pick_result.get('message', 'Success')}")
            
            print(f"\n--- Step {step_counter} ---")
            print(f"🧠 LLM: I'll calculate optimal placement considering basket dimensions and already-placed boxes.")
            print(f"💡 Using updated context with {len(current_context['objects'])} objects and their actual positions.")
            
            # Try intelligent placement with replanning
            placement_successful = False
            for attempt in range(3):  # Up to 3 attempts
                if attempt > 0:
                    print(f"\n🔄 Replanning attempt {attempt + 1} for {box_name}...")
                
                # Calculate optimal placement using spatial reasoning
                place_x, place_y = planner.calculate_optimal_placement(
                    box, placed_boxes, attempt
                )
                
                placement_info = planner.get_placement_info(place_x, place_y, box)
                print(f"🎯 Calculated placement: ({place_x:.3f}, {place_y:.3f})")
                print(f"📏 Within basket: {placement_info['within_basket']}")
                print(f"📏 Margins - Left:{placement_info['margins']['left']:.3f}, Right:{placement_info['margins']['right']:.3f}")
                
                print(f"⚡ Executing: place({{\"x\": {place_x:.2f}, \"y\": {place_y:.2f}, \"theta\": 0.0}})")
                place_result = place_tool.execute(place_x, place_y, 0.0)
                
                # Validate placement with detect_success tool  
                print(f"⚡ Executing: detect_success({{\"object_name\": \"{box_name}\", \"target_x\": {place_x:.2f}, \"target_y\": {place_y:.2f}, \"target_theta\": 0.0, \"context\": {{scene_info}}}})")
                success_result = detect_success_tool.execute(box_name, place_x, place_y, 0.0, current_context)
                
                if success_result.get('success', False):
                    print(f"   ✅ Placement Success: {success_result.get('feedback', 'Success')}")
                    # Update current context with actual object position
                    current_context = success_result.get('updated_context', current_context)
                    placement_successful = True
                    
                    # Get strategic scene analysis to guide next steps
                    print(f"⚡ Executing: describe_scene({{\"task_context\": \"packing remaining boxes\", \"last_action\": \"placed {box_name} at ({place_x:.2f}, {place_y:.2f})\", \"context\": {{updated_scene_info}}}})")
                    scene_analysis = describe_scene_tool.execute(
                        task_context=f"packing remaining boxes - {len(remaining_boxes) - i} left",
                        last_action=f"placed {box_name} at ({place_x:.2f}, {place_y:.2f})",
                        context=current_context
                    )
                    
                    # Add to placed boxes for future collision avoidance using actual position
                    actual_pos = success_result.get('actual_position', {'x': place_x, 'y': place_y})
                    placed_boxes.append({
                        'name': box_name,
                        'center': {'x': actual_pos['x'], 'y': actual_pos['y']},
                        'w': box.get('w', 0.035),
                        'l': box.get('l', 0.035)
                    })
                    break
                else:
                    print(f"   ❌ Placement Failed: {success_result.get('feedback', 'Unknown error')}")
                    
                    # Analyze why placement failed
                    print(f"⚡ Executing: describe_scene({{\"task_context\": \"troubleshooting placement failure\", \"last_action\": \"failed to place {box_name}\", \"context\": {{scene_info}}}})")
                    scene_analysis = describe_scene_tool.execute(
                        task_context="troubleshooting placement failure",
                        last_action=f"failed to place {box_name} at ({place_x:.2f}, {place_y:.2f})",
                        context=current_context
                    )
                    
                    insights = scene_analysis.get('insights', {})
                    if insights.get('collision_risk'):
                        print(f"🧠 LLM: Scene analysis detected collision risk. Adjusting strategy...")
                    else:
                        print(f"🧠 LLM: Placement failed for other reasons. Trying different coordinates...")
            
            if not placement_successful:
                print(f"❌ Could not place {box_name} after 3 attempts")
                
            step_counter += 1
            
            # Validate each placement
            if place_result.get('success', False):
                print(f"   ✅ Success: {place_result.get('message', 'Success')}")
                
                print(f"\n🔍 Validating {box_name} placement...")
                print("⚡ Executing: check_success({})")
                success_result = wrapper.check_success()
                print(f"🎯 Success Check: {success_result}")
                
                print("⚡ Executing: get_scene_info({}) for validation")
                validation_scene = wrapper.get_scene_info()
                print(f"📊 Updated Scene: Found {len(validation_scene['objects'])} objects")
                
                # Check if the box is now in the basket
                for obj in validation_scene['objects']:
                    if obj['name'] == box_name:
                        print(f"📍 {box_name} position: ({obj['center']['x']:.2f}, {obj['center']['y']:.2f}, {obj['center']['z']:.2f})")
                        break
                        
                # Check overall packing progress
                boxes_in_basket = []
                for obj in validation_scene['objects']:
                    if 'box' in obj['name'].lower() and obj['center']['x'] > 0.6:  # Rough basket check
                        boxes_in_basket.append(obj['name'])
                print(f"📦 Boxes in basket area: {boxes_in_basket}")
                
            else:
                print(f"   ❌ Failed: {place_result.get('message', 'Unknown error')}")
                print(f"⚠️  Continuing with next box...")
        else:
            print(f"   ❌ Failed: {pick_result.get('message', 'Unknown error')}")
            print(f"⚠️  Continuing with next box...")
    
    # Final comprehensive success check
    print(f"\n--- Final Step ---")
    print("🧠 LLM: Let me perform a final comprehensive success check.")
    print("⚡ Executing: check_success({})")
    
    final_success_result = wrapper.check_success()
    print(f"🎯 Final Success Check: {final_success_result}")
    
    print("⚡ Executing: get_scene_info({}) for final validation")
    final_scene = wrapper.get_scene_info()
    print(f"📊 Final Scene: Found {len(final_scene['objects'])} objects")
    
    # Count boxes in basket for final report
    final_boxes_in_basket = []
    for obj in final_scene['objects']:
        if 'box' in obj['name'].lower():
            if obj['center']['x'] > 0.6:  # Rough basket check
                final_boxes_in_basket.append(obj['name'])
                print(f"📍 {obj['name']} final position: ({obj['center']['x']:.2f}, {obj['center']['y']:.2f}, {obj['center']['z']:.2f})")
    
    print(f"\n🏆 FINAL RESULT: {len(final_boxes_in_basket)}/{len(boxes)} boxes successfully placed in basket!")
    print(f"✅ Boxes in basket: {final_boxes_in_basket}")
    
    # Final comprehensive scene analysis
    print(f"\n--- Final Scene Analysis ---")
    print(f"⚡ Executing: describe_scene({{\"task_context\": \"final task completion assessment\", \"last_action\": \"completed packing task\", \"context\": {{final_scene_info}}}})")
    final_analysis = describe_scene_tool.execute(
        task_context="final task completion assessment",
        last_action="completed packing task",
        context=final_scene
    )
    
    print(f"\n🏁 Planning Demo Complete!")
    print("✅ Architecture demonstrated:")
    print("  • LLM analyzes scene with get_scene_info()")
    print("  • LLM picks objects with pick(object_name)")
    print("  • LLM places objects with place(x, y, theta)")
    print("  • LLM validates each placement with detect_success()")
    print("  • LLM gets strategic insights with describe_scene()")
    print("  • LLM re-analyzes scene after each action")
    print("  • Step-by-step feedback enables adaptive planning")
    
    print(f"\n💡 The GUI shows the robot and environment in real-time!")
    print("You can see the robot arm, table, basket, and colored boxes.")
    
    print(f"\nKeeping environment open for 30 seconds...")
    try:
        for i in range(30, 0, -1):
            print(f"\rTime remaining: {i:2d}s (Ctrl+C to exit)", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nDemo finished!")


if __name__ == "__main__":
    demo_llm_planner()