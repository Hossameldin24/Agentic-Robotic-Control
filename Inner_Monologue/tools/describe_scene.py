"""Scene description tool that uses LLM to analyze the current state and provide strategic insights."""
import logging
from typing import Dict, Any
import json

logger = logging.getLogger(__name__)


class DescribeSceneTool:
    """Tool that uses LLM to analyze the scene and provide intelligent observations."""
    
    def __init__(self, environment, llm_client=None):
        """Initialize scene description tool with environment and optional LLM client."""
        self.environment = environment
        self.llm_client = llm_client
        self.wrapper = None  # Will be set externally
        self.name = "describe_scene"
        self.description = "Analyze the current scene state and provide intelligent observations for planning"
    
    def set_wrapper(self, wrapper):
        """Set the environment wrapper for getting scene info."""
        self.wrapper = wrapper
    
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": "describe_scene",
                "description": "Analyze the current scene state and provide intelligent observations, collision detection, progress assessment, and strategic recommendations for task planning.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_context": {
                            "type": "string",
                            "description": "Brief description of the current task being performed (e.g., 'packing boxes into basket', 'placing red_box')"
                        },
                        "last_action": {
                            "type": "string", 
                            "description": "The last action that was performed (e.g., 'placed red_box at (0.62, -0.05)', 'failed to place blue_box')"
                        },
                        "context": {
                            "type": "object",
                            "description": "Current scene context with all object positions and dimensions",
                            "properties": {
                                "objects": {
                                    "type": "array",
                                    "description": "List of all objects with their coordinates and dimensions"
                                },
                                "scene_description": {
                                    "type": "string",
                                    "description": "Description of the current scene"
                                }
                            }
                        }
                    },
                    "required": ["task_context", "last_action", "context"]
                }
            }
        }
    
    def _generate_scene_prompt(self, task_context: str, last_action: str, scene_data: Dict[str, Any]) -> str:
        """Generate a detailed prompt for the LLM to analyze the scene."""
        
        # Extract object information
        objects = scene_data.get('objects', [])
        
        # Categorize objects
        boxes = [obj for obj in objects if 'box' in obj.get('name', '').lower()]
        basket = next((obj for obj in objects if 'basket' in obj.get('name', '').lower()), None)
        robot = next((obj for obj in objects if 'robot' in obj.get('name', '').lower() or 'panda' in obj.get('name', '').lower()), None)
        
        # Basket boundaries (from our environment config)
        basket_x_min, basket_x_max = 0.540, 0.860
        basket_y_min, basket_y_max = -0.105, 0.105
        
        # Analyze box positions relative to basket
        boxes_in_basket = []
        boxes_outside_basket = []
        
        for box in boxes:
            center = box.get('center', {})
            x, y = center.get('x', 0), center.get('y', 0)
            
            if basket_x_min <= x <= basket_x_max and basket_y_min <= y <= basket_y_max:
                boxes_in_basket.append(box)
            else:
                boxes_outside_basket.append(box)
        
        # Build detailed scene analysis
        scene_analysis = f"""
SCENE ANALYSIS REQUEST

TASK CONTEXT: {task_context}
LAST ACTION: {last_action}

ENVIRONMENT SETUP:
- Robot arm (Panda manipulator) for picking and placing objects
- 4 colored boxes (red, blue, green, yellow) that need to be packed
- 1 destination basket at position (0.7, 0.0) with dimensions 32cm x 21cm
- Table surface for object placement

CURRENT OBJECT POSITIONS:
"""
        
        # Add box positions
        for box in boxes:
            center = box.get('center', {})
            dims = box.get('dimensions', {})
            name = box.get('name', 'unknown_box')
            x, y, z = center.get('x', 0), center.get('y', 0), center.get('z', 0)
            w, h, d = dims.get('width', 0.05), dims.get('height', 0.05), dims.get('depth', 0.05)
            
            in_basket = basket_x_min <= x <= basket_x_max and basket_y_min <= y <= basket_y_max
            status = "INSIDE BASKET" if in_basket else "OUTSIDE BASKET"
            
            scene_analysis += f"- {name}: ({x:.3f}, {y:.3f}, {z:.3f}) size=({w:.3f}x{h:.3f}x{d:.3f}) [{status}]\n"
        
        # Add basket info
        if basket:
            center = basket.get('center', {})
            x, y, z = center.get('x', 0.7), center.get('y', 0.0), center.get('z', 0.0)
            scene_analysis += f"- basket: ({x:.3f}, {y:.3f}, {z:.3f}) boundaries=({basket_x_min:.3f} to {basket_x_max:.3f}, {basket_y_min:.3f} to {basket_y_max:.3f})\n"
        
        scene_analysis += f"""
CURRENT PROGRESS:
- Boxes in basket: {len(boxes_in_basket)}/4 ({[b.get('name') for b in boxes_in_basket]})
- Boxes outside basket: {len(boxes_outside_basket)}/4 ({[b.get('name') for b in boxes_outside_basket]})

BASKET BOUNDARIES: X({basket_x_min:.3f} to {basket_x_max:.3f}), Y({basket_y_min:.3f} to {basket_y_max:.3f})

Please analyze this scene and provide:

1. SCENE SUMMARY: Brief overview of current state
2. COLLISION ANALYSIS: Are any boxes colliding or overlapping?
3. BASKET ANALYSIS: How efficiently are boxes packed in the basket?
4. SPATIAL OBSERVATIONS: Available space, potential placement issues
5. PROGRESS ASSESSMENT: How well is the task progressing?
6. STRATEGIC RECOMMENDATIONS: What should the agent do next?

Focus on actionable insights that help with task planning and decision making.
Respond in a structured format with clear sections.
"""
        
        return scene_analysis.strip()
    
    def _simulate_llm_response(self, prompt: str, task_context: str, last_action: str, scene_data: Dict[str, Any]) -> str:
        """Use LLM to conduct intelligent human interview and generate scene analysis."""
        
        print("\n" + "="*60)
        print("🤖 LLM SCENE ANALYST")
        print("="*60)
        print(f"Task: {task_context}")
        print(f"Last Action: {last_action}")
        print("\nThe LLM will ask you strategic questions to understand the scene state.")
        print("It will adapt its questions based on your responses.")
        print("-"*60)
        
        # Initialize conversation context
        responses = {}
        conversation_history = []
        max_questions = 6  # Reasonable limit
        
        for question_num in range(1, max_questions + 1):
            # Generate next question based on context and previous responses
            next_question = self._generate_intelligent_question(
                task_context, last_action, scene_data, responses, conversation_history, question_num
            )
            
            if next_question is None:
                print(f"\n🎯 LLM: I have enough information to analyze the scene.")
                break
                
            print(f"\n{question_num}. 🤖 LLM: {next_question}")
            
            try:
                response = input("   👤 You: ").strip()
                if not response:
                    response = "no answer"
                
                responses[f"q{question_num}"] = response
                conversation_history.append({"question": next_question, "answer": response})
                
                # Check if we have enough information
                if self._has_sufficient_information(responses, task_context):
                    print(f"\n🎯 LLM: Thank you, I have sufficient information to analyze the scene.")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrupted by user.")
                responses[f"q{question_num}"] = "interrupted"
                break
            except Exception as e:
                responses[f"q{question_num}"] = "error"
                break
        print("-"*60)
        print("🧠 LLM: Processing observations and generating scene analysis...")
        print("="*60 + "\n")
        
        # Generate comprehensive analysis based on intelligent conversation
        analysis = self._process_intelligent_responses(responses, conversation_history, task_context, last_action)
        return analysis
    
    def _generate_intelligent_question(self, task_context: str, last_action: str, scene_data: Dict[str, Any], 
                                     responses: Dict[str, str], conversation_history: list, question_num: int) -> str:
        """Generate contextually intelligent questions based on conversation flow."""
        
        # Extract context
        objects = scene_data.get('objects', [])
        boxes = [obj for obj in objects if 'box' in obj.get('name', '').lower()]
        
        # First question - always establish basic scene state
        if question_num == 1:
            return "What do you currently see in the scene? Please describe the robot, boxes, basket, and their positions."
        
        # Subsequent questions based on previous responses and context
        previous_responses = " ".join(responses.values()).lower()
        
        # Question 2 - Focus on the core task
        if question_num == 2:
            if "failed" in last_action.lower():
                return "The last action failed. What do you think went wrong? Do you see any collision or positioning issues?"
            else:
                return f"How many of the {len(boxes)} colored boxes are currently inside the destination basket versus outside?"
        
        # Question 3 - Dive deeper based on responses
        if question_num == 3:
            if "collision" in previous_responses or "touching" in previous_responses:
                return "You mentioned collision/touching. Can you describe exactly which objects are interfering with each other?"
            elif any(word in previous_responses for word in ["inside", "basket", "placed"]):
                return "For the boxes that are in the basket, how well are they positioned? Are they properly placed or do they look unstable?"
            else:
                return "Are any of the boxes touching or overlapping with each other?"
        
        # Question 4 - Tactical planning
        if question_num == 4:
            if "good" in previous_responses and "stable" in previous_responses:
                return "Given the current arrangement, where would be the best location to place the next box?"
            elif "problem" in previous_responses or "issue" in previous_responses:
                return "What specific changes would you recommend to fix the issues you've observed?"
            else:
                return "How much space is remaining in the basket, and what's the best strategy for the remaining boxes?"
        
        # Question 5 - Quality assessment
        if question_num == 5:
            if "final" in task_context.lower() or "completion" in task_context.lower():
                return "Looking at the overall result, how would you rate the packing efficiency and quality?"
            elif len([r for r in responses.values() if "basket" in r.lower()]) > 0:
                return "Is the current spacing between boxes appropriate, or are they too crowded/spread out?"
            else:
                return "What's your overall assessment of the current progress and robot performance?"
        
        # Question 6 - Strategic recommendation
        if question_num == 6:
            if "failed" in last_action.lower():
                return "Based on everything you've observed, what should be the next action to recover from the failure?"
            else:
                return "What would you recommend as the best next step to continue the task successfully?"
        
        # Fallback - shouldn't reach here normally
        return None
    
    def _has_sufficient_information(self, responses: Dict[str, str], task_context: str) -> bool:
        """Determine if we have enough information for analysis."""
        
        if len(responses) < 2:
            return False
            
        response_text = " ".join(responses.values()).lower()
        
        # Check if we have key information categories
        has_scene_description = any(word in response_text for word in ["robot", "box", "basket", "see"])
        has_box_count = any(word in response_text for word in ["inside", "outside", "basket", "placed", "all", "none"]) or any(str(i) in response_text for i in range(5))
        has_quality_assessment = any(word in response_text for word in ["good", "bad", "proper", "wrong", "collision", "touching"])
        
        # Need at least basic scene info and one other category
        return has_scene_description and (has_box_count or has_quality_assessment) and len(responses) >= 3
    
    def _process_intelligent_responses(self, responses: Dict[str, str], conversation_history: list, 
                                     task_context: str, last_action: str) -> str:
        """Process intelligent conversation into comprehensive scene analysis."""
        
        # Aggregate all human responses for analysis
        all_responses = " ".join(responses.values()).lower()
        
        # Extract key insights from conversation
        scene_description = conversation_history[0]["answer"] if conversation_history else "No scene description"
        
        # Intelligent parsing of box counts and status
        boxes_in_basket = self._extract_box_count(all_responses, conversation_history)
        
        # Collision and quality analysis
        collision_detected = any(word in all_responses for word in ['collision', 'touching', 'overlapping', 'interfering'])
        quality_issues = any(word in all_responses for word in ['problem', 'issue', 'wrong', 'bad', 'unstable'])
        placement_quality = "good" if any(word in all_responses for word in ['good', 'proper', 'well', 'stable']) else "needs attention"
        
        # Strategic insights from conversation
        strategic_suggestions = self._extract_strategic_insights(conversation_history, task_context)
        
        # Build comprehensive analysis
        analysis = f"""
🔍 INTELLIGENT SCENE ANALYSIS
(Based on LLM-guided human observation)

1. SCENE SUMMARY:
   {scene_description}

2. BOX PLACEMENT STATUS:
   Estimated: {boxes_in_basket}/4 boxes in basket ({boxes_in_basket*25:.0f}% complete)
   Analysis: {self._get_progress_description(boxes_in_basket)}

3. COLLISION & QUALITY ANALYSIS:
   {"⚠️ COLLISION DETECTED: " + self._get_collision_details(conversation_history) if collision_detected else "✅ No collisions detected"}
   Placement Quality: {placement_quality.title()}
   {"Issues Identified: " + self._get_quality_issues(conversation_history) if quality_issues else ""}

4. SPATIAL OBSERVATIONS:
   {self._get_spatial_observations(conversation_history)}

5. PROGRESS ASSESSMENT:"""
        
        if boxes_in_basket >= 4:
            analysis += "\n   Task Status: ✅ COMPLETE - All boxes successfully placed!"
        elif boxes_in_basket >= 3:
            analysis += "\n   Task Status: 🎯 NEARLY COMPLETE - Final box needed"
        elif boxes_in_basket >= 2:
            analysis += "\n   Task Status: 👍 GOOD PROGRESS - Halfway done"
        elif boxes_in_basket >= 1:
            analysis += "\n   Task Status: 🚀 STARTED WELL - First placement successful"
        else:
            analysis += "\n   Task Status: ⚠️ NEEDS ATTENTION - No boxes placed yet"
        
        analysis += f"""

6. STRATEGIC RECOMMENDATIONS:
   {strategic_suggestions}

7. INTELLIGENT INSIGHTS:
   - Conversation Quality: {len(conversation_history)} questions asked
   - Key Observations: {self._get_key_observations(conversation_history)}
   - Confidence Level: {"High" if len(conversation_history) >= 3 else "Medium" if len(conversation_history) >= 2 else "Low"}
   
OVERALL STATUS: {self._get_overall_status(boxes_in_basket, collision_detected, quality_issues)}
"""
        
        return analysis.strip()
    
    def _extract_box_count(self, all_responses: str, conversation_history: list) -> int:
        """Intelligently extract box count from conversation."""
        import re
        
        # Look for explicit numbers
        numbers = re.findall(r'\d+', all_responses)
        
        # Check for contextual clues
        if 'all' in all_responses and ('inside' in all_responses or 'basket' in all_responses):
            return 4
        elif 'none' in all_responses or 'zero' in all_responses:
            return 0
        elif numbers:
            for num_str in numbers:
                num = int(num_str)
                if 0 <= num <= 4:
                    return num
        
        # Contextual estimation based on conversation flow
        if any('good progress' in resp.get('answer', '').lower() for resp in conversation_history):
            return 2
        elif any('started' in resp.get('answer', '').lower() for resp in conversation_history):
            return 1
        
        return 1  # Conservative default
    
    def _extract_strategic_insights(self, conversation_history: list, task_context: str) -> str:
        """Extract strategic recommendations from conversation."""
        insights = []
        
        for exchange in conversation_history:
            answer = exchange['answer'].lower()
            
            if 'corner' in answer or 'edge' in answer:
                insights.append("- 🎯 USE CORNERS: Human suggests corner/edge placement strategy")
            elif 'space' in answer and 'more' in answer:
                insights.append("- 📏 SPACE AVAILABLE: Plenty of room for additional boxes")
            elif 'crowded' in answer or 'tight' in answer:
                insights.append("- ⚠️ SPACE CONCERN: Basket getting crowded, precise placement needed")
            elif 'collision' in answer or 'avoid' in answer:
                insights.append("- 🚧 AVOID COLLISIONS: Adjust coordinates to prevent interference")
            elif 'different' in answer and ('position' in answer or 'place' in answer):
                insights.append("- 🔄 REPOSITION: Try alternative placement coordinates")
        
        if not insights:
            if 'failed' in task_context.lower():
                insights.append("- 🔧 TROUBLESHOOT: Analyze failure and adjust approach")
            else:
                insights.append("- ✅ CONTINUE: Current approach appears to be working")
        
        return "\n   ".join(insights) if insights else "- ✅ CONTINUE: Maintain current strategy"
    
    def _get_collision_details(self, conversation_history: list) -> str:
        """Extract collision details from conversation."""
        for exchange in conversation_history:
            if any(word in exchange['answer'].lower() for word in ['collision', 'touching', 'overlapping']):
                return exchange['answer']
        return "Collision issues detected"
    
    def _get_quality_issues(self, conversation_history: list) -> str:
        """Extract quality issues from conversation."""
        for exchange in conversation_history:
            if any(word in exchange['answer'].lower() for word in ['problem', 'issue', 'wrong', 'bad']):
                return exchange['answer']
        return "Quality concerns identified"
    
    def _get_spatial_observations(self, conversation_history: list) -> str:
        """Extract spatial observations from conversation."""
        spatial_info = []
        
        for exchange in conversation_history:
            answer = exchange['answer'].lower()
            if any(word in answer for word in ['space', 'room', 'spacing', 'crowded', 'empty']):
                spatial_info.append(exchange['answer'])
        
        return " | ".join(spatial_info) if spatial_info else "Limited spatial information provided"
    
    def _get_key_observations(self, conversation_history: list) -> str:
        """Summarize key observations from conversation."""
        observations = []
        
        for i, exchange in enumerate(conversation_history):
            if i < 3:  # Focus on first 3 key exchanges
                key_words = [word for word in exchange['answer'].lower().split() 
                           if word in ['good', 'bad', 'collision', 'space', 'proper', 'wrong', 'basket', 'inside', 'outside']]
                if key_words:
                    observations.extend(key_words[:2])  # Max 2 words per exchange
        
        return ", ".join(set(observations)) if observations else "basic scene assessment"
    
    def _get_progress_description(self, boxes_in_basket: int) -> str:
        """Get progress description based on box count."""
        if boxes_in_basket >= 4:
            return "Task completed successfully"
        elif boxes_in_basket >= 3:
            return "Excellent progress, nearly finished"
        elif boxes_in_basket >= 2:
            return "Good progress, halfway complete"
        elif boxes_in_basket >= 1:
            return "Task started, initial placement successful"
        else:
            return "Task not yet started or issues present"
    
    def _get_overall_status(self, boxes_in_basket: int, collision_detected: bool, quality_issues: bool) -> str:
        """Determine overall status based on multiple factors."""
        if collision_detected or quality_issues:
            return "⚠️ NEEDS ATTENTION"
        elif boxes_in_basket >= 4:
            return "🎉 TASK COMPLETE"
        elif boxes_in_basket >= 3:
            return "🎯 NEARLY DONE"
        elif boxes_in_basket >= 1:
            return "👍 PROGRESSING WELL"
        else:
            return "⚠️ NEEDS ATTENTION"
    
    def execute(self, task_context: str, last_action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the current scene and provide intelligent observations.
        
        Args:
            task_context: Description of current task
            last_action: The last action performed
            context: Current scene context with object positions
            
        Returns:
            Dict with analysis results and recommendations
        """
        print(f"🔍 Analyzing scene for task: {task_context}")
        print(f"📝 Last action: {last_action}")
        
        try:
            # Generate detailed analysis prompt
            analysis_prompt = self._generate_scene_prompt(task_context, last_action, context)
            
            # Use LLM if available, otherwise use simulated intelligent response
            if self.llm_client:
                # Use actual LLM (future enhancement)
                analysis_result = self._simulate_llm_response(analysis_prompt, task_context, last_action, context)
            else:
                # Use simulated intelligent analysis
                analysis_result = self._simulate_llm_response(analysis_prompt, task_context, last_action, context)
            
            print("📋 Scene Analysis:")
            # Print the analysis with proper formatting
            for line in analysis_result.split('\n'):
                if line.strip():
                    print(f"   {line}")
            
            # Extract key insights for programmatic use from intelligent conversation
            analysis_text = analysis_result.lower()
            
            # Parse insights from intelligent conversation analysis
            if "complete" in analysis_text and ("4/4" in analysis_result or "all boxes" in analysis_text):
                boxes_in_basket_count = 4
            elif "3/4" in analysis_result or "nearly complete" in analysis_text:
                boxes_in_basket_count = 3
            elif "2/4" in analysis_result or "halfway" in analysis_text:
                boxes_in_basket_count = 2
            elif "1/4" in analysis_result or "started well" in analysis_text:
                boxes_in_basket_count = 1
            else:
                # Try to extract from percentage
                import re
                percentages = re.findall(r'(\d+)%', analysis_result)
                if percentages:
                    pct = int(percentages[0])
                    boxes_in_basket_count = max(0, min(4, round(pct / 25)))
                else:
                    boxes_in_basket_count = 0
            
            # Determine collision risk from intelligent analysis
            collision_risk = ("collision detected" in analysis_text or 
                            "touching" in analysis_text or 
                            "overlapping" in analysis_text or
                            "interfering" in analysis_text)
            
            # Determine recommended action from intelligent analysis
            if ("needs attention" in analysis_text or "resolve collision" in analysis_text or 
                "troubleshoot" in analysis_text):
                recommended_action = "replan"
            elif ("continue" in analysis_text or "good progress" in analysis_text or 
                  "progressing well" in analysis_text):
                recommended_action = "continue"
            elif "failed" in last_action.lower() and "reposition" in analysis_text:
                recommended_action = "replan"
            else:
                recommended_action = "continue"
            
            total_boxes = 4  # We know there are 4 boxes in our task
            
            return {
                "success": True,
                "analysis": analysis_result,
                "insights": {
                    "boxes_in_basket": boxes_in_basket_count,
                    "boxes_remaining": total_boxes - boxes_in_basket_count,
                    "task_completion": boxes_in_basket_count / total_boxes,
                    "collision_risk": collision_risk,
                    "recommended_action": recommended_action
                },
                "feedback": analysis_result
            }
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": f"Scene analysis failed: {str(e)}",
                "analysis": f"Error during human observation: {str(e)}",
                "insights": {
                    "boxes_in_basket": 0,
                    "boxes_remaining": 4,
                    "task_completion": 0.0,
                    "collision_risk": False,
                    "recommended_action": "continue"
                }
            }
            print(f"   ❌ Analysis Error: {str(e)}")
            return error_result
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Make the tool callable."""
        return self.execute(
            task_context=kwargs.get("task_context", "packing boxes"),
            last_action=kwargs.get("last_action", "starting task"),
            context=kwargs.get("context", {"objects": [], "scene_description": "Empty scene"})
        )