"""
Intelligent placement planning for compact packing in baskets.
Considers box dimensions, basket constraints, and already-placed objects.
"""
import math
from typing import List, Dict, Tuple, Optional


class PlacementPlanner:
    """Smart placement planning with spatial reasoning."""
    
    def __init__(self, basket_info: Dict):
        """Initialize with basket dimensions."""
        self.basket_x = basket_info.get('x', 0.7)
        self.basket_y = basket_info.get('y', 0.0)
        self.basket_w = basket_info.get('w', 0.32)  # width
        self.basket_l = basket_info.get('l', 0.21)  # length (y-direction)
        
        # Calculate basket boundaries
        self.basket_x_min = self.basket_x - self.basket_w / 2
        self.basket_x_max = self.basket_x + self.basket_w / 2
        self.basket_y_min = self.basket_y - self.basket_l / 2
        self.basket_y_max = self.basket_y + self.basket_l / 2
        
        print(f"📐 Basket boundaries: X({self.basket_x_min:.3f} to {self.basket_x_max:.3f}), Y({self.basket_y_min:.3f} to {self.basket_y_max:.3f})")
    
    def calculate_optimal_placement(self, 
                                  box_to_place: Dict,
                                  placed_boxes: List[Dict],
                                  attempt: int = 0) -> Tuple[float, float]:
        """
        Calculate optimal placement coordinates for a box.
        
        Args:
            box_to_place: Dict with box info (name, w, l, h)
            placed_boxes: List of already placed boxes with positions
            attempt: Retry attempt number for replanning
            
        Returns:
            Tuple of (x, y) coordinates
        """
        box_w = box_to_place.get('w', 0.05)
        box_l = box_to_place.get('l', 0.05)
        box_name = box_to_place.get('name', 'unknown')
        
        print(f"🧮 Planning placement for {box_name} (w={box_w:.3f}, l={box_l:.3f}), attempt {attempt + 1}")
        
        # Strategy 1: Grid-based placement for first attempt
        if attempt == 0:
            return self._grid_placement(box_w, box_l, placed_boxes)
        
        # Strategy 2: Linear arrangement for second attempt
        elif attempt == 1:
            return self._linear_placement(box_w, box_l, placed_boxes)
            
        # Strategy 3: Compact corners for third attempt
        elif attempt == 2:
            return self._corner_placement(box_w, box_l, placed_boxes)
            
        # Strategy 4: Random within basket for final attempt
        else:
            return self._fallback_placement(box_w, box_l)
    
    def _grid_placement(self, box_w: float, box_l: float, placed_boxes: List[Dict]) -> Tuple[float, float]:
        """Grid-based placement strategy."""
        # Try 2x2 grid positions within basket
        grid_positions = [
            (self.basket_x - 0.08, self.basket_y - 0.05),  # Bottom-left
            (self.basket_x + 0.08, self.basket_y - 0.05),  # Bottom-right
            (self.basket_x - 0.08, self.basket_y + 0.05),  # Top-left
            (self.basket_x + 0.08, self.basket_y + 0.05),  # Top-right
        ]
        
        for x, y in grid_positions:
            if self._is_position_valid(x, y, box_w, box_l, placed_boxes):
                print(f"✅ Grid placement found: ({x:.3f}, {y:.3f})")
                return x, y
                
        # Fallback to center
        return self.basket_x, self.basket_y
    
    def _linear_placement(self, box_w: float, box_l: float, placed_boxes: List[Dict]) -> Tuple[float, float]:
        """Linear arrangement strategy."""
        # Place boxes in a line from left to right
        start_x = self.basket_x_min + box_w/2 + 0.01  # Small margin
        
        for i in range(4):  # Try 4 positions along X
            x = start_x + i * (box_w + 0.02)  # 2cm spacing
            y = self.basket_y
            
            if x + box_w/2 <= self.basket_x_max and self._is_position_valid(x, y, box_w, box_l, placed_boxes):
                print(f"✅ Linear placement found: ({x:.3f}, {y:.3f})")
                return x, y
        
        return self.basket_x, self.basket_y
    
    def _corner_placement(self, box_w: float, box_l: float, placed_boxes: List[Dict]) -> Tuple[float, float]:
        """Corner placement strategy."""
        # Try corners and edges
        corner_positions = [
            (self.basket_x_min + box_w/2, self.basket_y_min + box_l/2),  # Bottom-left corner
            (self.basket_x_max - box_w/2, self.basket_y_min + box_l/2),  # Bottom-right corner
            (self.basket_x_min + box_w/2, self.basket_y_max - box_l/2),  # Top-left corner
            (self.basket_x_max - box_w/2, self.basket_y_max - box_l/2),  # Top-right corner
            (self.basket_x, self.basket_y_min + box_l/2),                # Bottom center
            (self.basket_x, self.basket_y_max - box_l/2),                # Top center
        ]
        
        for x, y in corner_positions:
            if self._is_position_valid(x, y, box_w, box_l, placed_boxes):
                print(f"✅ Corner placement found: ({x:.3f}, {y:.3f})")
                return x, y
                
        return self.basket_x, self.basket_y
    
    def _fallback_placement(self, box_w: float, box_l: float) -> Tuple[float, float]:
        """Fallback to safe center position."""
        print(f"⚠️  Using fallback center placement")
        return self.basket_x, self.basket_y
    
    def _is_position_valid(self, x: float, y: float, box_w: float, box_l: float, placed_boxes: List[Dict]) -> bool:
        """Check if position is valid (within basket and no collisions)."""
        
        # Check basket boundaries
        if (x - box_w/2 < self.basket_x_min or x + box_w/2 > self.basket_x_max or
            y - box_l/2 < self.basket_y_min or y + box_l/2 > self.basket_y_max):
            return False
        
        # Check collisions with placed boxes
        for placed_box in placed_boxes:
            placed_x = placed_box.get('center', {}).get('x', placed_box.get('x', 0))
            placed_y = placed_box.get('center', {}).get('y', placed_box.get('y', 0))
            placed_w = placed_box.get('w', 0.05)
            placed_l = placed_box.get('l', 0.05)
            
            # Check overlap with margin
            margin = 0.02  # 2cm margin between boxes
            if (abs(x - placed_x) < (box_w + placed_w)/2 + margin and 
                abs(y - placed_y) < (box_l + placed_l)/2 + margin):
                return False
        
        return True
    
    def get_placement_info(self, x: float, y: float, box_info: Dict) -> Dict:
        """Get detailed placement information."""
        box_w = box_info.get('w', 0.05)
        box_l = box_info.get('l', 0.05)
        
        return {
            'x': x,
            'y': y,
            'theta': 0.0,
            'within_basket': self._is_position_valid(x, y, box_w, box_l, []),
            'margins': {
                'left': x - box_w/2 - self.basket_x_min,
                'right': self.basket_x_max - (x + box_w/2),
                'bottom': y - box_l/2 - self.basket_y_min,
                'top': self.basket_y_max - (y + box_l/2)
            }
        }