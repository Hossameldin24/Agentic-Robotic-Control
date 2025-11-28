"""PyBullet environment wrapper for Inner Monologue."""
import sys
from pathlib import Path
import logging

# Add LLM-TAMP to path directly
current_dir = Path(__file__).parent
llm_tamp_path = current_dir.parent.parent / "LLM-TAMP" 
if str(llm_tamp_path) not in sys.path:
    sys.path.insert(0, str(llm_tamp_path))

try:
    from envs.pack_compact_env import PackCompactEnv
    from envs.pb_env import PybulletEnv
except ImportError as e:
    logging.warning(f"Could not import LLM-TAMP environment: {e}")
    PackCompactEnv = None
    PybulletEnv = None

logger = logging.getLogger(__name__)


class InnerMonologueEnv:
    """
    Wrapper for PyBullet environment from LLM-TAMP.
    
    Provides a clean interface for the Inner Monologue agent system.
    """
    
    def __init__(self, use_gui: bool = None):
        """
        Initialize the PyBullet environment.
        
        Args:
            use_gui: Whether to show GUI (defaults to settings.use_gui)
        """
        if PackCompactEnv is None:
            raise ImportError(
                "Could not import LLM-TAMP environment. "
                "Make sure LLM-TAMP is available at the configured path."
            )
        
        self.use_gui = use_gui if use_gui is not None else True  # Default to GUI
        self.env = PackCompactEnv()
        self._initialized = False
        
        # Initialize the environment immediately
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Initialize the environment with default setup."""
        try:
            print(f"[ENV] Starting PyBullet (GUI={self.use_gui})...")
            
            # Reset with default configuration to start PyBullet (matching LLM-TAMP)
            basket = {
                "x": 0.7,   # Match LLM-TAMP medium basket configuration
                "y": 0.0,
                "w": 0.32,  # Match LLM-TAMP basket width
                "l": 0.21   # Match LLM-TAMP basket length
            }
            
            boxes = self._default_boxes()
            
            # This will initialize PyBullet with GUI
            obs, obs_text = self.env.reset(
                basket=basket,
                boxes=boxes,
                use_gui=self.use_gui
            )
            
            self._initialized = True
            print(f"[ENV] Ready! Objects: {len(self.env.objects)} loaded")
            
        except Exception as e:
            logger.error(f"Failed to initialize environment: {e}")
            self._initialized = False
            raise
    
    def reset(
        self,
        basket: dict = None,
        boxes: dict = None,
        **kwargs
    ) -> tuple:
        """
        Reset the environment.
        
        Args:
            basket: Basket configuration
            boxes: Boxes configuration
            **kwargs: Additional reset parameters
            
        Returns:
            Tuple of (observation, observation_text)
        """
        if not self._initialized:
            # If not initialized, do full initialization
            self._initialize_environment()
            return None, "Environment initialized"
            
        if basket is None:
            basket = {
                "x": 0.7,   # Match LLM-TAMP configuration
                "y": 0.0,
                "w": 0.32,  # Match LLM-TAMP basket dimensions
                "l": 0.21
            }
        
        if boxes is None:
            boxes = self._default_boxes()
        
        print(f"[ENV] Resetting scene...")
        obs, obs_text = self.env.reset(
            basket=basket,
            boxes=boxes,
            use_gui=self.use_gui,
            **kwargs
        )
        
        return obs, obs_text
    
    def _default_boxes(self) -> dict:
        """Create default box configuration matching LLM-TAMP easy_box_medium_basket.json."""
        return {
            0: {
                "name": "red_box",
                "color": [1, 0, 0, 1],
                "w": 0.035,  # Small box from LLM-TAMP
                "l": 0.035,
                "h": 0.07,
                "x": 0.39,   # Simplified from LLM-TAMP 0.3916
                "y": -0.47,  # Simplified from LLM-TAMP -0.4744
                "z": 0.06
            },
            1: {
                "name": "blue_box",
                "color": [0, 0, 1, 1],
                "w": 0.07,   # Large box from LLM-TAMP
                "l": 0.07,
                "h": 0.07,
                "x": 0.39,   # Simplified from LLM-TAMP 0.3869
                "y": -0.01,  # Simplified from LLM-TAMP -0.0059
                "z": 0.06
            },
            2: {
                "name": "green_box",
                "color": [0, 1, 0, 1],
                "w": 0.035,  # Small box from LLM-TAMP
                "l": 0.035,
                "h": 0.07,
                "x": 0.39,   # Match x position
                "y": 0.47,   # Positive y for spacing
                "z": 0.06
            },
            3: {
                "name": "yellow_box",
                "color": [1, 1, 0, 1],
                "w": 0.035,  # Small box from LLM-TAMP
                "l": 0.035,
                "h": 0.07,
                "x": 0.39,   # Match x position
                "y": 0.24,   # Between green and blue
                "z": 0.06
            }
        }
    
    def get_observation(self) -> tuple:
        """Get current observation."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.get_observation()
    
    def check_goal(self) -> tuple:
        """Check if goal is achieved."""
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.env.check_goal()
    
    def destroy(self):
        """Destroy the environment."""
        if self._initialized:
            self.env.destroy()
            self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.destroy()


