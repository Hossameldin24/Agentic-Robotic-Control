"""Base tool class."""
from typing import Dict, Any


class BaseTool:
    """Base class for all tools."""
    
    def __init__(self):
        self.name = "base_tool"
        self.description = "Base tool class"
    
    def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the tool."""
        raise NotImplementedError("Tools must implement execute method")