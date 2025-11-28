"""Main entry point for Inner Monologue agent system."""
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from loguru import logger

from config import settings
from agents.planning_agent import PlanningAgent
from orchestration.graph import InnerMonologueGraph
from environment.pybullet_env import InnerMonologueEnv

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console = Console()


def print_banner():
    """Print welcome banner."""
    banner = """
# 🧠 Inner Monologue Agent System

**Advanced AI Agent Orchestration with LangGraph & Gemini**

- 🤖 PyBullet Robot Simulation
- 🎯 Gemini-Powered Planning
- 🔧 Multi-Tool Agent System
- 📊 LangGraph Orchestration
"""
    console.print(Panel(Markdown(banner), border_style="bold blue"))


def main():
    """Main execution function."""
    print("[DEBUG] main() function starting...")
    print_banner()
    
    print(f"[DEBUG] Settings loaded:")
    print(f"[DEBUG]   google_api_key: {'SET' if settings.google_api_key else 'NOT SET'}")
    print(f"[DEBUG]   gemini_model: {settings.gemini_model}")
    print(f"[DEBUG]   max_iterations: {settings.max_iterations}")
    print(f"[DEBUG]   use_gui: {settings.use_gui}")
    print(f"[DEBUG]   log_level: {settings.log_level}")
    
    # Check API key
    if not settings.google_api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set in environment or .env file[/red]")
        console.print("Please set your Gemini API key in .env file")
        return 1
    
    try:
        # Initialize components
        console.print("[cyan]Initializing components...[/cyan]")
        print("[DEBUG] Starting component initialization...")
        
        # Initialize environment
        console.print("  [yellow]→[/yellow] Setting up PyBullet environment...")
        print("[DEBUG] Initializing InnerMonologueEnv...")
        env = InnerMonologueEnv(use_gui=settings.use_gui)
        print("[DEBUG] InnerMonologueEnv initialized successfully")
        
        # Initialize planning agent
        console.print("  [yellow]→[/yellow] Initializing Gemini planning agent...")
        print("[DEBUG] Initializing PlanningAgent...")
        planning_agent = PlanningAgent()
        print("[DEBUG] PlanningAgent initialized successfully")
        
        # Initialize orchestration graph
        console.print("  [yellow]→[/yellow] Building LangGraph orchestration...")
        print("[DEBUG] Initializing InnerMonologueGraph...")
        graph = InnerMonologueGraph(planning_agent, environment=env)
        print("[DEBUG] InnerMonologueGraph initialized successfully")
        
        console.print("[green]✓ All components initialized[/green]\n")
        
        # Reset environment
        console.print("[cyan]Resetting environment...[/cyan]")
        print("[DEBUG] Calling env.reset()...")
        obs, obs_text = env.reset()
        print(f"[DEBUG] Environment reset complete, obs_text length: {len(obs_text)}")
        console.print(f"[green]✓ Environment ready[/green]\n")
        console.print(f"[dim]{obs_text[:200]}...[/dim]\n")
        
        # Define task
        task_goal = "Pack all boxes into the basket"
        console.print(f"[bold]Task Goal:[/bold] {task_goal}\n")
        print(f"[DEBUG] Task goal set: {task_goal}")
        
        # Run orchestration
        console.print("[cyan]Starting agent orchestration...[/cyan]\n")
        print("[DEBUG] Calling graph.run()...")
        final_state = graph.run(
            task_goal=task_goal,
            max_iterations=settings.max_iterations
        )
        print("[DEBUG] Graph execution completed")
        
        # Print results
        console.print("\n[bold green]Orchestration Complete![/bold green]\n")
        console.print(f"Success: {final_state['success']}")
        console.print(f"Iterations: {final_state['iteration']}")
        print(f"[DEBUG] Final state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'N/A'}")
        
        if final_state.get("reflection"):
            console.print("\n[bold]Reflection:[/bold]")
            console.print(Panel(final_state["reflection"], border_style="yellow"))
        
        # Cleanup
        print("[DEBUG] Cleaning up environment...")
        env.destroy()
        print("[DEBUG] Environment destroyed")
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        return 1
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        print(f"[ERROR] Exception details: {type(e).__name__}: {str(e)}")
        logger.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())

