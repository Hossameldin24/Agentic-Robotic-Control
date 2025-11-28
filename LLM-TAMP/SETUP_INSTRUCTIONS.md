# PyBullet Environment Setup Instructions

## Problem
Your system has Python 3.13, but PyBullet requires compilation on Windows and needs **Microsoft Visual C++ 14.0 or greater** to build from source.

## Solution Options

### Option 1: Install Visual C++ Build Tools (Recommended)

1. Download and install **Microsoft C++ Build Tools** from:
   https://visualstudio.microsoft.com/visual-cpp-build-tools/

2. During installation, select:
   - "Desktop development with C++"
   - Make sure "MSVC v142 - VS 2019 C++ x64/x86 build tools" is checked
   - Make sure "Windows 10 SDK" is checked

3. After installation completes, restart your terminal and run:
   ```powershell
   python -m pip install pybullet-planning
   ```

4. Then run the demo:
   ```powershell
   python demo_env.py
   ```

### Option 2: Use Python 3.11 (Easier)

PyBullet has pre-built wheels for Python 3.11 and earlier.

1. Create a new conda environment with Python 3.11:
   ```powershell
   conda create -n llm-tamp python=3.11 -y
   conda activate llm-tamp
   ```

2. Install all dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the demo:
   ```powershell
   python demo_env.py
   ```

### Option 3: Quick Test with Conda-Forge

Try installing pybullet from conda-forge which may have pre-compiled binaries:

```powershell
conda install -c conda-forge pybullet
pip install pybullet-planning numpy scipy termcolor pyyaml IPython hydra-core
python demo_env.py
```

## What the Demo Does

The `demo_env.py` script will:
- Load a PyBullet simulation environment
- Create a Franka Panda robot arm
- Place 4 colored boxes on a table
- Create a basket
- Display the environment in a 3D GUI window
- Show the initial state without running any motion planning

This lets you see the environment that LLM-TAMP uses for the box-packing tasks, WITHOUT needing any LLM API keys.

## Troubleshooting

If you see `ModuleNotFoundError: No module named 'pybullet'`:
- PyBullet failed to install
- Try Option 1 or Option 2 above

If you see `ModuleNotFoundError: No module named 'pybullet_planning'`:
- Install it separately: `pip install pybullet-planning`

If the GUI window doesn't appear:
- Check that `use_gui=True` in the demo script
- Try running with: `python demo_env.py` (not in background)

## Next Steps

Once you can run the demo successfully, you can:
1. Modify box positions in `demo_env.py`
2. Change basket size
3. Add more boxes
4. Explore the environment API in `envs/pack_compact_env.py`
