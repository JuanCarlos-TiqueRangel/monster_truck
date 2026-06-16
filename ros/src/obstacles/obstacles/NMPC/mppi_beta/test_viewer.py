#!/usr/bin/env python3
"""
test_viewer.py
--------------
Simple test to verify MuJoCo viewer works.
If you see a window pop up with the truck, the viewer works!

    python3 test_viewer.py
"""
import mujoco
import mujoco.viewer as mj_viewer
from pathlib import Path

XML_PATH = Path(__file__).parent / "monster_truck_flip_2d.xml"

print("\n" + "="*70)
print("MuJoCo Viewer Test")
print("="*70)
print(f"\nLoading model from: {XML_PATH}")
print("If a window appears, viewer is working!\n")

try:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    print("✓ Model loaded successfully")
    print(f"  Simulation time step: {model.opt.timestep}")

    # Launch viewer
    print("\nLaunching viewer...")
    print("(A window should pop up. Close it to continue.)\n")

    with mj_viewer.launch_passive(model, data) as viewer:
        # Run for 5 seconds
        while viewer.is_running() and data.time < 5.0:
            mujoco.mj_step(model, data)
            viewer.sync()

    print("✓ Viewer closed successfully")
    print("="*70)
    print("SUCCESS: Viewer is working!")
    print("="*70 + "\n")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nPossible causes:")
    print("1. X11 display not available (SSH without -X flag)")
    print("2. MuJoCo display extension not installed")
    print("3. No graphics support in environment")
    print("\nSolution: Run with RENDER=False for headless mode")
    print("="*70 + "\n")
