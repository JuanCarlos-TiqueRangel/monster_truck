#!/usr/bin/env python3
"""
simple_render_test.py
---------------------
Minimal test to debug why RENDER=True isn't showing the viewer.

    python3 simple_render_test.py [render]

Example:
    python3 simple_render_test.py True    # Try to show viewer
    python3 simple_render_test.py False   # Headless
"""
import sys
import os
import mujoco
import mujoco.viewer as mj_viewer
from pathlib import Path
import time

render = sys.argv[1].lower() == 'true' if len(sys.argv) > 1 else False

print(f"\n{'='*70}")
print(f"SIMPLE RENDER TEST")
print(f"{'='*70}")
print(f"RENDER mode: {render}")
print(f"Display: {os.environ.get('DISPLAY', 'Not set')}")
print(f"{'='*70}\n")

# Load model
xml_path = Path(__file__).parent / "monster_truck_flip_2d.xml"
print(f"Loading model from: {xml_path}")

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

print(f"✓ Model loaded (timestep={model.opt.timestep}s)")

if render:
    print("\n" + "="*70)
    print("📺 ATTEMPTING TO LAUNCH VIEWER...")
    print("="*70)
    print("If you see a window, it worked!")
    print("(Window may take 1-2 seconds to appear)\n")

    try:
        with mj_viewer.launch_passive(model, data) as viewer:
            print("✓ Viewer launched successfully")
            print("✓ Running simulation for 5 seconds...")

            start_time = time.time()
            while viewer.is_running() and data.time < 5.0:
                mujoco.mj_step(model, data)
                viewer.sync()

            elapsed = time.time() - start_time
            print(f"✓ Simulation completed ({elapsed:.1f}s elapsed)")
            print("✓ Viewer window should still be visible")
            print("\nKeep window open to verify it's working.")
            print("Close window or press Ctrl+C to exit.\n")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        print(f"\nViewer failed to launch. Possible causes:")
        print(f"  1. No X11 display (SSH without -X flag)")
        print(f"  2. Graphics driver issue")
        print(f"  3. Display server not running")
        print(f"\nTry: ssh -X user@host")
        print(f"Or use: RENDER=False for headless mode\n")

else:
    print("\n⚡ HEADLESS MODE (no viewer)")
    print("Running simulation for 5 seconds...\n")

    start_time = time.time()
    for step in range(5000):
        mujoco.mj_step(model, data)

    elapsed = time.time() - start_time
    print(f"✓ Simulation completed ({elapsed:.1f}s)")
    print(f"  (Real sim time: {data.time:.1f}s)")
    print(f"  Speed: {data.time/elapsed:.1f}x faster than real-time\n")

print("="*70)
print("✓ TEST COMPLETE")
print("="*70 + "\n")
