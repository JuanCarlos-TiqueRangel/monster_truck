# Fast headless SVGP + MPPI training

This project now has two modes:

1. **ROS2 real-time mode**
   - `mujoco_model/planar_mujoco_model.py`
   - `SVGP/online_learning_mppi_SVGP.py`
   - Good for ROS debugging and hardware-like timing.

2. **Headless in-process training mode**
   - `mujoco_model/planar_headless_env.py`
   - `SVGP/online_learning_mppi_SVGP_headless.py`
   - Good for fast learning runs without ROS2 pub/sub or wall-clock timers.

## Main changes applied
- Viewer is optional and **off by default** in `planar_mujoco_model.py`.
- Live plotting is disabled by default in the SVGP ROS controller.
- MPPI defaults are lighter for quick learning checks (`horizon=20`, `num_rollouts=512`).
- Top-level MPPI action signature is backward compatible and no longer rebuilds local models twice.
- Added a direct MuJoCo in-process environment for Gym-like headless stepping.

## Run the fast headless trainer
From the project root:

```bash
python3 SVGP/online_learning_mppi_SVGP_headless.py --episodes 50
```

Optional viewer for evaluation only:

```bash
python3 SVGP/online_learning_mppi_SVGP_headless.py --episodes 5 --viewer
```

Tune MPPI load quickly:

```bash
python3 SVGP/online_learning_mppi_SVGP_headless.py --episodes 20 --horizon 25 --rollouts 1024
```

## Run the ROS2 MuJoCo node without rendering
Viewer is off by default now:

```bash
python3 mujoco_model/planar_mujoco_model.py
```

To force the viewer on:

```bash
MUJOCO_ENABLE_VIEWER=1 python3 mujoco_model/planar_mujoco_model.py
```
