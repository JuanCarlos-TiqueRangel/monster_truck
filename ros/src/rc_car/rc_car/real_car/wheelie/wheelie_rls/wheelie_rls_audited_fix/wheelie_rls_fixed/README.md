# Audited RLS / MPPI wheelie implementation

**Status: implementation defects corrected; the supplied learned model is NOT approved for real-car balancing.**

Recorded-data tests still reveal model divergence. The node therefore starts with `drive_enabled=False`. The included weights are a reproducible diagnostic candidate, not a hardware-validated model. Read `AUDIT_REPORT.md` before changing that setting. No commands were sent to a vehicle during this audit.

## Files

The five replacements are `rls.py`, `train_rls.py`, `mppi.py`, `params_mppi.py`, and `wheelie_mppi_rls_node.py`. Keep them together in a clean directory. Do not mix them with old `mppi/` or `rls/` packages, or add those old directories to `sys.path`.

The additional scripts are `validate_rls.py`, `calibrate_rls.py`, `benchmark_mppi.py`, `simulate_model_only.py`, `audit_original.py`, `replay_original_mppi.py`, and `test_stack.py`. Numerical results, replay traces, figures and copies of the original source files are under `audit/`. The diagnostic candidate is `wheelie_nn/rls_learned_weights_v2.npz`.

Dependencies for offline work: NumPy, pandas, SciPy and PyTorch. Matplotlib is needed only for live plotting. The ROS node additionally needs your sourced ROS 2 environment, `rclpy`, `nav_msgs`, `geometry_msgs` and `std_msgs`. CPU tests here used PyTorch 2.10.0+cpu. CUDA and ROS execution were not available here; no GPU or hardware performance claim is made.

## Reproduce without driving

Run inside this directory, replacing `/path/to/dataset` with the folder containing the 147 top-level identification CSVs. Do not point to `backup_dataset`, and do not combine it recursively with the main folder.

```bash
python3 train_rls.py --dataset /path/to/dataset
python3 validate_rls.py --dataset /path/to/dataset
python3 -m unittest test_stack -v
python3 simulate_model_only.py --device cpu
python3 benchmark_mppi.py --device cuda
```

`train_rls.py` starts from zero every time. It does not resume old weights, overwrite your old `rls_learned_weights.npz`, alter the CSVs, or include held-out IDs divisible by five in fitting. It writes the distinct `_v2.npz` file and a screening CSV. Model types, feature definitions, training/test filenames and dataset hashes are saved with the parameters.

The shipped model was trained on 1,735 accepted causal transitions from 115 experiments; 29 complete experiments were excluded from fitting. Lag/ridge calibration uses IDs congruent to 1 modulo 5, with IDs congruent to 2, 3 or 4 for fitting during that search. The final fit adds calibration experiments back, but still excludes all IDs divisible by 5.

```bash
python3 calibrate_rls.py --dataset /path/to/dataset
python3 audit_original.py --dataset /path/to/dataset
python3 replay_original_mppi.py
```

The calibration selected three input-delay samples and ridge 10 within the original 18-coefficient feature family after requiring a modeled trim and admissible calibration predictions. Three samples is an **empirical effective pipeline lag**, not a measured 120 ms physical ESC delay. The CSV logger's actual command/measurement timestamp convention is still unverified.

## ROS dry run

After sourcing your ROS 2 environment:

```bash
python3 wheelie_mppi_rls_node.py --ros-args -p drive_enabled:=false -p device:=cuda
```

The node still publishes zero to `/drive` during dry run. **Dry run is not a passive subscriber:** stop other `/drive` publishers first. It cannot be run alongside an independent controller without interfering with that controller's commands. The default object ID is 810; topics remain `/optitrack/odom_810` and `/optitrack/twist_810`.

The node does not subscribe to acceleration or update RLS online. It pairs pose/twist by header timestamp, unwraps planar pitch, uses only published commands in the lag queue, and logs `u_plan` separately from `u_applied`. Here `u_applied` means published on ROS; there is no ESC acknowledgment or measured motor torque in these files.

Verify the pitch and velocity conventions with propulsion disabled. The revised quaternion formula is yaw-independent for planar motion, but roll must remain small for body `angular.y` to equal pitch rate. The old dataset's quaternion stream and frame metadata were not supplied, so nonplanar/frame consistency cannot be established from the CSVs alone. The node deliberately preserves `twist.linear.x` rather than silently rotating it into a different training convention. Position propagation also assumes a compatible straight-line frame; `q_x=0` avoids asking it to track an incompatible world position.

The step remains 0.04 s: 25 Hz, not 50 Hz. Benchmark on your machine before drive testing. The lag queue assumes approximately uniform control updates; it does not make an overrunning loop real-time. `compile_rollouts` in `params_mppi.py` is optional and defaults to false. `benchmark_mppi.py --compile --device cuda` can test compiled rollouts separately, before enabling them in the node.

There is no `solve_ms > 40 => stop` rule. Zero is published when disarmed, when no recent synchronized state exists, or when the solver has no valid solution. Those cases have distinct statuses in the log. Freshness tolerance and timeout are engineering defaults, not certified safety bounds. Use an independent receiver/ESC command timeout and emergency stop: Python cleanup cannot guarantee delivery after a crash, process kill or radio failure.

## What changed in the model

The original seven linear and eleven angular features are preserved, with the same coefficient order. This remains a model of full velocity-transition dynamics, not a nominal-physics residual model. Internal states remain `[x, v, theta_rad, omega]` in physical units; `/drive` is normalized input, not torque in N m.

Training now identifies `(v[k+1]-v[k])/dt` and `(omega[k+1]-omega[k])/dt` from `state[k]` and `u[k-delay]`. Velocity is updated by that discrete forward model. Position and pitch use trapezoidal kinematics. **Do not integrate these coefficients with RK4**: they identify one-step velocity changes, not an instantaneous continuous vector field.

RLS uses a persistent covariance over the training corpus, forgetting factor 1, and globally scaled feature columns with ridge initialization. Exported coefficients are converted back to physical units. RLS `*_hat` fields are pre-update predictions; `*_hat_post` fields are explicitly labeled posterior fits. Validation never calls `rls_update` or replaces predicted states with measured states after the initial condition.

Known problematic files 003, 011 and 079 are quarantined, and all transitions are screened. Screening limits and accepted counts are explicit in `train_rls.py` and the generated CSV; this does not establish that the remaining data are noise-free. Validation reports screened and unscreened results, including nonfinite and domain-violation counts. Four initial rows are excluded from common-window comparisons because earlier command history is unknown. These comparisons therefore do not validate startup from an unlogged prehistory.

## What changed in MPPI

The sampled command range is restricted to the observed `[0, 0.30]`. Configured state bounds are actually enforced for predictions. Only physical ground contact is projected; invalid trajectories stay invalid rather than being converted to finite plausible states. The weighted mean is evaluated as its own trajectory and compared with the best feasible sample; whichever has lower predicted cost is returned. The displayed endpoint and main cost describe that returned plan.

All declared cost terms are implemented. `q_v` and terminal speed cost are deliberately zero: pitch tracking is not also a request to stop the vehicle. Speed is bounded instead. Sustained-input candidates and two-step noise blocks improve launch exploration. These are practical MPPI-style sampling modifications, not a claim of formal path-integral optimality or closed-loop stability.

## Remaining acceptance problem

The included model fails recorded-data rollouts, even after screening. In particular, `rls_id_045.csv` at start row 9 produces a 427.05 degree 10-step pitch RMSE in unconstrained model replay. MPPI rejects that predicted trajectory, but rejection does not make the dynamics accurate. A controller can exclude a maneuver that the real vehicle actually needs because its model falsely predicts a failure.

The controller follows -55 degrees on its own model in a six-second synthetic test (0.56 degree final-second RMSE). That verifies an implementation path, **not model accuracy on the car**. Do not treat it as a substitute for recorded-data validation, timestamp verification and controlled hardware acceptance tests.
