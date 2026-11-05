# Numerical wheelie simulation

This directory is an offline numerical experiment. It does not publish ROS
messages or command the physical car.

Run in the existing container:

```bash
docker exec -e MPLBACKEND=Agg -w /ros_ws/src/rc_car/rc_car/numerical_sim monster_truck_ros python3 wheelie_mppi_rls_pytorch.py
```

The default run saves `results/wheelie_mppi_torch_rls_nmpc_cost.png`,
`results/wheelie_mppi_results.npz` and `results/wheelie_mppi_metrics.json`.
It runs headlessly; add `--show` to display the plot. The first CUDA invocation
compiles rollout kernels before the control loop. The reported warm-up time is
separate from control latency. `--no-compile` uses eager Torch; `--device cpu`
works without CUDA but is slower. Dependencies are NumPy, PyTorch (2.x with
Inductor/Triton for compiled CUDA) and Matplotlib, available in monster_truck_ros.

Other options: `--seed 1`, `--duration 10`, `--no-rls`, `--mismatch`,
`--output-dir /tmp/wheelie-results`. Mismatch applies -10% drive gain,
+10% longitudinal damping and +20% pitch damping to the plant only.

## Important reference change

The default is **launch from level ground and balance at zero speed**, with
position free. Using the existing COM estimates, the stationary pitch is
`atan2(0.065, 0.110) = 30.579 degrees`. `pitch_ref=None` computes this angle.
The original 70-degree zero-speed target is not an equilibrium of this model.
`--pitch 70` now raises an explanatory error rather than pretending it can hold
that reference. Tracking other angles during acceleration would require a
separate, feasible time-varying maneuver reference and is not implemented here.
This is an explicit change of the default task, not successful tracking of 70°.

## Definitions and assumptions

- State: `[rear_axle_x_m, rear_axle_v_m_s, nose_up_pitch_rad, pitch_rate_rad_s]`.
- Input: **total driven-wheel torque in N m**, limited to +/-0.44 in this numerical
  test using the supplied full-throttle estimate. The negative limit assumes
  symmetric braking torque; this has not been measured. A normalized input `u`
  would need conversion `tau=0.44*u` before entering this model. Never mix these
  units inside RLS or MPPI. No normalized PWM is sent by this program.
- Mass 1.250 kg, wheelbase 0.18 m. `x_com=0.065` is forward of the rear axle;
  `h_com=0.110` is **above the rear axle**, not above the floor.
- Wheel radius 0.0375 m, COM, damping and uniform-box pitch inertia are retained
  estimates. They are not validated from the photograph. Inertia is recomputed
  for constructor mass/dimension overrides, or can be supplied as `I_body`.
- Rear wheels roll without slip, the rear axle remains at fixed height, and
  front contact/landing is unilateral and inelastic. Rear-contact loss is
  rejected as outside the model. Tire slip, suspension, motor torque-speed
  response, ESC delay, body collisions, flight and flips are not represented.
- Optional `wheel_inertia_equivalent_mass=sum(I_wheel/r**2)` accounts for wheel
  rotational inertia; its default is zero because this value is unmeasured.

## Coupled equations

Define `A=x_com*cos(theta)-h_com*sin(theta)`,
`B=x_com*sin(theta)+h_com*cos(theta)`, `I=I_body+m*(x_com²+h_com²)` and
`M=m+wheel_inertia_equivalent_mass`. For front wheels lifted:

```text
[ M    -m*B ] [ v_dot     ] = [ gain*tau/r - M*c_v*v + m*A*omega² ]
[-m*B    I  ] [ omega_dot ]   [ gain*tau - m*g*A - c_pitch*omega  ]
```

The drive wheel's opposite reaction on the body supplies the `+gain*tau` term.
Translation and pitch must be solved together. With front contact supporting the
body, pitch acceleration is zero until the computed lifting moment is positive.
On landing, the angular rate is reset with a corresponding longitudinal velocity
impulse preserving horizontal generalized momentum. Pitch is never clamped at
the upper operating limit to hide a fall.

This predicts a from-rest launch threshold of approximately 0.203 N m with the
nominal parameters. The former 0.12 N m allowance could not launch this model.
Changing the available torque does not establish real-world torque capability.

## Controller and learning changes

MPPI optimizes **open-loop wheel-torque sequences directly**. There is no PD,
PID, LQR, inverse-dynamics policy, error-to-torque formula, or auxiliary feedback
controller. The only state feedback is the MPPI solve at each control tick.
Every simulated rollout applies its sampled torque sequence unchanged apart from
the actuator clamp.

Each solve runs three exponential-weighted sampling updates over a 0.40-second
horizon (10 steps at 0.04 s), with 4096 sequences. Temporally correlated Gaussian
noise provides smooth exploration around the shifted previous solution. One
eighth of the samples explore a broader, state-independent distribution about
zero to discover launch sequences. The optimizer starts from zero torque. The
weighted mean is used when feasible; only an infeasible mean falls back to a
feasible sampled sequence. Neither route calls another controller.

Stage and terminal costs penalize pitch, rate and speed. The temporal noise
correlation is fixed and does not use pitch or rate errors. These changes replace
the previously added feedback-guided controller entirely.

Samples crossing pitch, speed, pitch-rate or rear-contact limits are rejected at
integration substeps. The weighted mean is rolled out again because averaging
feasible nonlinear trajectories need not produce a feasible trajectory. An
infeasible mean falls back to the best feasible sample. If no sample is feasible,
the simulation fails explicitly. This is a sampled finite-horizon check, not a
formal safety guarantee under arbitrary model error.

RLS identifies only `[drive_gain, c_v, c_pitch]` with physical bounds. It fits
momentum balances, uses only the longitudinal equation on the ground, and skips
contact transitions. Gravity and geometry cannot be learned into arbitrary
unphysical coefficients. Acceleration diagnostics compare the observed step with
a prediction **before** learning from that step. The logging/plots now use the
correct parameter names and include the true final integrated state.

## Reproducible checks

From this directory inside the container:

```bash
python3 -m unittest -v test_numerical_sim
python3 validate_closed_loop.py --output-dir /tmp/wheelie-validation
```

The unit tests cover zero-input ground support, liftoff, equilibrium feasibility,
independent mass-matrix and energy checks, NumPy/Torch RK4 parity, dissipative
landing, constructor inertia, RLS contact gating, and invalid inputs/rollouts.
The integration suite checks five seeds, RLS disabled, plant mismatch, and an
initial pitch/speed perturbation. It asserts tracking, torque, speed/rate behavior,
positive rear contact, and effective sample size. A separate actuation-contract test verifies that rollout inputs are independent of the simulated state. Generated validation
metrics are saved with each trajectory; it returns nonzero on a failed assertion.

Changes are restricted to numerical_sim. These tests establish behavior in this
reduced numerical model; COM/inertia and actuator calibration are still needed
before claiming correspondence with the physical car.

## Matplotlib warning in monster_truck_ros

The warning came from Matplotlib 3.11.1 in `/usr/local/lib/python3.12/dist-packages`
being combined with a legacy `mpl_toolkits` namespace from the system Matplotlib
3.6.3 installation. The container now uses the matching 3.11.1 toolkit. No warning
filter was added. Fresh Python processes import and render 3D plots successfully
with `-W error`. Restart an already-running Python process to reload its imports.

The following obsolete namespace files were renamed in the existing container
(their contents are preserved, not deleted):

- `/usr/lib/python3/dist-packages/matplotlib-3.6.3-nspkg.pth` to the same path plus `.disabled-wheelie-backup`.
- `/usr/lib/python3/dist-packages/mpl_toolkits/__init__.py` to the same path plus `.disabled-wheelie-backup`.

This lets Python's native namespace resolution use the current toolkit. The
repair is local to this container; rebuilding from its old image would require
fixing the mixed Matplotlib installation in that image as well. To undo it, rename
the two backups back to their original paths and start a fresh Python process.
