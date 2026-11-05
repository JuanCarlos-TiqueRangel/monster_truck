# RLS / MPPI wheelie audit

**Finding:** the supplied implementation contains several demonstrated defects, and the learned-model validation is inadequate for pitch control. The revised software passes numerical consistency tests, but the included candidate model still fails some real-data rollouts. It is not a hardware-validated controller.

**Scope:** five uploaded Python files and `dataset(2).zip`. The archive contains identification experiments, not closed-loop MPPI execution logs. The actual deployed `rls_learned_weights.npz`, quaternion/timestamp streams, ESC acknowledgments and videos were not supplied. Consequently, reconstructed coefficients and controller outputs below are reproducible offline examples, not a claim to have reconstructed the exact failing hardware experiment. No vehicle commands were sent.

Original sources are preserved verbatim under `audit/originals/`. All original line numbers below refer to those archived files. Source data were read without running the archive's cleanup script or changing its CSVs.

## 1. Most consequential implementation findings

### Future commands are used as training inputs

The active trainer contains `df['u'].shift(-2).fillna(0.0)` at line 720. For the ordinary row-aligned command convention, the input attached to state k is therefore u[k+2], not u[k] or a past command. At the nominal 0.04 second period, this is 0.08 seconds of lookahead. A delayed actuator normally requires past commands or an explicit actuator/history state, not future commands. A special logger timestamp offset could change how rows should be interpreted, but no evidence establishing such an offset was supplied.

More importantly, the original deployment does not reproduce this training relationship: the original MPPI applies the sampled command directly to the current simulated state. A fit based on one temporal relationship is being used as though it represented another.

The replacement fits a causal discrete target using the convention that command u[k] is issued at row k. The selected model uses u[k-3] for the transition k to k+1. This three-sample effective pipeline lag was selected by offline calibration; it is **not a measured physical ESC delay**, and actual command/state acquisition timestamps still need verification.

### Training plots are not frozen-model rollout validation

In original `rls.py`, lines 444–454 return predictions computed after updating the coefficients with the same target. Original `train_rls.py`, lines 760–795, collects these posterior predictions using measured state at every row, then integrates the collected acceleration series. Both the weights and the state inputs have already benefited from the measurement being compared against.

This can demonstrate online interpolation/tracking of observed data. It cannot demonstrate that a fixed model will recursively predict an unseen trajectory from its own states, which is the operation MPPI needs. The original function called RK4 on a precomputed acceleration time series is algebraically trapezoidal quadrature with interpolated accelerations; it is not an independent check of state-dependent RK4 rollouts.

Rebuilding the original trainer from zero on IDs 001–145 gave angular acceleration training RMSE 345.36 rad/s² for predictions before updates, versus 9.10 rad/s² for the posterior values the script displays. These are training diagnostics, not held-out prediction errors. Their large difference makes posterior plots especially misleading here. Linear acceleration values were 27.53 versus 7.00 m/s². See `audit/original_metrics.json`.

The new trainer starts from zero reproducibly, preserves covariance across the corpus, uses forgetting factor one for a fixed batch-equivalent fit, and reserves complete experiments for evaluation. The validator never calls RLS update and never substitutes measured states after initialization.

### The supposedly frozen deployment is adaptive

Original `wheelie_mppi_rls_node.py`, lines 67–88, loads coefficients under a frozen-model comment but initializes a fresh covariance. Lines 193–210 update those coefficients, and lines 231–237 call that update before every solve.

Thus, even an accurate saved model is immediately changed in deployment. It is updated from previous state/command and latest separately received, clipped acceleration, without timestamp matching. The three streams need not refer to the same physical interval.

Original forgetting factor 0.95 corresponds to a rough exponentially weighted sample count 1/(1-0.95)=20. At 25 Hz this is 0.8 seconds; the contribution half-life is about 0.54 seconds. Resetting covariance to 1000 for each training file further prevents the final fit from representing a single stable corpus-wide estimate. This is not an argument that online RLS is inherently unsuitable; it means the current regression pairs and evaluation must first be made consistent.

The revised node freezes the model and removes the acceleration subscription and online update path. Online adaptation should be reintroduced only with an explicitly validated transition estimator and separately evaluated adaptation behavior.

### MPPI reports a different trajectory from the one it returns

Original `mppi.py`, lines 398–417, forms and returns a weighted-average control sequence, while `cost` and `best_theta_N_deg` refer to the best individual sampled sequence. For nonlinear dynamics, the response to averaged inputs is not the averaged response, and a feasible sampled plan does not prove the average is feasible or useful.

An offline replay of the original controller with the reconstructed 001–145 coefficients, state zero, reference -55 degrees, K=5000, N=20, and random seed 2 gave:

| Quantity | Value |
|---|---:|
| Reported best-sample endpoint | -59.336961 degrees |
| Endpoint obtained by rolling out the actual returned mean plan | +127.825638 degrees |
| Returned first command | 0.02494622 |

Only the hard-coded CUDA device was changed to CPU for this replay. See `replay_original_mppi.py`, `audit/original_mean_vs_best.json`, and the corresponding figure. This result does not depend on observing a hardware run to establish the diagnostic mismatch.

The revised controller simulates the average plan explicitly, checks its feasibility, compares its cost with the best feasible sampled plan, and returns the lower-cost feasible plan. Main endpoint and cost fields describe the plan actually returned; best-sample fields remain separately named.

### Invalid predictions are converted into finite states

Original `mppi.py`, lines 282–305, clips velocity, pitch and rate and applies `nan_to_num`, including replacement of NaN by zero. This can turn failed model predictions into apparently usable states before the cost checks run. The original ground rule only sets positive angular acceleration to zero at nonnegative pitch; it does not stop an already positive angular velocity from crossing the floor.

The original parameter bounds are also not the bounds used by the rollout. Parameters say pitch -90 to 0 degrees, speed ±5 m/s and angular rate ±5 rad/s, whereas the actual step permits pitch ±360 degrees, speed ±15 and rate ±50 before clipping.

The replacement projects only unilateral ground contact. A trajectory that becomes nonfinite or leaves the configured operating domain keeps infinite cost permanently. Internal zeroing after invalidation protects arithmetic but cannot restore a finite cost. This is a numerical correctness fix, not proof that every physically feasible maneuver will be predicted accurately.

### Control search exceeds data support

All main identification samples have command u in [0, 0.30]; none contain negative commands. Original MPPI searches [-0.40, 0.40]. In particular, a predicted advantage for negative commands is unsupported by these data. The replacement searches [0, 0.30]. This is a data-support restriction, not a claim that 0.30 is a certified hardware-safe input.

The original stage function also computes effort, position and speed costs but omits them from the returned sum. The replacement implements every declared cost. Position and speed tracking weights are explicitly zero by default: the task is pitch regulation, not simultaneous stopping. Predicted speed still has a bound.

## 2. What the data support

There are 147 top-level identification files, totaling 3,192 rows. The `backup_dataset` directory contains 147 backup CSVs with 3,921 rows; these are not 147 additional independent experiments. The median top-level experiment lasts only 0.68003 seconds, shorter than the 0.8 second planning horizon.

There are 125 rows within five degrees of -55 degrees, but only nine of them also satisfy |omega|<0.5 rad/s. Even those nine rows do not establish equilibrium, because small instantaneous angular velocity does not establish small angular acceleration or sustained balance. The archive therefore contains much more transient maneuver evidence than sustained equilibrium evidence.

Examples requiring quarantine relative to the intended operating domain include file 011 with reported speed 44.607548 m/s, file 003 with -7.913379 m/s, and file 079 with angular rate 33.529386 rad/s and large pitch/rate kinematic inconsistency. These values do not identify the sensor failure mechanism, if any. No statement that all remaining data are clean is warranted.

Dataset angle degrees and radians agree numerically to about 2.1e-14 degrees. NumPy and Torch feature definitions also match in the original active code. An obvious coefficient-order error or simple degree/radian mismatch is therefore **not** the demonstrated primary failure.

The replacement quarantines those three files and screens transition time, finiteness, endpoint speed/rate, forward acceleration and angular kinematic consistency. Screening is explicit in `train_rls.py`, the model screening CSV and `audit/dataset_inventory.csv`. There are 2,666 accepted transitions before delay and fit/test exclusions; the final fit uses 1,735 transitions from 115 experiments. Twenty-nine complete experiments, IDs divisible by five, remain excluded from fitting.

## 3. The original reconstructed model gives misleading control directions

For the model reconstructed from the original script, at v=1 m/s, theta=-55 degrees and omega=0:

    predicted angular acceleration = 66.603828 + 180.181425*u

Its angular balance input is approximately -0.36965, outside the observed command range. This is a statement about the reconstructed model, not the true car. Under the current controller bounds it encourages the optimizer to explore an unobserved negative-command regime.

At stationary ground, the same model's raw angular prediction is approximately 33.78404 - 108.01513*u. The original contact rule suppresses positive angular acceleration, so none of u in [0, 0.30] produces immediate negative angular acceleration there. Its linear prediction also gives approximately +9.48219 m/s² at zero state and zero command. These are serious launch-model inconsistencies. They do not prove the model can never predict lift later: its speed-dependent coupling can subsequently invent different behavior.

The corrected candidate has a modeled joint trim at theta=-55 degrees of v=2.08704 m/s and u=0.258818, with both predicted acceleration residuals near floating-point zero. This passes an algebraic feasibility check, not a physical equilibrium measurement or stability proof. It must not be advertised as a guaranteed equilibrium for the real vehicle.

## 4. Frozen, free-running held-out evaluation

The revised model predicts forward velocity changes:

    target_v[k] = (v[k+1]-v[k])/dt[k]
    target_w[k] = (omega[k+1]-omega[k])/dt[k]
    target[k] approximately equals Phi(state[k], u[k-delay]) * coefficients

Velocities advance with the identified discrete update; position and pitch use trapezoidal kinematics. The original seven plus eleven nonlinear features are retained. Exported coefficients are in physical units after training-time feature scaling. This is full learned velocity-transition dynamics, not a nominal-model residual correction. The saved coefficients must not be interpreted as an instantaneous continuous vector field and integrated with RK4.

The same held-out windows were evaluated with an original-procedure model fit on the corresponding non-test split. The original procedure includes its own reset, forgetting, future-command alignment and clipping behavior. The revised pipeline also changes screening and integration, so this is a pipeline comparison, not an isolated ablation proving which individual modification caused each improvement.

| Prediction length | Experiments contributing | Overlapping windows | Original pitch RMSE | Revised pitch RMSE |
|---|---:|---:|---:|---:|
| 1 step, nominal 0.04 s | 29 | 470 | 3.68 degrees | 3.00 degrees |
| 5 steps, nominal 0.20 s | 29 | 295 | 17.26 degrees | 8.37 degrees |
| 10 steps, nominal 0.40 s | 19 | 145 | 28.90 degrees | **37.36 degrees** |
| 20 steps, nominal 0.80 s | 2 | 44 | 12.26 degrees | 6.72 degrees |

The ten-step aggregate gets worse. Its median window error improves from 21.69 to 6.78 degrees, but a small number of divergent rollouts dominate the aggregate. Specifically, file 045 starting at zero-based row 9 has 427.05 degrees of ten-step pitch RMSE and about -1284.52 degrees of endpoint error. Row 10 of the same file has 78.24 degrees RMSE. These predictions were retained and reported, not clipped away.

The 20-step result covers only two screened experiments; it does not establish good horizon-wide generalization. Its smaller error than the ten-step row comes from a different, narrower subset, not evidence that prediction improves with longer horizons. Windows overlap and are not independent trials. Four initial rows are omitted for common-start comparisons because earlier command history is unknown; startup from unlogged prehistory is therefore not validated by this table.

Unscreened twenty-step evaluation contains 103 windows from six experiments, including two nonfinite windows and three domain violations. The finite-window-only pitch RMSE is 12.99 degrees; those failures must be reported alongside it. Full metrics, per-window errors and raw prediction/truth arrays are under `audit/validation/`.

**Acceptance conclusion:** this candidate is not sufficiently validated for real-car pitch regulation. Rejecting an invalid predicted rollout can prevent numerical exploitation, but it cannot repair an inaccurate model or make its safe set match the real vehicle's feasible set.

## 5. Verified software behavior and unverified deployment behavior

All eleven numerical unit tests passed: feature parity, RLS/batch-ridge equivalence, pre/post-update semantics, timestep use, contact projection, permanent invalidation, nonfinite measurement handling, input delay, actual-returned-plan cost consistency, frozen weights, planar pitch geometry and command-bound floating-point tolerance. All replacement Python files compile.

In a six-second closed-loop simulation with the same revised learned model serving as both prediction model and vehicle, the controller had zero failed solves. Final-second pitch RMSE was 0.56349 degrees; final pitch was -55.16192 degrees. This establishes that the revised optimization/integration/history path can regulate **its own** model. The recorded-data failures show why that must not be equated with regulating the real car.

A CPU-only benchmark with PyTorch 2.10.0+cpu, one CPU thread, no compilation, K=5000 and N=20 gave median solve time 18.09 ms and p95 18.78 ms across thirty solves. This is not a GPU benchmark or an end-to-end ROS deadline guarantee. CUDA and ROS execution were unavailable in the audit environment.

The replacement node pairs odometry and twist by header timestamps, checks reception freshness, normalizes quaternions, unwraps a yaw-independent planar pitch and logs timing and model identity. It preserves the CSV velocity convention instead of silently changing frames. Because the CSVs omit the original quaternion stream and frame metadata, pitch/rate and world/body velocity consistency still require disabled-propulsion verification. The planar assumption requires small roll for body omega_y to equal pitch rate.

Reception freshness is not a guarantee that sensor acquisition itself was fresh. Header timestamps must actually identify acquisition time to resolve that uncertainty. The three-command delay history also assumes approximately regular control ticks; it does not solve a persistently overrunning loop. The node logs cycle interval and solve time, and does not impose a `solve_ms > 40` stop rule.

The configured step remains 0.04 seconds, meaning 25 Hz, not 50 Hz. Independent emergency-stop and receiver command-timeout mechanisms remain necessary; Python cleanup cannot ensure delivery after process termination or a communication failure.

## 6. Use and next acceptance steps

Keep the five replacement files together in a clean folder; do not combine the v2 code with old weights or legacy import directories. The v2 file has a distinct filename and explicit schema. The node rejects legacy model metadata and starts disarmed.

Follow the commands in `README.md` to reproduce training, frozen validation, numerical tests, same-model simulation and timing. A ROS dry run still publishes zero to `/drive`: it is not a passive monitor and must not compete with another controller's publisher.

Before live balancing, the next evidence needed is correctly timestamped state and published-command histories around launch, recovery and sustained near-reference operation. Actual deployed weights and an MPPI execution log would establish whether the demonstrated original pathologies are present in the failing run or were modified online. Log acquisition and publication times separately; a published command is not confirmation that the ESC applied it.

Additional training should include near-reference low-rate states with input variation sufficient to distinguish stabilizing and destabilizing responses, rather than only collecting more similar launch transients. This is a recommendation based on the observed coverage gap, not a guaranteed collection protocol. Prediction tests should cover launch, descent/contact, release of throttle and sustained balance, with entire trials excluded from fitting. Model structure may then need separate contact and airborne modes or an actuator/history state. The supplied evidence does not establish that merely adding more polynomial terms or reducing MPPI temperature will solve the remaining mismatch.

## 7. External technical context

These references support general interpretation; all numerical findings above come from the supplied code and data, not these papers.

- pandas, `DataFrame.shift`: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shift.html — interpretation of positive/negative row shifts.
- Williams et al., *Information Theoretic Model Predictive Control: Theory and Applications to Autonomous Driving* (2017): https://arxiv.org/abs/1707.02342 — sampling-based control and weighted sequence optimization. The revised constrained sampling/elitist selection is an engineering MPPI-style variant, not a new optimality or stability guarantee.
- Janner et al., *When to Trust Your Model: Model-Based Policy Optimization* (2019): https://arxiv.org/abs/1906.08253 — model bias and the importance of matching model usage to generalization. It does not establish that MBPO or a neural model is needed for this car.

The central result is narrower and testable: correct causal alignment, frozen recursive validation, consistent deployed weights and valid actual-returned plans are prerequisites that the original stack did not satisfy. The revisions repair those implementation paths while exposing, rather than hiding, the remaining model failure.
