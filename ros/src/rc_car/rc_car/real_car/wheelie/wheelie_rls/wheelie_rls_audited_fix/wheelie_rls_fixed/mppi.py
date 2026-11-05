#!/usr/bin/env python3
"""Bounded MPPI-style control of the v2 discrete RLS model.

Weights are frozen. Delay history contains ACTUALLY PUBLISHED commands,
oldest first. Invalid rollouts retain infinite cost; only ground contact is
projected. The weighted-mean plan is re-simulated and compared with the best feasible sample.
"""
import math
import multiprocessing as mp
import queue
import numpy as np
import torch
from rls import rls_step_torch


def _plot_worker(packets):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    plan, = ax.plot([], [], marker='o', label='Returned plan')
    reference, = ax.plot([], [], linestyle='--', label='Reference')
    ax.set(xlabel='Time ahead [s]', ylabel='Pitch [deg]', ylim=(-100, 10))
    ax.grid(True, alpha=0.3)
    ax.legend()
    while plt.fignum_exists(fig.number):
        try:
            packet = packets.get_nowait()
            if packet is None:
                break
            t, pitch, ref = packet
            plan.set_data(t, pitch)
            reference.set_data(t, np.full_like(t, ref))
            ax.set_xlim(0.0, float(t[-1]))
        except queue.Empty:
            pass
        plt.pause(0.05)
    plt.close(fig)


class MPPITorch:
    def __init__(self, p, cfg, rls_parameters, device=None, fast=True, integrator='forward', dtype=None, live_plot=False):
        if integrator != 'forward':
            raise ValueError('The v2 forward model must use integrator="forward", not RK4.')
        self.p, self.cfg = p, cfg
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dtype = dtype or (torch.float32 if fast else torch.float64)
        if self.device.type == 'cpu':
            torch.set_num_threads(cfg.cpu_threads)
        self.K, self.SIGMA, self.LAM = int(cfg.K), float(cfg.SIGMA), float(cfg.LAM)
        self.delay = int(cfg.input_delay_steps)
        if self.K < 16 or cfg.N <= self.delay or self.delay < 0 or cfg.dt <= 0 or self.SIGMA <= 0 or self.LAM <= 0 or cfg.noise_hold_steps < 1:
            raise ValueError('Invalid MPPI configuration.')
        if p.u_min >= p.u_max:
            raise ValueError('u_min must be smaller than u_max.')
        self.rls_parameters = torch.as_tensor(rls_parameters, dtype=self.dtype, device=self.device).clone()
        if self.rls_parameters.shape != (18,) or not bool(torch.isfinite(self.rls_parameters).all()):
            raise ValueError('Expected 18 finite RLS coefficients.')
        if not bool(torch.any(self.rls_parameters != 0.0)):
            raise ValueError('The RLS model is untrained (all coefficients are zero).')
        self.last_solution, self.last_prediction = None, None
        self._plot_process, self._plot_queue = None, None
        if live_plot:
            ctx = mp.get_context('spawn')
            self._plot_queue = ctx.Queue(maxsize=1)
            self._plot_process = ctx.Process(target=_plot_worker, args=(self._plot_queue,), daemon=True)
            self._plot_process.start()
        self._evaluate = torch.compile(self._rollout, fullgraph=True) if cfg.compile_rollouts else self._rollout

    def reset(self):
        self.last_solution, self.last_prediction = None, None

    def _step(self, state, u, dt):
        return rls_step_torch(state, u, dt, self.rls_parameters)

    def _stage_cost(self, state, u, previous, ref):
        c = self.cfg
        e = state-ref
        return c.q_x*e[:, 0].square() + c.q_v*e[:, 1].square() + c.q_theta*e[:, 2].square() + c.q_omega*e[:, 3].square() + c.r_u*u.square() + c.r_du*(u-previous).square()

    def _rollout(self, s0, controls, ref, u_previous, history):
        p, c = self.p, self.cfg
        state = s0.unsqueeze(0).expand(controls.shape[0], -1).clone()
        previous = torch.full_like(controls[:, 0], u_previous)
        cost = torch.zeros_like(previous)
        valid = torch.ones_like(previous, dtype=torch.bool)
        trajectory = [state]
        for k in range(c.N):
            command = controls[:, k]
            cost = cost + self._stage_cost(state, command, previous, ref)
            effective = history[k].expand_as(command) if k < self.delay else controls[:, k-self.delay]
            state = self._step(state, effective, c.dt)
            admissible = torch.isfinite(state).all(-1) & (state[:, 1] >= p.v_min) & (state[:, 1] <= p.v_max) & (state[:, 2] >= p.theta_min) & (state[:, 2] <= p.theta_max) & (state[:, 3] >= p.omega_min) & (state[:, 3] <= p.omega_max)
            valid = valid & admissible
            # Arithmetic protection only: once invalid, this sample can NEVER regain a finite cost.
            state = torch.where(valid[:, None], state, torch.zeros_like(state))
            trajectory.append(state)
            previous = command
        e = state-ref
        cost = cost + c.q_terminal_x*e[:, 0].square() + c.q_terminal_v*e[:, 1].square() + c.q_terminal_theta*e[:, 2].square() + c.q_terminal_omega*e[:, 3].square()
        valid = valid & torch.isfinite(cost)
        return cost.masked_fill(~valid, float('inf')), torch.stack(trajectory, dim=1), valid

    @torch.no_grad()
    def solve(self, state, ref, u_prev, input_history=None):
        p, c = self.p, self.cfg
        raw_state, raw_ref = np.asarray(state, float), np.asarray(ref, float)
        if raw_state.shape != (4,) or raw_ref.shape != (4,) or not np.isfinite(raw_state).all() or not np.isfinite(raw_ref).all() or not math.isfinite(u_prev):
            self.reset()
            return 0.0, {'success': False, 'reason': 'nonfinite_or_invalid_state'}
        if not p.v_min <= raw_state[1] <= p.v_max or not p.omega_min <= raw_state[3] <= p.omega_max or raw_state[2] < p.theta_min or raw_state[2] > math.radians(8.0):
            self.reset()
            return 0.0, {'success': False, 'reason': 'state_outside_model_domain'}
        if not p.theta_min <= raw_ref[2] <= p.theta_max or not p.u_min-1e-6 <= u_prev <= p.u_max+1e-6:
            raise ValueError('Reference pitch or previous applied command is outside configured bounds.')
        u_prev = float(np.clip(u_prev, p.u_min, p.u_max))
        history = np.asarray(input_history if input_history is not None else [], float)
        if history.shape != (self.delay,) or not np.isfinite(history).all() or np.any(history < p.u_min-1e-6) or np.any(history > p.u_max+1e-6):
            raise ValueError(f'Pass {self.delay} actually applied commands, oldest first, as input_history.')
        history = np.clip(history, p.u_min, p.u_max)
        s0 = torch.as_tensor(raw_state, dtype=self.dtype, device=self.device).clone()
        # Small positive ground-pitch measurements cannot represent penetration of the floor.
        s0[2] = s0[2].clamp(max=p.theta_max)
        ref = torch.as_tensor(raw_ref, dtype=self.dtype, device=self.device)
        history = torch.as_tensor(history, dtype=self.dtype, device=self.device)
        nominal = torch.full((c.N,), u_prev, dtype=self.dtype, device=self.device) if self.last_solution is None else torch.cat((self.last_solution[1:], self.last_solution[-1:]))
        hold = c.noise_hold_steps
        noise = self.SIGMA*torch.randn(self.K, math.ceil(c.N/hold), dtype=self.dtype, device=self.device)
        controls = (nominal[None] + noise.repeat_interleave(hold, dim=1)[:, :c.N]).clamp(p.u_min, p.u_max)
        controls[0] = nominal
        # Deterministic sustained actions ensure lift commands are actually considered.
        controls[1:12] = torch.linspace(p.u_min, p.u_max, 11, dtype=self.dtype, device=self.device)[:, None]
        costs, trajectories, valid = self._evaluate(s0, controls, ref, float(u_prev), history)
        if not bool(valid.any()):
            self.reset()
            return 0.0, {'success': False, 'reason': 'no_feasible_rollout', 'cost': float('inf')}
        best = torch.argmin(costs)
        weights = torch.softmax(-(costs-costs[best])/self.LAM, dim=0)
        mean_plan = (weights[:, None]*controls).sum(0).clamp(p.u_min, p.u_max)
        mean_cost, mean_trajectory, mean_valid = self._evaluate(s0, mean_plan[None], ref, float(u_prev), history)
        fallback = not bool(mean_valid[0]) or bool(mean_cost[0] > costs[best])
        chosen = controls[best].clone() if fallback else mean_plan
        chosen_trajectory = trajectories[best] if fallback else mean_trajectory[0]
        chosen_cost = costs[best] if fallback else mean_cost[0]
        self.last_solution = chosen.clone()
        self.last_prediction = chosen_trajectory.cpu().numpy()
        info = {'success': True, 'cost': float(chosen_cost.item()), 'best_cost': float(costs[best].item()), 'pitch_cost_now': float(c.q_theta*(s0[2]-ref[2]).square().item()), 'pred_theta_N_deg': math.degrees(float(chosen_trajectory[-1, 2].item())), 'pred_omega_N': float(chosen_trajectory[-1, 3].item()), 'pred_v_N': float(chosen_trajectory[-1, 1].item()), 'best_theta_N_deg': math.degrees(float(trajectories[best, -1, 2].item())), 'best_omega_N': float(trajectories[best, -1, 3].item()), 'effective_samples': float(weights.square().sum().reciprocal().item()), 'valid_fraction': float(valid.float().mean().item()), 'mean_plan_rejected': fallback}
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait((np.arange(c.N+1)*c.dt, np.rad2deg(self.last_prediction[:, 2]), math.degrees(raw_ref[2])))
            except queue.Full:
                pass
        return float(np.clip(chosen[0].item(), p.u_min, p.u_max)), info

    def close_plot(self):
        if self._plot_queue is not None:
            try:
                self._plot_queue.put_nowait(None)
            except queue.Full:
                pass
        if self._plot_process is not None:
            self._plot_process.join(timeout=0.3)
            if self._plot_process.is_alive():
                self._plot_process.terminate()
            self._plot_queue.close()
