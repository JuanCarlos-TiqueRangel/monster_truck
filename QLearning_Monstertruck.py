# QLearning_Monstertruck_YPR.py
# Q-learning monster-truck flipping using roll, pitch, and throttle (pitch-only spin reward)
# Simplified shaping: integrated baseline into time penalty
# Requires: pip install mujoco glfw numpy matplotlib

import os, json, math, time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
from mujoco.glfw import glfw


def clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# ============================================================
# Utility: quaternion → roll, pitch
# ============================================================
def quat_to_rp(q):
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else (-1.0 if t2 < -1.0 else t2)
    pitch = math.asin(t2)
    return roll, pitch


# ============================================================
# Environment
# ============================================================
class MonsterTruckFlipEnvYPR:
    def __init__(self, xml_path="monstertruck.xml",
                 frame_skip=10, max_steps=2000,
                 render=False, realtime=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Cannot find {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_enabled = render
        self.realtime = realtime
        self.dt = self.model.opt.timestep

        # Expanded 9-level throttle
        self.actions = np.linspace(-1.0, 1.0, 9)
        self.last_throttle = 0.0
        self.step_count = 0
        self.hold_counter = 0
        self.hold_needed = 4

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")

        # Tuned reward weights (baseline integrated into time)
        self.R = dict(
            upright=20.0,    # reward for improving uprightness
            spin=1,       # encourage forward/back rotation (pitch)
            settle=0.5,     # penalty for moving near upright
            energy=0.01,    # energy cost
            time=0.035,       # stronger per-step penalty (includes baseline)
            success=800.0,  # terminal success
        )

        self.prev_upright = 0.0
        self.prev_roll, self.prev_pitch = 0.0, 0.0

        self._viewer_ready = False
        if render:
            self._init_viewer()

    # ---------------------- Rendering ----------------------
    def _init_viewer(self):
        if self._viewer_ready:
            return
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1000, 800, "MonsterTruck YPR Flip", None, None)
        glfw.make_context_current(self.window)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 3.0, 90, -25
        self._viewer_ready = True

    def _render(self):
        if not self.render_enabled:
            return
        if not self._viewer_ready:
            self._init_viewer()
        if glfw.window_should_close(self.window):
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False
            self.render_enabled = False
            return

        self.cam.lookat[:] = self.data.xpos[self.body_id]
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        w, h = glfw.get_framebuffer_size(self.window)
        mujoco.mjr_render(mujoco.MjrRect(0, 0, w, h), self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    # ---------------------- Core API ----------------------
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = np.array([0, 0, 0.6]) + 0.02 * np.random.randn(3)
        self.data.qvel[:] = 0.0
        self.data.qpos[3:7] = np.array([0, 1, 0, 0])  # upside-down start
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_throttle = 0.0
        self.hold_counter = 0
        self.prev_upright = -1.0  # start fully inverted
        self.prev_roll, self.prev_pitch = 0.0, 0.0
        return self._get_obs()

    def _get_rp(self):
        q = np.copy(self.data.qpos[3:7])
        return quat_to_rp(q)

    def _get_obs(self):
        roll, pitch = self._get_rp()
        u = math.cos(roll) * math.cos(pitch)
        return np.array([u, roll, pitch, self.last_throttle], dtype=np.float32)

    def step(self, a_idx):
        throttle = float(self.actions[a_idx])
        self.last_throttle = throttle
        done, success = False, False

        for _ in range(self.frame_skip):
            if self.model.nu >= 2:
                self.data.ctrl[0] = throttle
                self.data.ctrl[1] = throttle
            mujoco.mj_step(self.model, self.data)
            if self.realtime:
                time.sleep(self.dt)

        roll, pitch = self._get_rp()
        u = math.cos(roll) * math.cos(pitch)
        du = u - self.prev_upright
        self.prev_upright = u

        droll = roll - self.prev_roll
        dpitch = pitch - self.prev_pitch
        self.prev_roll, self.prev_pitch = roll, pitch

        upright = (u + 1.0) * 0.5
        ang_motion = abs(droll) + abs(dpitch)
        R = self.R

        # ===== Reward shaping (pitch-only, integrated time penalty) =====
        upright_reward = R["upright"] * max(du, 0.0)
        spin_reward = R["spin"] * abs(dpitch) * (1.0 - upright) if u > -0.8 else 0.0
        settle_penalty = -R["settle"] * ang_motion if (u > 0.9 and ang_motion > 0.02) else 0.0
        energy_penalty = -R["energy"] * (throttle ** 2)
        time_penalty = -R["time"]

        reward = (upright_reward + spin_reward + settle_penalty +
                  energy_penalty + time_penalty)

        # Success detection
        if u > 0.96 and ang_motion < 0.01:
            self.hold_counter += 1
            if self.hold_counter >= self.hold_needed:
                reward += R["success"]
                success, done = True, True
        else:
            self.hold_counter = 0

        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True

        reward = clip(reward, -10, 150)
        self._render()
        return self._get_obs(), reward, done, {"success": success}

    def close(self):
        if self._viewer_ready:
            glfw.destroy_window(self.window)
            glfw.terminate()
            self._viewer_ready = False


# ============================================================
# Q-Learning Agent
# ============================================================
class QAgent:
    def __init__(self,
                 obs_bins=(20, 15, 15, 9),
                 obs_ranges=((-1, 1), (-math.pi, math.pi),
                             (-math.pi / 2, math.pi / 2), (-1, 1)),
                 n_actions=9,
                 alpha=0.05, gamma=0.98,
                 eps_start=0.99, eps_end=0.01, total_episodes=5000):

        self.obs_bins = obs_bins
        self.obs_ranges = obs_ranges
        self.n_actions = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.total_episodes = total_episodes
        self.eps = eps_start
        self.eps_decay = (eps_end / eps_start) ** (1.0 / total_episodes)
        self.q = np.zeros(obs_bins + (n_actions,), dtype=np.float32)
        self.bin_edges = [np.linspace(lo, hi, nb - 1)
                          for (lo, hi), nb in zip(obs_ranges, obs_bins)]

    def _disc(self, obs):
        idxs = [int(np.digitize(o, edges)) for o, edges in zip(obs, self.bin_edges)]
        return tuple(idxs)

    def act(self, obs):
        s = self._disc(obs)
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q[s]))

    def act_greedy(self, obs):
        s = self._disc(obs)
        return int(np.argmax(self.q[s]))

    def learn(self, obs, a, r, next_obs, done):
        s, sn = self._disc(obs), self._disc(next_obs)
        best_next = 0.0 if done else np.max(self.q[sn])
        target = r + self.gamma * best_next
        self.q[s + (a,)] += self.alpha * (target - self.q[s + (a,)])

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, path="qtable_ypr.npy", meta="qmeta_ypr.json"):
        np.save(path, self.q)
        with open(meta, "w") as f:
            json.dump({"obs_bins": self.obs_bins, "obs_ranges": self.obs_ranges,
                       "n_actions": self.n_actions}, f, indent=2)
        print(f"💾 Saved Q-table to {path}")

    def load(self, path="qtable_ypr.npy", meta="qmeta_ypr.json"):
        if not (os.path.exists(path) and os.path.exists(meta)):
            return False
        try:
            self.q = np.load(path)
            with open(meta, "r") as f:
                m = json.load(f)
            self.obs_bins = tuple(m["obs_bins"])
            self.obs_ranges = [tuple(r) for r in m["obs_ranges"]]
            self.n_actions = m["n_actions"]
            print("✅ Loaded Q-table.")
            return True
        except Exception as e:
            print(f"⚠️ Failed to load Q-table: {e}")
            return False


# ============================================================
# Evaluation helper
# ============================================================
def evaluate_episode(env, agent, max_steps=1500, render=False):
    obs = env.reset()
    total = 0.0
    success_flag = 0
    for _ in range(max_steps):
        a = agent.act_greedy(obs)
        obs, r, done, info = env.step(a)
        total += r
        if done:
            if info.get("success", False):
                success_flag = 1
            break
    return total, success_flag


# ============================================================
# Training
# ============================================================
def train(episodes=3000, render_every=100, max_steps=1500, load_model=False, eval_every=100):
    agent = QAgent(total_episodes=episodes)
    if load_model and agent.load():
        print("🔁 Continuing training from saved Q-table")

    successes = 0
    rewards_list = []
    success_flags = []
    eval_ep_indices, eval_rewards = [], []

    env = MonsterTruckFlipEnvYPR(render=False, realtime=False,
                                 frame_skip=10, max_steps=max_steps)

    for ep in range(1, episodes + 1):
        do_render = (render_every and ep % render_every == 0)
        env.render_enabled = do_render
        env.realtime = do_render

        obs = env.reset()
        ep_ret = 0.0
        success_flag = 0

        for _ in range(max_steps):
            a = agent.act(obs)
            next_obs, r, done, info = env.step(a)
            agent.learn(obs, a, r, next_obs, done)
            obs = next_obs
            ep_ret += r
            if done:
                if info.get("success", False):
                    success_flag = 1
                    successes += 1
                break

        rewards_list.append(ep_ret)
        success_flags.append(success_flag)
        agent.decay_eps()

        if ep % 10 == 0:
            recent_rewards = rewards_list[-10:]
            recent_success = sum(success_flags[-10:])
            print(f"Ep {ep:4d} | eps {agent.eps:5.3f} | total success {successes}")
            print(f"   Last 10 rewards: {[round(r, 2) for r in recent_rewards]}")
            print(f"   Success (last 10): {recent_success}/10")

        if eval_every and (ep % eval_every == 0):
            er, es = evaluate_episode(env, agent, max_steps=max_steps, render=False)
            eval_ep_indices.append(ep)
            eval_rewards.append(er)
            print(f"   [Eval @ Ep {ep}] reward={er:.2f} | success={es}")

    env.close()
    agent.save()
    print(f"✅ Training complete! Successes: {successes}/{episodes}")

    np.save("training_rewards.npy", np.array(rewards_list, dtype=np.float32))
    np.save("eval_rewards.npy", np.array(eval_rewards, dtype=np.float32))
    np.save("eval_episodes.npy", np.array(eval_ep_indices, dtype=np.int32))

    if len(eval_ep_indices) > 0:
        plt.figure(figsize=(8, 4.8))
        plt.plot(eval_ep_indices, eval_rewards, marker='o', color='blue')
        plt.xlabel("Episode")
        plt.ylabel("Evaluation Reward")
        plt.title("Evaluation Reward vs Episode (Pitch-Only Spin, Integrated Time Penalty)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("eval_rewards.png", dpi=150)
        print("📈 Saved evaluation plot to eval_rewards.png")

    return rewards_list, (eval_ep_indices, eval_rewards)


if __name__ == "__main__":
    rewards, (eval_eps, eval_rewards) = train(
        episodes=4000, render_every=4000, max_steps=1500,
        load_model=False, eval_every=50
    )
    print("💾 Saved all episode rewards to training_rewards.npy")
