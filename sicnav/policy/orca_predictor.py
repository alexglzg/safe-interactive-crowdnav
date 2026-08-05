import numpy as np
import rvo2
import torch


class ORCAPredictor:
    """
    In-process replacement for the ROS service queried by
    interfaces/ros_jackalsimulator_interface.py::query_orca_predictions in the
    mppi_orca ROS package. Preserves that function's contract exactly:

        input : samples  (C, T, 2) float tensor -- robot xy rollouts
                                     (cluster representatives, not all K)
        output: predicted (C, T, n_humans, 4) -- [px, py, vx, vy] per human per
                                     predicted step, REACTING to each sample's
                                     own trajectory; same device/dtype as input.

    Humans' preferred velocities point at `human_goal_estimates` (a
    constant-velocity projection of current speed, matching SICNav-np's
    non-privileged goal estimator in campc.py -- NOT the simulator's true
    goals), so the comparison against SICNav-np isolates the ORCA-vs-CVMM
    interaction model rather than goal-estimation quality.

    Uncertainty over the interaction model is injected as a persistent
    per-(cluster representative, human) draw -- angular bias on the goal
    direction, a multiplicative speed-scale, and a reciprocity weight in
    [0, 1] blending each human's ORCA response with-robot vs. without-robot.
    Drawn ONCE per predict() call (i.e. once per control tick) and held fixed
    across the horizon: receding-horizon replanning at the env's control rate
    absorbs zero-mean per-step noise, so per-step noise would show nothing.
    """

    def __init__(self, dt, horizon, orca_params, device, uncertainty=None, seed=None):
        self.dt = dt
        self.horizon = horizon
        self.device = device

        self.neighbor_dist = orca_params.get('neighbor_dist', 10.0)
        self.max_neighbors = orca_params.get('max_neighbors', 10)
        self.time_horizon = orca_params.get('time_horizon', 2.0)
        self.time_horizon_obst = orca_params.get('time_horizon_obst', 0.5)
        self.robot_radius = orca_params.get('radius', 0.3)
        self.safety_space = orca_params.get('safety_space', 0.0)
        self.human_max_speed = orca_params.get('max_speed', 1.0)

        u = uncertainty or {}
        self.angle_bias_std = float(u.get('angle_bias_std', 0.0))
        self.speed_scale_std = float(u.get('speed_scale_std', 0.0))
        self.reciprocity_alpha = u.get('reciprocity_alpha', 1.0)

        self._rng = np.random.default_rng(seed)

        self.n_humans = 0
        self._human_pos0 = np.zeros((0, 2))
        self._human_vel0 = np.zeros((0, 2))
        self._human_radii = np.zeros((0,))
        self._human_goals = np.zeros((0, 2))
        self._static_obs = []

        self._sim = None
        self._sim_norobot = None

    def set_scene(self, human_states, static_obs, human_goal_estimates):
        n = len(human_states)
        self._human_pos0 = np.array([[h.px, h.py] for h in human_states], dtype=np.float64).reshape(n, 2)
        self._human_vel0 = np.array([[h.vx, h.vy] for h in human_states], dtype=np.float64).reshape(n, 2)
        self._human_radii = np.array([h.radius for h in human_states], dtype=np.float64).reshape(n)
        self._human_goals = np.asarray(human_goal_estimates, dtype=np.float64).reshape(n, 2)

        if static_obs != self._static_obs:
            self._sim = None
            self._sim_norobot = None
        self._static_obs = list(static_obs) if static_obs else []

        if n != self.n_humans:
            self._sim = None
            self._sim_norobot = None
        self.n_humans = n

    def __call__(self, samples):
        device, dtype = samples.device, samples.dtype
        n = self.n_humans
        C, T, _ = samples.shape

        if n == 0:
            return torch.zeros((C, T, 0, 4), device=device, dtype=dtype)

        samples_np = samples.detach().cpu().numpy().astype(np.float64)  # (C, T, 2)

        reciprocity = self.reciprocity_alpha
        if np.isscalar(reciprocity):
            reciprocity = np.full(n, float(reciprocity), dtype=np.float64)
        else:
            reciprocity = np.asarray(reciprocity, dtype=np.float64).reshape(n)
        need_norobot_sim = np.any(reciprocity < 1.0)

        angle_bias = (self._rng.normal(0.0, self.angle_bias_std, size=(C, n))
                      if self.angle_bias_std > 0 else np.zeros((C, n)))
        speed_scale = (1.0 + self._rng.normal(0.0, self.speed_scale_std, size=(C, n))
                       if self.speed_scale_std > 0 else np.ones((C, n)))
        speed_scale = np.clip(speed_scale, 0.1, None)

        sim = self._ensure_sim(with_robot=True)
        sim_norobot = self._ensure_sim(with_robot=False) if need_norobot_sim else None

        out = np.zeros((C, T, n, 4), dtype=np.float64)

        for c in range(C):
            hum_pos = self._human_pos0.copy()
            hum_vel = self._human_vel0.copy()
            prev_robot_pos = samples_np[c, 0]

            for t in range(T):
                robot_pos = samples_np[c, t]
                robot_vel = (robot_pos - prev_robot_pos) / self.dt if t > 0 else np.zeros(2)
                prev_robot_pos = robot_pos

                sim.setAgentPosition(0, tuple(robot_pos))
                sim.setAgentVelocity(0, tuple(robot_vel))

                pref_vels = np.zeros((n, 2))
                for i in range(n):
                    goal_vec = self._human_goals[i] - hum_pos[i]
                    dist = np.linalg.norm(goal_vec)
                    direction = goal_vec / dist if dist > 1e-6 else np.zeros(2)
                    theta = np.arctan2(direction[1], direction[0]) + angle_bias[c, i]
                    speed = self.human_max_speed * speed_scale[c, i]
                    pref_vels[i] = speed * np.array([np.cos(theta), np.sin(theta)])

                    sim.setAgentPosition(i + 1, tuple(hum_pos[i]))
                    sim.setAgentVelocity(i + 1, tuple(hum_vel[i]))
                    sim.setAgentPrefVelocity(i + 1, tuple(pref_vels[i]))

                    if sim_norobot is not None:
                        sim_norobot.setAgentPosition(i, tuple(hum_pos[i]))
                        sim_norobot.setAgentVelocity(i, tuple(hum_vel[i]))
                        sim_norobot.setAgentPrefVelocity(i, tuple(pref_vels[i]))

                sim.doStep()
                if sim_norobot is not None:
                    sim_norobot.doStep()

                for i in range(n):
                    v_new = np.array(sim.getAgentVelocity(i + 1))
                    if sim_norobot is not None and reciprocity[i] < 1.0:
                        v_without = np.array(sim_norobot.getAgentVelocity(i))
                        v_new = reciprocity[i] * v_new + (1.0 - reciprocity[i]) * v_without

                    hum_pos[i] = hum_pos[i] + v_new * self.dt
                    hum_vel[i] = v_new
                    out[c, t, i, 0:2] = hum_pos[i]
                    out[c, t, i, 2:4] = v_new

        return torch.as_tensor(out, device=device, dtype=dtype)

    def _ensure_sim(self, with_robot):
        cache_attr = '_sim' if with_robot else '_sim_norobot'
        sim = getattr(self, cache_attr)
        needed_agents = self.n_humans + 1 if with_robot else self.n_humans
        if sim is not None and sim.getNumAgents() != needed_agents:
            sim = None

        if sim is None:
            sim = rvo2.PyRVOSimulator(
                self.dt, self.neighbor_dist, self.max_neighbors,
                self.time_horizon, self.time_horizon_obst,
                self.robot_radius, self.human_max_speed,
            )
            for line in self._static_obs:
                sim.addObstacle(line)
            if self._static_obs:
                sim.processObstacles()

            if with_robot:
                sim.addAgent(
                    (0.0, 0.0), self.neighbor_dist, self.max_neighbors,
                    self.time_horizon, self.time_horizon_obst,
                    self.robot_radius + self.safety_space, self.human_max_speed,
                    (0.0, 0.0),
                )
            for i in range(self.n_humans):
                sim.addAgent(
                    (0.0, 0.0), self.neighbor_dist, self.max_neighbors,
                    self.time_horizon, self.time_horizon_obst,
                    self._human_radii[i] + self.safety_space, self.human_max_speed,
                    (0.0, 0.0),
                )
            setattr(self, cache_attr, sim)

        return sim
