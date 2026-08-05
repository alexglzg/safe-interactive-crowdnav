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

        eps = 1e-3
        for c in range(C):
            hum_pos = self._human_pos0.copy()
            hum_vel = self._human_vel0.copy()

            for t in range(T):
                robot_pos = samples_np[c, t]

                # Position-only, matching the reference (_rollout_one_sample
                # in orca_mppi_cond_node.py): agent 0's velocity is
                # deliberately left alone (~0, since it never gets a
                # setAgentPrefVelocity call either) -- only its POSITION
                # drives the humans' avoidance geometry there. Setting a
                # real finite-differenced velocity here (as an earlier
                # version of this file did) makes every human react more
                # defensively to every rollout than the reference ever did,
                # and was the primary cause of a frozen-robot regression.
                sim.setAgentPosition(0, tuple(robot_pos))

                pref_vels = np.zeros((n, 2))
                for i in range(n):
                    # Reference formula (orca_mppi_cond_node.py
                    # _rollout_one_sample): raw goal vector, capped at
                    # vmax-eps if beyond one step's reach, else used AS-IS
                    # (slows down approaching the goal instead of
                    # overshooting/oscillating -- this gym's CV-extrapolated
                    # goal is often close, so the clamp matters). angle_bias
                    # rotates the vector (magnitude-preserving, so it doesn't
                    # disturb the distance-based branch) and speed_scale
                    # scales vmax; both reduce to the reference exactly at
                    # (angle_bias=0, speed_scale=1).
                    raw_vel = self._human_goals[i] - hum_pos[i]
                    dist = np.linalg.norm(raw_vel)
                    theta = np.arctan2(raw_vel[1], raw_vel[0]) + angle_bias[c, i]
                    rot_vel = dist * np.array([np.cos(theta), np.sin(theta)])
                    vmax = self.human_max_speed * speed_scale[c, i]
                    pref_vels[i] = (rot_vel / dist * (vmax - eps)) if dist > (vmax - eps) else rot_vel

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

            # +0.01 fixed margin matches the reference (_init_worker in
            # orca_mppi_cond_node.py, and orca.py/orca_plus.py elsewhere in
            # this codebase) exactly -- radius + 0.01 + safety_space.
            if with_robot:
                sim.addAgent(
                    (0.0, 0.0), self.neighbor_dist, self.max_neighbors,
                    self.time_horizon, self.time_horizon_obst,
                    self.robot_radius + 0.01 + self.safety_space, self.human_max_speed,
                    (0.0, 0.0),
                )
            for i in range(self.n_humans):
                sim.addAgent(
                    (0.0, 0.0), self.neighbor_dist, self.max_neighbors,
                    self.time_horizon, self.time_horizon_obst,
                    self._human_radii[i] + 0.01 + self.safety_space, self.human_max_speed,
                    (0.0, 0.0),
                )
            setattr(self, cache_attr, sim)

        return sim
