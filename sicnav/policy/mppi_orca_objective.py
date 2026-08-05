import torch

from sicnav.policy.mppi_orca_core.prob_mppi.cost_functions import ObjectiveFunctionsClass
from sicnav.policy.mppi_orca_core.prob_mppi.dynamic_obstacles import DynamicObstacles


class MPPIORCAObjective(ObjectiveFunctionsClass):
    """
    Running cost for the ORCA-interaction-aware MPPI policy.

    This gym has nx=3 [x, y, theta] and a POINT goal, no global path, no
    lanes -- unlike the ROS objective this is adapted from (jackal_objective.py),
    which assumed a spline-following robot (nx=6, contouring/lane costs). So:
      KEPT unmodified from the vendored core: calculate_goal_cost,
        calculate_bound_dyn_cost (the risk term against per-cluster ORCA
        predictions -- this is the contribution).
      DROPPED: line_follow_cost, contouring_cost, lane_boundary_cost,
        calculate_tracking_velocity_cost (all assumed a global reference
        spline this gym doesn't have).
      ADDED: a wall cost (point-to-segment vs state.static_obs), matching the
        existing CVMM-MPPI objective (mppi_objective.py) so the two policies
        are cost-comparable apart from the human term.

    Called once per command() with the FULL horizon batch (compute_cost_once
    path in mppi_orca_core/planner/mppi.py is the only path that invokes
    human_predictor/kmeans clustering, so this objective only implements the
    state.ndim == 3 case): state is (K, T, 3).
    """

    def __init__(self, cfg, device):
        super().__init__(cfg, stat_obstacles=None, dyn_obstacles=None, goal=cfg["goal"])
        self.device = device

        w = cfg["weights"]
        self.w_goal = w["w_goal"]
        self.w_wall = w["w_wall"]
        self.wall_buffer = w["wall_buffer"]
        self.wall_collision_cost = w["wall_collision_cost"]

        self.hard_cp_constraint = cfg["obstacles"]["hard_cp_constraint"]
        self.orca_cov_scale = cfg["obstacles"].get("orca_cov_scale", 0.1)
        # DynamicObstacles.monte_carlo_sample_batch's isotropic fast path (the
        # only path we ever hit -- batch_shape is always (C, T, n_humans))
        # ignores the covariance object built from orca_cov_scale entirely
        # and uses cfg["obstacles"]["isotropic_sigma"] as the actual spread of
        # the collision-probability Gaussian. orca_cov_scale is effectively
        # inert here; isotropic_sigma (read directly by DynamicObstacles) is
        # the knob that matters.
        self.hard_collision_penalty = cfg["obstacles"].get("hard_collision_penalty", 1e6)
        self.coll_prob_weight = cfg["obstacles"].get("coll_prob_weight", 1000.0)

        self.rob_radius = 0.0
        self.S = 0
        self.wall_p0 = None
        self.wall_p1 = None

        # Stashed for horizon rendering: the policy adapter reads these back
        # after command() to plot the predicted-human trajectory of whichever
        # cluster the chosen (lowest-cost) rollout actually belongs to.
        self.last_predicted_humans = None
        self.last_human_cluster_labels = None
        self.last_human_cluster_center_idx = None
        self.last_frac_exceed_hard = None  # diagnostic, see calculate_bound_dyn_cost

        self.set_goal(*cfg["goal"])

    def set_goal(self, gx, gy):
        # Base class stores nav_goal as float16 (cost_functions.py); redo as
        # float32 since this gets overwritten every predict() call with the
        # env's actual goal, not just the yaml placeholder.
        self.nav_goal = torch.tensor([gx, gy], dtype=torch.float32, device=self.device)

    def set_robot_radius(self, r):
        self.rob_radius = float(r)

    def set_walls(self, segments):
        self.S = len(segments)
        if self.S == 0:
            self.wall_p0 = None
            self.wall_p1 = None
            return
        p0 = [seg[0] for seg in segments]
        p1 = [seg[1] for seg in segments]
        self.wall_p0 = torch.tensor(p0, dtype=torch.float32, device=self.device)  # (S, 2)
        self.wall_p1 = torch.tensor(p1, dtype=torch.float32, device=self.device)  # (S, 2)

    def ensure_dyn_obstacles(self, cfg, human_pos0, human_vel0):
        n_humans = human_pos0.shape[0]
        if self.dyn_obstacles is not None and self.dyn_obstacles.n_obstacles == n_humans:
            return
        x = torch.as_tensor(human_pos0[:, 0], device=self.device, dtype=torch.float32)
        y = torch.as_tensor(human_pos0[:, 1], device=self.device, dtype=torch.float32)
        yaw = torch.zeros(n_humans, device=self.device, dtype=torch.float32)
        cov = torch.eye(2, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(n_humans, 1, 1) * self.orca_cov_scale
        vx = torch.as_tensor(human_vel0[:, 0], device=self.device, dtype=torch.float32)
        vy = torch.as_tensor(human_vel0[:, 1], device=self.device, dtype=torch.float32)
        self.dyn_obstacles = DynamicObstacles(cfg, x, y, yaw, cov, vx, vy)

    def calculate_bound_dyn_cost(self, obstacle_modes: torch.Tensor, obstacle_covs: torch.Tensor, t=None):
        # Copied verbatim from mppi_orca_core/prob_mppi/jackal_objective.py --
        # that class isn't subclassable here because its __init__ hard-requires
        # a global-path spline object this gym doesn't have.
        if obstacle_modes.ndim == 2:
            coll_prob = self.calculate_multi_modal_dynamic_obstacle_cost(obstacle_modes[:, 0], obstacle_modes[:, 1], obstacle_covs, t)
        elif obstacle_modes.ndim == 3:
            coll_prob = self.calculate_multi_modal_dynamic_obstacle_cost(
                obstacle_modes[:, :, 0], obstacle_modes[:, :, 1], obstacle_covs)
        elif obstacle_modes.ndim == 4:
            coll_prob = self.calculate_multi_modal_dynamic_obstacle_cost(
                obstacle_modes[..., 0], obstacle_modes[..., 1], obstacle_covs)

        # Original ROS constants (20000 / *100) were 50x weaker than this
        # gym's own wall_collision_cost (1e6) for no principled reason --
        # made configurable and matched in magnitude to that hard wall
        # penalty so the planner doesn't treat human contact as far cheaper
        # than wall contact.
        # DIAGNOSTIC (temporary, per user request): fraction of (cluster,
        # timestep) pairs whose predicted collision probability exceeds the
        # hard threshold -- read back by the policy adapter after command().
        self.last_frac_exceed_hard = (coll_prob > self.hard_cp_constraint).float().mean().item()

        dyn_cost = torch.where(coll_prob > self.hard_cp_constraint, self.hard_collision_penalty, 0.)
        dyn_cost += coll_prob * self.coll_prob_weight
        return dyn_cost

    def _wall_cost(self):
        # self.x, self.y: (K, T) -> dist (K, T, S)
        pos = torch.stack((self.x, self.y), dim=-1)  # (K, T, 2)
        seg = self.wall_p1 - self.wall_p0  # (S, 2)
        seg_len2 = (seg * seg).sum(dim=1).clamp(min=1e-8)  # (S,)
        w = pos.unsqueeze(2) - self.wall_p0.view(1, 1, -1, 2)  # (K, T, S, 2)
        tproj = (w * seg.view(1, 1, -1, 2)).sum(dim=-1) / seg_len2.view(1, 1, -1)  # (K, T, S)
        tproj = tproj.clamp(0.0, 1.0)
        closest = self.wall_p0.view(1, 1, -1, 2) + tproj.unsqueeze(-1) * seg.view(1, 1, -1, 2)  # (K, T, S, 2)
        dist = torch.norm(pos.unsqueeze(2) - closest, dim=-1)  # (K, T, S)

        margin = (self.rob_radius + self.wall_buffer) - dist
        cost = self.w_wall * torch.relu(margin).pow(2).sum(dim=-1)
        cost = cost + (margin > 0).any(dim=-1).float() * self.wall_collision_cost
        return cost  # (K, T)

    def compute_running_cost(self, state: torch.Tensor, t: int = None, action: torch.Tensor = None,
                              predicted_humans: torch.Tensor = None, human_cluster_labels: torch.Tensor = None,
                              human_cluster_center_idx: torch.Tensor = None, **kwargs):
        self.x = state[:, :, 0]  # (K, T)
        self.y = state[:, :, 1]
        self.heading = state[:, :, 2]

        total_cost = self.calculate_goal_cost() * self.w_goal  # (K, T)

        if self.wall_p0 is not None:
            total_cost = total_cost + self._wall_cost()

        if predicted_humans is not None and human_cluster_labels is not None and predicted_humans.shape[-2] > 0:
            # Each cluster representative's ORCA prediction is treated as the
            # MEAN of a small Gaussian (orca_cov_scale, fixed -- same role as
            # CVMM's fixed covariance elsewhere in this codebase; the risk
            # signal here is cross-cluster disagreement in the MEAN, driven
            # by ORCAPredictor's per-cluster uncertainty draw, not variance).
            predicted_x = predicted_humans[..., 0]  # (C, T, n_humans)
            predicted_y = predicted_humans[..., 1]
            obstacle_modes = torch.stack((predicted_x, predicted_y), dim=-1)  # (C, T, n_humans, 2)
            orca_cov = torch.eye(2, device=state.device, dtype=state.dtype) * self.orca_cov_scale

            # Narrow self.x/self.y to the C representative robot trajectories
            # so the Monte Carlo integral's query points match the ORCA
            # prediction's (C, T, n_humans) batch size instead of the full K.
            full_x, full_y = self.x, self.y
            self.x = full_x[human_cluster_center_idx]  # (C, T)
            self.y = full_y[human_cluster_center_idx]  # (C, T)

            cluster_cost = self.calculate_bound_dyn_cost(obstacle_modes, orca_cov)  # (C, T)

            self.x, self.y = full_x, full_y

            # Broadcast per-cluster cost back out to all K samples.
            total_cost = total_cost + cluster_cost[human_cluster_labels]  # (K, T)

        self.last_predicted_humans = predicted_humans
        self.last_human_cluster_labels = human_cluster_labels
        self.last_human_cluster_center_idx = human_cluster_center_idx

        return total_cost
