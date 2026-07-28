import torch


class Objective:
    """
    Running cost for CVMM-MPPI: goal attraction + disk collision cost against
    constant-velocity humans + disk collision cost against wall segments.

    mppi_torch's MPPIPlanner never passes a time index into running_cost (and its
    `_dynamics` wrapper drops the index it's given too), so the horizon step is
    tracked with an internal counter that must be reset before each command() call.
    """

    def __init__(self, dt, horizon, weights, device):
        self.dt = dt
        self.T = horizon
        self.device = device

        self.w_goal = weights['w_goal']
        self.w_human = weights['w_human']
        self.w_wall = weights['w_wall']
        self.buffer = weights['buffer']
        self.collision_cost = weights['collision_cost']

        self.goal = torch.zeros(2, device=device)
        self.rob_radius = 0.0
        self.M = 0
        self.S = 0
        self.t = 0

    def reset_t(self):
        self.t = 0

    def set_goal(self, gx, gy):
        self.goal = torch.tensor([gx, gy], dtype=torch.float32, device=self.device)

    def set_robot_radius(self, r):
        self.rob_radius = float(r)

    def set_humans(self, pos0, vel0, radii):
        self.M = pos0.shape[0]
        if self.M == 0:
            return
        pos0_t = torch.as_tensor(pos0, dtype=torch.float32, device=self.device)
        vel0_t = torch.as_tensor(vel0, dtype=torch.float32, device=self.device)
        t_idx = torch.arange(self.T, dtype=torch.float32, device=self.device).view(1, self.T, 1) * self.dt
        self.human_xy = pos0_t.unsqueeze(1) + t_idx * vel0_t.unsqueeze(1)  # (M, T, 2)
        self.human_radii = torch.as_tensor(radii, dtype=torch.float32, device=self.device)

    def set_walls(self, segments):
        self.S = len(segments)
        if self.S == 0:
            return
        p0 = [seg[0] for seg in segments]
        p1 = [seg[1] for seg in segments]
        self.wall_p0 = torch.tensor(p0, dtype=torch.float32, device=self.device)  # (S, 2)
        self.wall_p1 = torch.tensor(p1, dtype=torch.float32, device=self.device)  # (S, 2)

    def _point_seg_dist(self, pos):
        # pos: (K, 2) -> returns (K, S) distance to each wall segment
        seg = self.wall_p1 - self.wall_p0  # (S, 2)
        seg_len2 = (seg * seg).sum(dim=1).clamp(min=1e-8)  # (S,)
        w = pos.unsqueeze(1) - self.wall_p0.unsqueeze(0)  # (K, S, 2)
        tproj = (w * seg.unsqueeze(0)).sum(dim=2) / seg_len2.unsqueeze(0)  # (K, S)
        tproj = tproj.clamp(0.0, 1.0)
        closest = self.wall_p0.unsqueeze(0) + tproj.unsqueeze(2) * seg.unsqueeze(0)  # (K, S, 2)
        return torch.norm(pos.unsqueeze(1) - closest, dim=2)  # (K, S)

    def compute_running_cost(self, states):
        t = min(self.t, self.T - 1)
        self.t += 1

        pos = states[:, :2]  # (K, 2)
        cost = self.w_goal * torch.norm(pos - self.goal, dim=1)

        if self.M > 0:
            hum_pos = self.human_xy[:, t, :]  # (M, 2)
            dist = torch.norm(pos.unsqueeze(1) - hum_pos.unsqueeze(0), dim=2)  # (K, M)
            margin = (self.rob_radius + self.human_radii.unsqueeze(0) + self.buffer) - dist
            cost = cost + self.w_human * torch.relu(margin).pow(2).sum(dim=1)
            cost = cost + (margin > 0).any(dim=1).float() * self.collision_cost

        if self.S > 0:
            dist = self._point_seg_dist(pos)  # (K, S)
            margin = (self.rob_radius + self.buffer) - dist
            cost = cost + self.w_wall * torch.relu(margin).pow(2).sum(dim=1)
            cost = cost + (margin > 0).any(dim=1).float() * self.collision_cost

        return cost
