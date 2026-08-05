import torch


class UnicycleDynamics:
    """
    Batched unicycle model matching crowd_sim_plus.envs.utils.agent_plus.Agent.compute_position:
        theta_new = theta + r          # r is a per-step heading increment, not an angular rate
        x_new = x + cos(theta_new) * v * dt
        y_new = y + sin(theta_new) * v * dt
    """

    nx = 3
    nu = 2

    def __init__(self, dt, device):
        self.dt = dt
        self.device = device

    def step(self, states, actions, t=None):
        if actions.ndim == 3:
            # Full-horizon call used by mppi_orca_core's compute_cost_once
            # path: states is the (K, nx) initial state, actions is the full
            # (K, T, nu) perturbed sequence. r is accumulated (not r*dt --
            # it's already a per-step increment) via cumsum so this is exact,
            # not an approximation of the sequential recursion below.
            v = actions[:, :, 0]
            r = actions[:, :, 1]
            theta_new = states[:, 2:3] + torch.cumsum(r, dim=1)  # (K, T)
            delta_x = torch.cos(theta_new) * v * self.dt
            delta_y = torch.sin(theta_new) * v * self.dt
            x_new = states[:, 0:1] + torch.cumsum(delta_x, dim=1)
            y_new = states[:, 1:2] + torch.cumsum(delta_y, dim=1)
            out = torch.stack([x_new, y_new, theta_new], dim=-1)  # (K, T, nx)
            return out, actions

        v = actions[:, 0]
        r = actions[:, 1]
        theta_new = states[:, 2] + r
        x_new = states[:, 0] + torch.cos(theta_new) * v * self.dt
        y_new = states[:, 1] + torch.sin(theta_new) * v * self.dt

        out = states.clone()
        out[:, 0] = x_new
        out[:, 1] = y_new
        out[:, 2] = theta_new
        return out, actions
