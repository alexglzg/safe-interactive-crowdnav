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
