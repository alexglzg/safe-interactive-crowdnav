import os

import numpy as np
import torch
import yaml

from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionRot

from sicnav.policy.mppi_core.mppi import MPPIPlanner
from sicnav.policy.mppi_dynamics import UnicycleDynamics
from sicnav.policy.mppi_objective import Objective

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'mppi.yaml')


class _MpcEnvStub:
    """
    crowd_sim_plus's render() reads policy.mpc_env.{nx_r,np_g,nx_hum,hum_model}
    whenever a policy exposes all_x_val, to locate each human's (x, y) row pair
    inside the flat x_val state-history array (offset = nx_r + np_g, stride =
    nx_hum) and to skip an ORCA-KKT-only goal-estimate branch. These values
    describe the layout MPPIPolicy itself writes into all_x_val below, not a
    real MPC state vector.
    """

    def __init__(self):
        self.nx_r = 2
        self.np_g = 0
        self.nx_hum = 2
        self.hum_model = 'cvmm'


class MPPIPolicy(Policy):
    """
    Vanilla MPPI robot policy: unicycle dynamics, humans treated as constant
    velocity (CVMM), disk collision cost, no interaction/risk modeling.
    """

    def __init__(self):
        super().__init__()
        self.name = 'mppi'
        self.trainable = False
        self.kinematics = 'unicycle'
        self.planner = None
        self.mpc_env = _MpcEnvStub()
        self.all_x_val = []
        self.all_x_goals = []

    def configure(self, config):
        with open(CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def _lazy_init(self):
        dt = self.time_step
        # Forced to CPU regardless of the harness's CUDA auto-detect: at this problem
        # size (K=2000, T=8) CPU already runs well inside the 0.25s control period,
        # and torch==1.13.1+cu117 doesn't support every GPU architecture (e.g. sm_120
        # Blackwell laptop GPUs raise/hang on real kernel launches despite
        # torch.cuda.is_available() reporting True).
        device = torch.device('cpu')
        mppi_cfg = dict(self.cfg['mppi'])
        mppi_cfg['device'] = str(device)

        self.dyn = UnicycleDynamics(dt, device)
        self.obj = Objective(dt, mppi_cfg['horizon'], self.cfg['weights'], device)
        self.planner = MPPIPlanner(
            mppi_cfg,
            nx=UnicycleDynamics.nx,
            dynamics=self.dyn.step,
            running_cost=self.obj.compute_running_cost,
        )

    def predict(self, state):
        if self.planner is None:
            self._lazy_init()

        self_state = state.self_state
        robot = np.array([self_state.px, self_state.py, self_state.theta], dtype=np.float32)

        human_states = state.human_states
        M = len(human_states)
        if M > 0:
            pos0 = np.array([[h.px, h.py] for h in human_states], dtype=np.float32)
            vel0 = np.array([[h.vx, h.vy] for h in human_states], dtype=np.float32)
            radii = np.array([h.radius for h in human_states], dtype=np.float32)
        else:
            pos0 = np.zeros((0, 2), dtype=np.float32)
            vel0 = np.zeros((0, 2), dtype=np.float32)
            radii = np.zeros((0,), dtype=np.float32)

        self.obj.set_goal(self_state.gx, self_state.gy)
        self.obj.set_humans(pos0, vel0, radii)
        self.obj.set_walls(state.static_obs)
        self.obj.set_robot_radius(self_state.radius)
        self.obj.reset_t()

        u = self.planner.command(robot).detach().cpu().numpy()

        self._record_horizon(M)

        return ActionRot(float(u[0]), float(u[1]))

    def _record_horizon(self, num_humans):
        # crowd_sim_plus's render() unconditionally reads per-human rows out of
        # all_x_val[...] once policy.all_x_val exists (see crowd_sim_plus.py,
        # ~line 1572-1714), regardless of whether all_opt_x/all_origin_x are set.
        # So the CVMM human rollouts (already computed for the cost) have to be
        # embedded in the same x_val row layout _MpcEnvStub describes, not just
        # the robot's own planned path.
        best_states, _ = self.planner.get_n_best_samples(1)
        robot_xy = best_states[0][:, :2].detach().cpu().numpy().T  # (2, T)
        horizon = robot_xy.shape[1]

        if num_humans > 0:
            hum_xy = self.obj.human_xy.detach().cpu().numpy()  # (M, T, 2)
            hum_rows = np.empty((2 * num_humans, horizon), dtype=np.float32)
            hum_rows[0::2, :] = hum_xy[:, :horizon, 0]
            hum_rows[1::2, :] = hum_xy[:, :horizon, 1]
            x_val = np.vstack([robot_xy, hum_rows])
        else:
            x_val = robot_xy

        self.all_x_val.append(x_val)

        goal_traj = np.tile(self.obj.goal.detach().cpu().numpy().reshape(2, 1), (1, horizon))
        self.all_x_goals.append(goal_traj)
