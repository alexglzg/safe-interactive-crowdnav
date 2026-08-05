import os

import numpy as np
import torch
import yaml

from crowd_sim_plus.envs.policy.policy import Policy
from crowd_sim_plus.envs.utils.action import ActionRot

from sicnav.policy.mppi_orca_core.planner.mppi import MPPIPlanner
from sicnav.policy.mppi_dynamics import UnicycleDynamics
from sicnav.policy.mppi_orca_objective import MPPIORCAObjective
from sicnav.policy.orca_predictor import ORCAPredictor

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'mppi_orca.yaml')

# Matches campc.py's non-privileged human goal estimator exactly (SICNav-np):
# gx = px + vx * GOAL_EXTRAPOLATE_T, so the ORCA-vs-CVMM comparison isolates
# the interaction model rather than goal-estimation quality.
GOAL_EXTRAPOLATE_T = 2.0


class _MpcEnvStub:
    """
    crowd_sim_plus's render() reads policy.mpc_env.{nx_r,np_g,nx_hum,hum_model}
    whenever a policy exposes all_x_val, to locate each human's (x, y) row pair
    inside the flat x_val state-history array (offset = nx_r + np_g, stride =
    nx_hum) and to skip an ORCA-KKT-only goal-estimate branch. These values
    describe the layout MPPIORCAPolicy itself writes into all_x_val below, not
    a real MPC state vector. (Same stub as mppi_policy.py -- duplicated rather
    than imported to keep this policy independent of the CVMM one.)
    """

    def __init__(self):
        self.nx_r = 2
        self.np_g = 0
        self.nx_hum = 2
        self.hum_model = 'cvmm'


class MPPIORCAPolicy(Policy):
    """
    Risk- and interaction-aware MPPI robot policy: unicycle dynamics, humans
    modeled with ORCA (reacting to each candidate robot rollout) instead of
    CVMM, with an uncertainty dial over the interaction model itself (goal
    direction/speed bias, reciprocity) so the collision-risk cost integrates
    over disagreement in the predicted reaction, not just fixed sensor noise.
    """

    def __init__(self):
        super().__init__()
        self.name = 'mppi_orca'
        self.trainable = False
        self.kinematics = 'unicycle'
        self.priviledged_info = False
        self.planner = None
        self.mpc_env = _MpcEnvStub()
        self._tick = 0
        self.all_x_val = []
        self.all_x_goals = []

    def configure(self, config):
        with open(CONFIG_PATH) as f:
            self.cfg = yaml.safe_load(f)

    def _lazy_init(self):
        dt = self.time_step
        # Forced to CPU for the same reasons as mppi_policy.py: this problem
        # size runs well inside the control period on CPU, and torch==1.13.1
        # +cu117 doesn't support every GPU architecture.
        device = torch.device('cpu')

        mppi_cfg = dict(self.cfg['mppi'])
        mppi_cfg['device'] = str(device)

        full_cfg = {
            'device': str(device),
            'goal': [0.0, 0.0],
            'start': [0.0, 0.0],
            'mppi': {'num_samples': mppi_cfg['num_samples'], 'horizon': mppi_cfg['horizon']},
            'obstacles': dict(self.cfg['obstacles']),
            'weights': dict(self.cfg['weights']),
        }
        self.full_cfg = full_cfg

        self.dyn = UnicycleDynamics(dt, device)
        self.obj = MPPIORCAObjective(full_cfg, device)
        self.predictor = ORCAPredictor(
            dt=dt, horizon=mppi_cfg['horizon'], orca_params=dict(self.cfg['orca']),
            device=device, uncertainty=dict(self.cfg['uncertainty']),
        )
        self.planner = MPPIPlanner(
            mppi_cfg,
            nx=UnicycleDynamics.nx,
            dynamics=self.dyn.step,
            running_cost=self.obj.compute_running_cost,
            human_predictor=self.predictor,
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
            goal_estimates = pos0 + vel0 * GOAL_EXTRAPOLATE_T
        else:
            pos0 = np.zeros((0, 2), dtype=np.float32)
            vel0 = np.zeros((0, 2), dtype=np.float32)
            goal_estimates = np.zeros((0, 2), dtype=np.float32)

        self.obj.set_goal(self_state.gx, self_state.gy)
        self.obj.set_walls(state.static_obs)
        self.obj.set_robot_radius(self_state.radius)

        self.predictor.set_scene(human_states, state.static_obs, goal_estimates)
        if M > 0:
            self.obj.ensure_dyn_obstacles(self.full_cfg, pos0, vel0)

        u, states_xy, cost_total = self.planner.command(robot)
        u = u.detach().cpu().numpy()

        self._log_diagnostics()

        self._record_horizon(M, states_xy, cost_total)

        return ActionRot(float(u[0]), float(u[1]))

    def _log_diagnostics(self):
        # Temporary diagnostic (per user request while debugging frozen-robot
        # behavior): per tick, fraction of (cluster, timestep) predictions
        # exceeding hard_cp_constraint, and whether the highest-weighted
        # sample is the null (braking) action -- index K-1 under
        # sample_null_action, per _compute_rollout_costs_once's
        # perturbed_actions[-1,:,:] = 0. Remove once calibration resumes.
        self._tick += 1
        omega = getattr(self.planner, 'omega', None)
        frac_exceed = self.obj.last_frac_exceed_hard
        if omega is None:
            print(f"[mppi_orca diag] tick={self._tick} frac_exceed_hard={frac_exceed} omega=<unset>")
            return
        K = omega.shape[0]
        winning_idx = int(torch.argmax(omega).item())
        winning_weight = float(omega[winning_idx].item())
        null_weight = float(omega[K - 1].item())
        print(f"[mppi_orca diag] tick={self._tick} frac_exceed_hard={frac_exceed} "
              f"winning_idx={winning_idx}/{K} is_null={winning_idx == K - 1} "
              f"winning_weight={winning_weight:.4f} null_action_weight={null_weight:.4f}")

    def _record_horizon(self, num_humans, states_xy, cost_total):
        # crowd_sim_plus's render() unconditionally reads per-human rows out of
        # all_x_val[...] once policy.all_x_val exists (see crowd_sim_plus.py,
        # ~line 1572-1714), regardless of whether all_opt_x/all_origin_x are
        # set. So the ORCA-conditioned human rollout that actually drove the
        # chosen action has to be embedded in the same x_val row layout
        # _MpcEnvStub describes, not just the robot's own planned path.
        best_idx = int(torch.argmin(cost_total).item())
        robot_xy = states_xy[best_idx].detach().cpu().numpy().T  # (2, T)
        horizon = robot_xy.shape[1]

        hum_xy = None
        if num_humans > 0 and self.obj.last_predicted_humans is not None \
                and self.obj.last_predicted_humans.shape[-2] > 0:
            cluster = int(self.obj.last_human_cluster_labels[best_idx].item())
            # (T, n_humans, 2) predicted positions for the cluster the chosen
            # rollout actually belongs to -- this is what the robot believed
            # would happen when it picked this action.
            hum_xy = self.obj.last_predicted_humans[cluster, :, :, :2].detach().cpu().numpy()

        if hum_xy is not None:
            hum_rows = np.empty((2 * num_humans, horizon), dtype=np.float32)
            T_pred = min(horizon, hum_xy.shape[0])
            hum_rows[0::2, :T_pred] = hum_xy[:T_pred, :, 0].T
            hum_rows[1::2, :T_pred] = hum_xy[:T_pred, :, 1].T
            if T_pred < horizon:
                hum_rows[:, T_pred:] = hum_rows[:, T_pred - 1:T_pred]
            x_val = np.vstack([robot_xy, hum_rows])
        else:
            x_val = robot_xy

        self.all_x_val.append(x_val)

        goal_traj = np.tile(self.obj.nav_goal.detach().cpu().numpy().reshape(2, 1), (1, horizon))
        self.all_x_goals.append(goal_traj)
