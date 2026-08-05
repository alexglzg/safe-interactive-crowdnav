import torch
import functools
import numpy as np
import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
from typing import Optional, List, Callable, Union
from torch.distributions.multivariate_normal import MultivariateNormal
from scipy import signal
import scipy.interpolate as si
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
try:
    from mppi_utils import generate_gaussian_halton_samples, scale_ctrl, cost_to_go
except ModuleNotFoundError:
    from mppi_torch.mppi_utils import generate_gaussian_halton_samples, scale_ctrl, cost_to_go


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def bspline(c_arr, t_arr=None, n=100, degree=3):
    sample_device = c_arr.device
    sample_dtype = c_arr.dtype
    cv = c_arr.cpu().numpy()

    if(t_arr is None):
        t_arr = np.linspace(0, cv.shape[0], cv.shape[0])
    else:
        t_arr = t_arr.cpu().numpy()
    spl = si.splrep(t_arr, cv, k=degree, s=0.5)
    xx = np.linspace(0, cv.shape[0], n)
    samples = si.splev(xx, spl, ext=3)
    samples = torch.as_tensor(samples, device=sample_device, dtype=sample_dtype)
    return samples


# TODO: integrate with localplannerbench, using class inheritence
@dataclass
class MPPIConfig(object):
    """
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param mppi_mode: 'halton-spline' or 'simple' corresponds to the type of mppi.
        :param sampling_method: 'halton' or 'random', sampling strategy while using mode 'halton-spline'. In 'simple', random sampling is forced to 'random' 
        :param noise_sigma: variance per action
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean        
        :param device: pytorch device
        :param lambda_: inverse temperature, positive scalar where smaller values will allow more exploration
        :param update_lambda: flag for updating inv temperature
        :param update_cov: flag for updating covariance
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param rollout_var_discount: Discount cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost   
    """

    name: str = "mppi"
    num_samples: int = 100
    horizon: int = 30
    mppi_mode: str = 'halton-spline'
    sampling_method: str = "halton"
    noise_sigma: Optional[List[List[float]]] = None
    noise_mu: Optional[List[float]] = None
    device: str = "cuda:0"
    lambda_: float = 0.0
    update_lambda: bool = False
    update_cov: bool = False
    u_min: Optional[List[float]] = None
    u_max: Optional[List[float]] = None
    u_init: float = 0.0
    U_init: Optional[List[List[float]]] = None
    u_scale: float = 1
    u_per_command: int = 1
    rollout_var_discount: float = 0.95
    sample_null_action: bool = False
    noise_abs_cost: bool = False
    filter_u: bool = False
    use_priors: bool = False
    dt: float = 0.1
    dt_horizon: float = 0.2
    compute_cost_once: bool = True
    horizon_cutoff: int = 30
    nx: int = 6
    nu: int = 5
    n_clusters: int = 10


class MPPIPlanner(ABC):
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories thus it scales with the number of samples K. 

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning'  
    and 'STORM: An Integrated Framework for Fast Joint-Space Model-Predictive Control for Reactive Manipulation'

    Code based off and https://github.com/NVlabs/storm

    This mppi can run in two modes: 'simple' and a 'halton-spline':
        - simple:           random sampling at each MPPI iteration from normal distribution with simple mean update. To use this set 
                            mppi_mode = 'simple_mean'
        - halton-spline:    samples only at the start a halton-spline which is then shifted according to the current moments of the control distribution. 
                            Moments are updated using gradient. To use this set
                            mppi_mode = 'halton-spline', sample_mode = 'halton'
                            Alternatively, one can also sample random trajectories at each iteration using gradient mean update by setting
                            mppi_mode = 'halton-spline', sample_mode = 'random'
    """

    def __init__(self, cfg: Union[MPPIConfig, dict], nx: int, dynamics: Callable, running_cost: Callable, prior: Optional[Callable] = None, human_predictor=None):
        if type(cfg) is not MPPIConfig:
            cfg = MPPIConfig(**cfg)

        # Parameters for mppi and sampling method
        self.mppi_mode = cfg.mppi_mode
        self.sample_method = cfg.sampling_method

        # Utility vars
        self.K = cfg.num_samples        # N_SAMPLES 
        self.T = cfg.horizon            # TIMESTEPS
        self.filter_u = cfg.filter_u    # Flag for Sav-Gol filter
        self.lambda_ = cfg.lambda_
        self.tensor_args={'device':cfg.device, 'dtype':torch.float32}
        self.delta = None
        self.sample_null_action = cfg.sample_null_action
        self.u_per_command = cfg.u_per_command
        self.terminal_state_cost = None
        self.update_lambda = cfg.update_lambda
        self.update_cov = cfg.update_cov

        # Bound actions
        self.u_min = cfg.u_min
        self.u_max = cfg.u_max
        self.u_scale = cfg.u_scale        

        # Noise and input initialization
        self.noise_abs_cost = cfg.noise_abs_cost
        
        if not cfg.noise_sigma:
            cfg.noise_sigma = np.identity(int(nx/2)).tolist()
        assert all([len(cfg.noise_sigma[0]) == len(row) for row in cfg.noise_sigma])

        if not cfg.noise_mu:
            cfg.noise_mu = [0.0] * len(cfg.noise_sigma)
        if not cfg.U_init:
            cfg.U_init = [[0.0] * len(cfg.noise_mu)] * cfg.horizon

        # Make sure if any of the input limits are specified, both are specified
        if cfg.u_max and not cfg.u_min:
            cfg.u_min = -cfg.u_max
        if cfg.u_min and not cfg.u_max:
            cfg.u_max = -cfg.u_min
        self.cfg = cfg

        self.dynamics = dynamics
        self.running_cost = running_cost
        self.prior = prior
        self.human_predictor   = human_predictor   # Callable[[torch.Tensor], torch.Tensor] or None
        self.compute_cost_once = cfg.compute_cost_once

        # Convert lists in cfg to tensors and put them on device
        self.noise_sigma = torch.tensor(cfg.noise_sigma, device=cfg.device)
        self.noise_mu = torch.tensor(cfg.noise_mu, device=cfg.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma
        )
        self.u_init = torch.tensor(cfg.u_init, device=cfg.device)
        self.U = torch.tensor(cfg.U_init, device=cfg.device)
        # self.U = self.noise_dist.sample((self.T,))
        self.u_max = torch.tensor(cfg.u_max, device=cfg.device)
        self.u_min = torch.tensor(cfg.u_min, device=cfg.device)

        # Dimensions of state nx and control nu
        self.nx = nx
        self.nu = 1 if len(self.noise_sigma.shape) == 0 else self.noise_sigma.shape[0]
        
        # Moments and best trajectory
        self.mean_action = torch.zeros(self.nu, device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        self.best_traj = self.mean_action.clone()

        # Sampled results from last command
        self.state = None
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

        # handle 1D edge case
        if self.nu == 1:
            self.noise_mu = self.noise_mu.view(-1)
            self.noise_sigma = self.noise_sigma.view(-1, 1)
    
        # Generate a random integer for the seed
        gen_seed = np.random.randint(0, 1000)

        # Halton sampling 
        self.knot_scale = 1             # From mppi config storm is 4
        self.seed_val = gen_seed               # From mppi config storm
        # self.seed_val = 0               # From mppi config storm
        self.n_knots = self.T//self.knot_scale
        self.ndims = self.n_knots * self.nu
        self.degree = 1                 # From sample_lib storm is 2
        self.Z_seq = torch.zeros(1, self.T, self.nu, **self.tensor_args)
        self.cov_action = torch.diagonal(self.noise_sigma, 0)
        self.scale_tril = torch.sqrt(self.cov_action)
        self.squash_fn = 'clamp'
        self.step_size_mean = 1. # 0.98     # From storm

        # Discount
        self.gamma = cfg.rollout_var_discount 
        self.gamma_seq = torch.cumprod(torch.tensor([1.0] + [self.gamma] * (self.T - 1)),dim=0).reshape(1, self.T)
        self.gamma_seq = self.gamma_seq.to(**self.tensor_args)
        self.beta = 1 # param storm

        self.n_clusters = cfg.n_clusters # if hasattr(cfg, 'n_clusters') else 10


        # Filtering
        self.sgf_window = 9
        self.sgf_order = 2
        if (self.sgf_window % 2) == 0:
            self.sgf_window -=1       # Some versions of the sav-go filter require odd window size

        # Lambda update, for now the update of lambda is not performed
        self.eta_max = 0.1      # 10%
        self.eta_min = 0.01     # 1%
        self.lambda_mult = 0.1  # Update rate

        # covariance update  for now the update of lambda is not performed
        self.step_size_cov = 0.7
        self.kappa = 0.005

        self.eta_u_bound = 10
        self.eta_l_bound = 5
        self.beta_lm = 0.9
        self.beta_um = 1.2

    def _dynamics(self, state, u, t=None):
        return self.dynamics(state, u, t=t)

    def _running_cost(self, state, action=None, t=None, predicted_humans=None, human_cluster_labels=None, human_cluster_center_idx=None):
        return self.running_cost(state, action=action, t=t, predicted_humans=predicted_humans, human_cluster_labels=human_cluster_labels, human_cluster_center_idx=human_cluster_center_idx)

    def _exp_util(self, costs, actions):
        """
           Calculate weights using exponential utility given cost
        """
        traj_costs = cost_to_go(costs, self.gamma_seq)
        traj_costs = traj_costs[:,0]

        #control_costs = self._control_costs(actions)
        total_costs = traj_costs - torch.min(traj_costs) #+ self.beta * control_costs

        # Normalization of the weights
        exp_ = torch.exp((-1.0/self.beta) * total_costs)
        eta = torch.sum(exp_)       # tells how many significant samples we have, more or less
        w = 1/eta*exp_
        # print(self.beta)
        # beta update 
        if eta > self.eta_u_bound:
            self.beta = self.beta*self.beta_lm
        elif eta < self.eta_l_bound:
            self.beta = self.beta*self.beta_um
        
        #w = torch.softmax((-1.0/self.beta) * total_costs, dim=0)
        self.total_costs = total_costs
        return w

    def get_samples(self, sample_shape, **kwargs): 
        """
        Gets as input the desired number of samples and returns the actual samples. 

        Depending on the method, the samples can be Halton or Random. Halton samples a 
        number of knots, later interpolated with a spline
        """
        if(self.sample_method=='halton'):
            self.knot_points = generate_gaussian_halton_samples(
                sample_shape,               # Number of samples
                self.ndims,                 # n_knots * nu (knots per number of actions)
                use_ghalton=True,
                seed_val=self.seed_val,     # seed val is 0 
                device=self.tensor_args['device'],
                float_dtype=self.tensor_args['dtype'])
            
            # Sample splines from knot points:
            # iteratre over action dimension:
            knot_samples = self.knot_points.view(sample_shape, self.nu, self.n_knots) # n knots is T/knot_scale (30/4 = 7)
            self.samples = torch.zeros((sample_shape, self.T, self.nu), **self.tensor_args)
            for i in range(sample_shape):
                for j in range(self.nu):
                    self.samples[i,:,j] = bspline(knot_samples[i,j,:], n=self.T, degree=self.degree)

        elif(self.sample_method == 'random'):
            self.samples = self.noise_dist.sample((self.K, self.T))
        
        return self.samples

    def command(self, state, guidance=None):
        """
            Given a state, returns the best action sequence
        """

        torch.cuda.empty_cache()
        
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.tensor_args['dtype'], device=self.tensor_args['device'])

        if self.mppi_mode == 'simple':
            self.U = torch.roll(self.U, -1, dims=0)
            cost_total = self._compute_total_cost_batch_simple(guidance)

            beta = torch.min(cost_total)
            self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)

            eta = torch.sum(self.cost_total_non_zero)
            self.omega = (1. / eta) * self.cost_total_non_zero
            
            self.U += torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)

            action = self.U

        elif self.mppi_mode == 'halton-spline':
            # shift command 1 time step
            saved_action = self.mean_action[-1]
            self.mean_action = torch.roll(self.mean_action, -1, dims=0)
            self.mean_action[-1] = saved_action
            cost_total = self._compute_total_cost_batch_halton(guidance)
              
            action = torch.clone(self.mean_action)

        # Lambda update
        if self.update_lambda and self.mppi_mode == 'simple':
            if eta > self.eta_u_bound:
                self.lamdba_ = self.beta*self.beta_lm
            elif eta < self.eta_l_bound:
                self.lambda_ = self.beta*self.beta_um

        # Smoothing with Savitzky-Golay filter
        if self.filter_u:
            u_ = action.cpu().numpy()
            u_filtered = signal.savgol_filter(
                u_, 
                self.sgf_window, 
                self.sgf_order, 
                deriv=0, 
                delta=1.0, 
                axis=0, 
                mode='interp', 
                cval=0.0
                )
            if self.tensor_args['device'] == "cpu":
                action = torch.from_numpy(u_filtered).to('cpu')
            else:
                action = torch.from_numpy(u_filtered).to('cuda')
        
        # Reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]

        return action, self.states[:,:,:2], cost_total

    def _compute_rollout_costs(self, perturbed_actions, guidance=None):
        """
            Given a sequence of perturbed actions, forward simulates their effects and calculates costs for each rollout
        """
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_horizon = torch.zeros([K, T], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_samples = cost_total

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)

        states = []
        actions = []

        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]

            # Last rollout is a braking manover
            if self.sample_null_action:
                u[self.K - 1, :] = torch.zeros_like(u[self.K -1, :])
                self.perturbed_action[self.K - 1][t] = u[self.K -1, :]
            
            '''
            if self.prior and len(guidance) > 0:
                prior_samples = self.prior(state, t, torch.tensor(guidance[0].x, dtype=torch.float64, device=self.tensor_args['device']),
                                           torch.tensor(guidance[0].y, dtype=torch.float64, device=self.tensor_args['device']))
                n_priors = len(prior_samples)
                
                u[0:n_priors, :] = prior_samples
                self.perturbed_action[0:n_priors, t] = u[0:n_priors, :]
            '''
            
                
            state, u = self._dynamics(state, u, t)
            c = self._running_cost(state, u, t)

            # Update action if there were changes in fusion mppi due for instance to suction constraints
            self.perturbed_action[:,t] = u
            cost_samples += c
            cost_horizon[:, t] = c 

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states  = torch.stack(states, dim=-2)


        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c
        cost_total += cost_samples.mean(dim=0)
        
        if self.mppi_mode == 'halton-spline':
            self.noise = self._update_distribution(cost_horizon, actions)

        return cost_total, states, actions

    def _compute_rollout_costs_once(self, perturbed_actions, guidance=None):
        """
            Same as above but passes full trajectories to the cost function
        """
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total   = torch.zeros(K, device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_horizon = torch.zeros([K, T], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'])
        cost_samples = cost_total

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        if self.state.shape == (K, self.nx):
            state = self.state
        else:
            state = self.state.view(1, -1).repeat(K, 1)
        

        if self.sample_null_action:
            perturbed_actions[-1,:,:] = 0
            self.perturbed_action[-1,:,:] = 0
        
        
        states, actions =self._dynamics(state, perturbed_actions) 
        predicted_humans = None
        labels     = None
        center_idx = None

        if self.human_predictor is not None:
            xy = states[:, :, :2]  # (K, T, 2)
            labels, center_idx = self.kmeans_clustering(xy, k=self.n_clusters, iters=8)

            predicted_humans = self.human_predictor(xy[center_idx])  # (C, T, n_humans, 4)

        # Broadcast: every sample inherits its cluster's prediction.
        # labels is (K,) long tensor of cluster idx per sample -- fancy
        # indexing does the broadcast in one shot, no Python loop.
        # predicted_full = predicted_reps[labels]  # (K, T, n_humans, 4)


        # print("states dimension is: ", states.shape)
        # print("actions dimension is: ", actions.shape) 
        c = self._running_cost(states, actions, predicted_humans=predicted_humans, human_cluster_labels=labels, human_cluster_center_idx=center_idx)  # (K, T)

        cost_samples = torch.sum(c, dim=1)   # [400]
        cost_horizon = c   # [400, 20]

        # cost_horizon = c.T  # [400, 20]
        # cost_samples = c.sum(dim=0)  # [400]

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples += c

        ## We are taking the mean cost of the samples and adding it?!!!!!!!! Why add this?
        ## Removed just in case because I dont understand the purpose of this
        cost_total = cost_samples
        cost_total += cost_samples.mean(dim=0)

        if self.mppi_mode == 'halton-spline':
            self.noise = self._update_distribution(cost_horizon, actions)

        return cost_total, states, actions
    

    ## this is used for the orca ego conditioning to reduce the computational time of the mppi
    ##  by clustering the human trajectories and only using the cluster centers for the cost 
    ##  computation
    def kmeans_clustering(self, trajs: torch.Tensor, k: int = 10, iters: int = 8, seed: int = 0):
        """
        Fast k-means for trajectories (N,T,2) using Euclidean distance in flattened space.
        Vectorized assignment + centroid update via scatter_add (no Python loop over k).

        Returns:
          labels: (N,) long
          centers: (k,T,2) float
        """
        assert trajs.ndim == 3 and trajs.shape[-1] == 2, "Expected (N,T,2)"
        device = trajs.device
        N, T, D = trajs.shape
        P = T * D

        # Flatten to (N, P)
        X = trajs.reshape(N, P).contiguous()
        if X.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            X = X.float()

        # --- init: random points (fast). If you need better quality, use kmeans++ (slower).
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        init_idx = torch.randperm(N, generator=g, device=device)[:k]
        C = X[init_idx].clone()  # (k, P)

        # Pre-allocate buffers to avoid re-allocations
        labels = torch.empty(N, dtype=torch.long, device=device)
        ones = torch.ones(N, dtype=X.dtype, device=device)
        sums = torch.empty(k, P, dtype=X.dtype, device=device)
        counts = torch.empty(k, dtype=X.dtype, device=device)

        # Precompute x^2 term once (||x||^2)
        x2 = (X * X).sum(dim=1, keepdim=True)  # (N,1)

        for _ in range(iters):
            # Compute squared distances using: ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x c^T
            c2 = (C * C).sum(dim=1).unsqueeze(0)          # (1,k)
            xc = X @ C.t()                                 # (N,k)
            dist2 = x2 + c2 - 2.0 * xc                      # (N,k)

            new_labels = dist2.argmin(dim=1)
            # early stop (optional; small N so it’s cheap)
            if torch.equal(new_labels, labels):
                break
            labels.copy_(new_labels)

            # Centroid update: sums = sum_{i in cluster j} X[i]
            sums.zero_()
            sums.scatter_add_(0, labels[:, None].expand(-1, P), X)

            # counts = number of points in each cluster
            counts.zero_()
            counts.scatter_add_(0, labels, ones)

            # Avoid divide-by-zero: reinit empty clusters to random points
            empty = counts == 0
            if empty.any():
                ridx = torch.randint(0, N, (int(empty.sum().item()),), generator=g, device=device)
                C[empty] = X[ridx]
                counts = counts.clamp_min(1)

            C.copy_(sums / counts[:, None])

    #     centers = C.reshape(k, T, D)


        c2 = (C * C).sum(dim=1).unsqueeze(0)        # (1,k)
        xc = X @ C.t()                               # (N,k)
        dist2 = x2 + c2 - 2.0 * xc                   # (N,k)

        center_idx = torch.empty(k, dtype=torch.long, device=device)
        for j in range(k):
            mask = labels == j
            # choose closest trajectory inside the cluster
            idx_j = torch.where(mask)[0]
            if idx_j.numel() == 0:
                center_idx[j] = trajs.shape[0]-1
                continue
            d_j = dist2[idx_j, j]
            center_idx[j] = idx_j[d_j.argmin()]
        return labels, center_idx
    
 
    def _update_distribution(self, costs, actions):
        """
            Update moments using sample trajectories.
            So far only mean is updated, eventually one could also update the covariance
        """
        w = self._exp_util(costs, actions)
        
        # Compute also top n best actions to plot
        # top_values, top_idx = torch.topk(self.total_costs, 10)
        # self.top_values = top_values
        # self.top_idx = top_idx
        # self.top_trajs = torch.index_select(actions, 0, top_idx).squeeze(0)

        # Update best action
        best_idx = torch.argmax(w)
        self.best_idx = best_idx
        self.best_traj = torch.index_select(actions, 0, best_idx).squeeze(0)
       
        weighted_seq = w * actions.T

        sum_seq = torch.sum(weighted_seq.T, dim=0)
        new_mean = sum_seq

        # Gradient update for the mean
        self.mean_action = (1.0 - self.step_size_mean) * self.mean_action +\
            self.step_size_mean * new_mean 
       
        delta = actions - self.mean_action.unsqueeze(0)

        #Update Covariance
        if self.update_cov:
            #Diagonal covariance of size AxA
            weighted_delta = w * (delta ** 2).T
            # cov_update = torch.diag(torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0))
            cov_update = torch.mean(torch.sum(weighted_delta.T, dim=0), dim=0)
    
            self.cov_action = (1.0 - self.step_size_cov) * self.cov_action + self.step_size_cov * cov_update
            self.cov_action += self.kappa #* self.init_cov_action
            # self.cov_action[self.cov_action < 0.0005] = 0.0005
            self.scale_tril = torch.sqrt(self.cov_action)
        return delta

    def get_action_cost(self):
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv # Like original paper
        return action_cost

    def _compute_total_cost_batch_simple(self, guidance=None):
        """
            Samples random noise and computes perturbed action sequence at each iteration. Returns total cost
        """
        # Resample noise each time we take an action
        # self.noise = self.noise_dist.sample((self.K, self.T))
        if self.sample_method == 'random':
            self.delta = self.get_samples(self.K, base_seed=self.seed_val)
        elif self.delta == None and self.sample_method == 'halton':
            self.delta = self.get_samples(self.K, base_seed=self.seed_val)
            #add zero-noise seq so mean is always a part of samples

        # # Add zero-noise seq so mean is always a part of samples
        self.delta[-1,:,:] = self.Z_seq
        # Keeps the size but scales values
        # check why we scale the samples
        scaled_delta = torch.matmul(self.delta, torch.diag(self.scale_tril)).view(self.delta.shape[0], self.T, self.nu)
        # Broadcast own control to noise over samples; now it's K x T x nu
        self.perturbed_action = self.U + scaled_delta
        
        # Naively bound control
        self.perturbed_action = self._bound_action(self.perturbed_action)

        if self.compute_cost_once:
            self.cost_total, self.states, self.actions = self._compute_rollout_costs_once(self.perturbed_action, guidance)
        else:
            self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action, guidance)
        self.actions /= self.u_scale

        # Bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

        # action_cost = self.get_action_cost()

        # # Action perturbation cost
        # perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        # self.cost_total += perturbation_cost
        return self.cost_total

    def _compute_total_cost_batch_halton(self, guidance=None):
        """
            Samples Halton splines once and then shifts mean according to control distribution. If random sampling is selected 
            then samples random noise at each step. Mean of control distribution is updated using gradient
        """
        if self.sample_method == 'random':
            self.delta = self.get_samples(self.K, base_seed=self.seed_val)
        elif self.delta == None and self.sample_method == 'halton':
            self.delta = self.get_samples(self.K, base_seed=self.seed_val)
            #add zero-noise seq so mean is always a part of samples

        # Add zero-noise seq so mean is always a part of samples
        self.delta[-1,:,:] = self.Z_seq
        # Keeps the size but scales values
        scaled_delta = torch.matmul(self.delta, torch.diag(self.scale_tril)).view(self.delta.shape[0], self.T, self.nu)
        
        # First time mean is zero then it is updated in the distribution
        act_seq = self.mean_action + scaled_delta

        # Scales action within bounds. act_seq is the same as perturbed actions
        act_seq = scale_ctrl(act_seq, self.u_min, self.u_max, squash_fn=self.squash_fn)
        act_seq[self.nu, :, :] = self.best_traj
        
        self.perturbed_action = torch.clone(act_seq)
        
        if self.compute_cost_once:
            self.cost_total, self.states, self.actions = self._compute_rollout_costs_once(self.perturbed_action, guidance)
        else:
            self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action, guidance)


        self.actions /= self.u_scale

        action_cost = self.get_action_cost()

        # Action perturbation cost
        perturbation_cost = torch.sum(self.mean_action * action_cost, dim=(1, 2))
        # self.cost_total += perturbation_cost
        return self.cost_total

    
    def _bound_action(self, action):
        if self.u_max is not None:
            action = torch.max(torch.min(action, self.u_max), self.u_min)
        return action

    def get_n_best_samples(self, n):
        """
            Returns the n best state sequences based on their total cost
        """
        top_cost, top_idx = torch.topk(-self.total_costs, n)
        return self.states[top_idx], -top_cost