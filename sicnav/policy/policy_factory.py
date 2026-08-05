from crowd_sim_plus.envs.policy.policy_factory import policy_factory
from sicnav.policy.dwa import DynamicWindowApproach
from sicnav.policy.campc import CollisionAvoidMPC
from sicnav.policy.mppi_policy import MPPIPolicy
from sicnav.policy.mppi_orca_policy import MPPIORCAPolicy

policy_factory['dwa'] = DynamicWindowApproach
policy_factory['campc'] = CollisionAvoidMPC
policy_factory['mppi'] = MPPIPolicy
policy_factory['mppi_orca'] = MPPIORCAPolicy