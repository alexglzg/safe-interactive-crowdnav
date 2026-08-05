#!/usr/bin/env bash
set -e
python run_tro.py --policy mppi_orca \
  --env_config sicnav/configs/env.config \
  --policy_config sicnav/configs/policy.config \
  --hallway_bottleneck --num_humans 3
