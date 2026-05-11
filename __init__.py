"""LAP eval-driver client (consumed by ``script/eval_policy.py``).

Exports ``get_model``, ``encode_obs``, ``eval``, ``reset_model`` so the
generic RoboTwin eval entry-point can drive the LAP cascade-VLA policy
served on the pod via ``policy/lap/scripts/serve_policy.py``.
"""

from .deploy_policy import *  # noqa: F401,F403
