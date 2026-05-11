import dataclasses
import enum
import logging
import socket
from typing import Literal

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
import tyro

import tensorflow as tf
# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch / JAX)
tf.config.set_visible_devices([], 'GPU')

import lap.policies.policy_config_adapter as _policy_config
from lap.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    # LAP-3B, generating actions via action expert
    LAP = "lap"
    # LAP-3B, generating language actions via autogressive sampling
    LAP_AR = "lap_ar"
    # LAP-3B fine-tuned on LIBERO
    LAP_LIBERO = "lap_libero"
    # Cascade-VLA Stage 2: LAP-3B (PaliGemma 2B + 300M action expert) finetuned
    # on the RoboTwin task suite (pick_place / arrange_blocks / stack_blocks).
    # 14-DoF bimanual action head, head_camera as base + active arm wrist.
    LAP_ROBOTWIN = "lap_robotwin"
    # Open-sourced baseline model from Physical Intelligence
    PI05_DROID = "pi05_droid"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    # ``flow``  → action expert flow sampling, returns ``actions`` only.
    # ``ar``    → AR sampling of cascade tokens, returns decoded text.
    # ``dual``  → run BOTH per inference; response carries ``actions`` + ``reasoning_text``.
    #             Used for sim eval so video overlay can show cascade output.
    type: Literal["flow", "ar", "dual"] = "flow"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LAP
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | None = None


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.LAP: Checkpoint(config="lap", dir="checkpoints/lap", type="flow"),
    EnvMode.LAP_AR: Checkpoint(config="lap", dir="checkpoints/lap", type="ar"),
    EnvMode.LAP_LIBERO: Checkpoint(config="lap_libero", dir="checkpoints/lap_libero", type="flow"),
    EnvMode.LAP_ROBOTWIN: Checkpoint(
        config="lap_robotwin_finetune",
        dir="checkpoints/lap_robotwin_finetune/lap_robotwin_run0/30000",
        # Dual mode: server emits actions (flow) + cascade reasoning_text (AR)
        # per request, so sim eval can overlay the [plan]/[stage]/[action] text
        # on rollout videos. ~2× inference latency.
        type="dual",
    ),
    EnvMode.PI05_DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid", type="flow"),
}


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    checkpoint = args.policy or DEFAULT_CHECKPOINT.get(args.env)

    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {args.env}")

    config = _config.get_config(checkpoint.config)
    # Always disable stop_action_to_vlm_grad for inference — this flag is only
    # meaningful during training.
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, stop_action_to_vlm_grad=False))

    if checkpoint.type == "ar":
        return _policy_config.create_trained_policy_ar(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
    if checkpoint.type == "flow":
        return _policy_config.create_trained_policy(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
    if checkpoint.type == "dual":
        # Build the flow policy once, then wrap its underlying model in an
        # ARPolicy for the cascade-text path. We deliberately reuse the
        # same loaded params so this costs no extra GPU memory.
        from lap.policies.policy_adapter import ARPolicy, DualModePolicy
        flow_policy = _policy_config.create_trained_policy(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
        ar_policy = ARPolicy(flow_policy)
        return DualModePolicy(flow=flow_policy, ar=ar_policy)
    raise NotImplementedError


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
