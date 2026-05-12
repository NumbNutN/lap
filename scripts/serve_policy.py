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
    # ``flow``     → action expert flow sampling, returns ``actions`` only.
    #                 NOTE: this path does NOT generate the cascade context.
    #                 ``sample_actions`` reads ``tokenized_prompt`` verbatim,
    #                 so prefix == bare instruction. The action expert sees a
    #                 different prefix than during training (no cascade). This
    #                 is the source of the train/test gap; prefer ``cascade``.
    # ``ar``       → AR sampling of cascade tokens, returns decoded text only.
    # ``cascade``  → full cascade-VLA pipeline per request:
    #                AR generate cascade → append to prefix → flow sample
    #                actions conditioned on the *generated* cascade.
    #                Returns ``actions`` + ``reasoning_text``. Costs ~2× a
    #                pure-flow request.
    # ``dual``     → DEPRECATED alias for ``cascade``. The original
    #                ``DualModePolicy`` ran AR and flow *independently* (flow
    #                ignored the AR output), which silently kept the train/test
    #                gap; this alias now routes to the cascade pipeline.
    type: Literal["flow", "ar", "cascade", "dual"] = "flow"


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
        # Cascade pipeline: AR generate cascade → flow sample actions
        # conditioned on the *generated* cascade. Closes the train/test gap
        # that plain ``flow`` mode leaves open.
        type="cascade",
    ),
    EnvMode.PI05_DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid", type="flow"),
}


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    checkpoint = args.policy or DEFAULT_CHECKPOINT.get(args.env)

    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {args.env}")

    config = _config.get_config(checkpoint.config)
    # Disable BOTH stop_grad knobs for inference — they only matter for the
    # training-time gradient flow, and `stop_action_to_vlm_grad=True` triggers
    # a shape-mismatch bug inside `sample_tokens` (the `cross_to_expert0` mask
    # is sized to the un-padded prefix but the KV cache is padded for
    # max_decoding_steps; `probs * cross_to_expert0` then fails to broadcast).
    # Note: ``LAPConfig.__post_init__`` derives ``stop_action_to_vlm_grad``
    # from ``stop_grad_mode`` *every* time the dataclass is re-instantiated
    # (which ``dataclasses.replace`` does). So we must override both, in this
    # order — set ``stop_grad_mode="off"`` first, otherwise the post-init
    # will flip ``stop_action_to_vlm_grad`` back to True.
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            stop_grad_mode="off",
            stop_action_to_vlm_grad=False,
        ),
    )

    if checkpoint.type == "ar":
        return _policy_config.create_trained_policy_ar(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
    if checkpoint.type == "flow":
        return _policy_config.create_trained_policy(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
    if checkpoint.type in ("cascade", "dual"):
        # ``dual`` is kept as a back-compat alias; both route to the cascade
        # pipeline (the previous ``DualModePolicy`` was incorrect — it ran AR
        # and flow independently and never fed the cascade into the flow
        # prefix). The CascadePipelinePolicy reuses the same loaded params as
        # the flow policy, so this costs no extra GPU memory.
        from lap.policies.policy_adapter import CascadePipelinePolicy
        base = _policy_config.create_trained_policy(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
        return CascadePipelinePolicy(base)
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
