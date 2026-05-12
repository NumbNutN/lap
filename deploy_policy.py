"""LAP cascade-VLA deploy adapter for the generic RoboTwin eval driver.

The driver script ``script/eval_policy.py`` resolves a policy by
``importlib.import_module(policy_name)`` then calls ``get_model``,
``eval``, ``reset_model`` from that module. This file makes
``policy_name="lap"`` route to the LAP RoboTwin Stage 2 client (talks to
the server over WebSocket).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

from .lap_model import LAP


def encode_obs(observation):
    """Pull head + left + right RGB and the 14-d bimanual joint state.

    The order ``[head, right, left]`` is intentional — it mirrors the pi05
    encoder so ``LAP.update_observation_window(img_arr, state)`` can keep
    the same positional contract. LAP currently only uses head + left.
    """
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    train_config_name = usr_args["train_config_name"]
    model_name = usr_args["model_name"]
    checkpoint_id = usr_args["checkpoint_id"]
    pi0_step = usr_args["pi0_step"]

    server_host = usr_args.get("server_host", "localhost")
    server_port = usr_args.get("server_port", 8000)
    exec_horizon = usr_args.get("exec_horizon", pi0_step)
    infer_max_retries = usr_args.get("infer_max_retries", 3)
    infer_retry_backoff_s = usr_args.get("infer_retry_backoff_s", 1.0)
    action_smooth_alpha = float(usr_args.get("action_smooth_alpha", 1.0))

    model = LAP(
        train_config_name=train_config_name,
        model_name=model_name,
        checkpoint_id=checkpoint_id,
        pi0_step=pi0_step,
        server_host=server_host,
        server_port=server_port,
        infer_max_retries=infer_max_retries,
        infer_retry_backoff_s=infer_retry_backoff_s,
        action_smooth_alpha=action_smooth_alpha,
    )
    model.exec_horizon = min(exec_horizon, pi0_step)

    # CFG flags from deploy_policy.yml are accepted-and-ignored by LAP
    # (Stage 2 has no negative-prompt distillation). Print a warning if the
    # user expected them to take effect.
    if usr_args.get("guidance_scale", 0.0) > 0 and usr_args.get("neg_prompts"):
        print("[lap.deploy_policy] guidance_scale + neg_prompts ignored — "
              "Stage 2 LAP has no CFG distillation. Leave at 0.")

    verbose = usr_args.get("verbose", False)
    if isinstance(verbose, str):
        verbose = verbose.lower() in ("true", "1", "yes")
    if verbose:
        model.verbose = True
        task_name = usr_args.get("task_name", "unknown")
        model.verbose_dir = tempfile.mkdtemp(prefix=f"lap_verbose_{task_name}_")
        print(f"[lap.deploy_policy] Verbose mode: per-inference images at "
              f"{model.verbose_dir}")

    return model


def eval(TASK_ENV, model, observation):
    """Single eval-tick: feed observation, query model, execute returned action chunk."""
    if model.observation_window is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(observation)
    model.update_observation_window(input_rgb_arr, input_state)

    # Snapshot the step at which inference is fired — used by the per-frame
    # video overlay so the user can see exactly which frame triggered the
    # cascade emission for the chunk that follows.
    infer_step = int(getattr(TASK_ENV, "take_action_cnt", 0))
    # Snapshot which arm we packed into the wrist slot RIGHT NOW (i.e. the
    # one used to condition the inference we're about to fire). Compare
    # against the arm the cascade output asks for after the infer returns;
    # if they differ we downgrade to a single-step exec to give the next
    # tick a chance to re-pack the obs with the correct wrist (A1.b
    # "emergency 1-step" — see stage2_design_discussions_zh.md §1).
    wrist_arm_before_infer = getattr(model, "last_arm", None)
    # Training-data invariant (RoboTwin demo_clean): the first action of the
    # chunk equals the *current* qpos — i.e. ``actions[0] == state``. The sim's
    # ``take_action`` TOPP-interpolates from current_qpos → target_qpos, so
    # executing ``actions[0]`` is a no-op (the robot is already there). For
    # exec_horizon=1 this means we'd never move; for larger horizons it
    # wastes one step per chunk. Strip ``actions[0]`` so every executed
    # action is a *future* waypoint that actually moves the arm.
    chunk = model.get_action()
    if len(chunk) >= model.pi0_step + 1:
        actions = chunk[1:model.pi0_step + 1]   # drop chunk[0] (=current state)
    else:
        actions = chunk[1:] if len(chunk) > 1 else chunk

    # If the cascade text just selected a different arm than the wrist
    # image we conditioned on, the action chunk was sampled under stale
    # visual context. Execute only one step so the next inference can
    # re-pack the obs with the now-correct wrist before committing to a
    # full chunk. ``model.last_arm`` has already been updated by
    # ``get_action`` → ``_parse_arm_from_cascade``.
    wrist_arm_after_infer = getattr(model, "last_arm", None)
    arm_switched = (
        wrist_arm_before_infer is not None
        and wrist_arm_after_infer is not None
        and wrist_arm_after_infer != wrist_arm_before_infer
    )
    if arm_switched:
        print(
            f"[LAP] arm switch detected: wrist={wrist_arm_before_infer} → "
            f"{wrist_arm_after_infer}. Executing 1 step then re-querying."
        )
        exec_n = min(1, len(actions))
    else:
        exec_n = min(model.exec_horizon, len(actions))

    # Push a fresh overlay payload onto the env so the rawvideo writer can
    # annotate each frame ffmpeg ingests. The env safely no-ops if it lacks
    # the ``set_video_overlay`` helper (older RoboTwin checkout).
    set_overlay = getattr(TASK_ENV, "set_video_overlay", None)

    for i, action in enumerate(actions[:exec_n]):
        # Action smoothing (mitigation D from stage2_sim_eval_diagnosis.md):
        # EMA blend with the previously-executed action to dampen
        # chunk-boundary jumps that drive the IL policy off-trajectory.
        # ``model.smooth_action`` is a pass-through when
        # ``action_smooth_alpha >= 1.0``.
        action_to_exec = model.smooth_action(action)
        if set_overlay is not None:
            set_overlay({
                "infer_step": infer_step,
                "chunk_step": i + 1,
                "chunk_len": exec_n,
                "wrist": getattr(model, "last_wrist", "left"),
                "reasoning": getattr(model, "last_reasoning_text", "") or "",
            })
        TASK_ENV.take_action(action_to_exec)
        observation = TASK_ENV.get_obs()
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    if set_overlay is not None:
        # Clear the overlay between chunks so frames written outside the
        # take_action loop (e.g. on-success final frame) don't carry stale data.
        set_overlay(None)


def reset_model(model):
    model.reset_obsrvationwindows()
