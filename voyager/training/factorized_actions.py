"""Hierarchical sampling over the frozen Civilization v2 action registry."""

from __future__ import annotations

from typing import Literal

import numpy as np

from voyager.sim.registries_v2 import (
    V2_FLAT_ACTION_COUNT,
    V2_MEANINGFUL_ACTIONS,
    V2_TARGET_COUNT,
    CivilizationV2Argument,
    CivilizationV2Verb,
    flatten_v2_action,
)

FactorizedInferenceMode = Literal["deterministic", "seeded_stochastic"]
FACTOR_VERB_COUNT = len(CivilizationV2Verb)
FACTOR_ARGUMENT_COUNT = len(CivilizationV2Argument)
FACTOR_TARGET_COUNT = V2_TARGET_COUNT

FLAT_VERBS = np.asarray(
    [int(verb) for verb, _argument, _target in V2_MEANINGFUL_ACTIONS],
    dtype=np.int32,
)
FLAT_ARGUMENTS = np.asarray(
    [int(argument) for _verb, argument, _target in V2_MEANINGFUL_ACTIONS],
    dtype=np.int32,
)
FLAT_TARGETS = np.asarray(
    [int(target) for _verb, _argument, target in V2_MEANINGFUL_ACTIONS],
    dtype=np.int32,
)


def action_components(flat_actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert flattened registry indices into verb, argument, and target arrays."""

    actions = np.asarray(flat_actions, dtype=np.int64)
    if np.any(actions < 0) or np.any(actions >= V2_FLAT_ACTION_COUNT):
        raise ValueError("Flattened Civilization actions are out of range.")
    return FLAT_VERBS[actions], FLAT_ARGUMENTS[actions], FLAT_TARGETS[actions]


def verb_masks(flat_masks: np.ndarray) -> np.ndarray:
    """Return legal verbs for each flattened legal-action mask."""

    masks = _validated_flat_masks(flat_masks)
    result = np.zeros((masks.shape[0], FACTOR_VERB_COUNT), dtype=np.bool_)
    for verb in range(FACTOR_VERB_COUNT):
        result[:, verb] = np.any(masks[:, FLAT_VERBS == verb], axis=1)
    return result


def argument_masks(flat_masks: np.ndarray, verbs: np.ndarray) -> np.ndarray:
    """Return legal arguments conditional on one selected verb per row."""

    masks = _validated_flat_masks(flat_masks)
    selected_verbs = _validated_components(verbs, masks.shape[0], FACTOR_VERB_COUNT, "verb")
    result = np.zeros((masks.shape[0], FACTOR_ARGUMENT_COUNT), dtype=np.bool_)
    for row, verb in enumerate(selected_verbs):
        legal = masks[row] & (FLAT_VERBS == verb)
        result[row, np.unique(FLAT_ARGUMENTS[legal])] = True
    _require_legal_component(result, "argument")
    return result


def target_masks(
    flat_masks: np.ndarray,
    verbs: np.ndarray,
    arguments: np.ndarray,
) -> np.ndarray:
    """Return legal targets conditional on selected verb and argument values."""

    masks = _validated_flat_masks(flat_masks)
    selected_verbs = _validated_components(verbs, masks.shape[0], FACTOR_VERB_COUNT, "verb")
    selected_arguments = _validated_components(
        arguments,
        masks.shape[0],
        FACTOR_ARGUMENT_COUNT,
        "argument",
    )
    result = np.zeros((masks.shape[0], FACTOR_TARGET_COUNT), dtype=np.bool_)
    for row, (verb, argument) in enumerate(zip(selected_verbs, selected_arguments, strict=True)):
        legal = masks[row] & (FLAT_VERBS == verb) & (FLAT_ARGUMENTS == argument)
        result[row, np.unique(FLAT_TARGETS[legal])] = True
    _require_legal_component(result, "target")
    return result


def flatten_components(
    verbs: np.ndarray,
    arguments: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Map component arrays back through the frozen public action registry."""

    selected_verbs = np.asarray(verbs, dtype=np.int64)
    selected_arguments = np.asarray(arguments, dtype=np.int64)
    selected_targets = np.asarray(targets, dtype=np.int64)
    if not (selected_verbs.shape == selected_arguments.shape == selected_targets.shape):
        raise ValueError("Factorized action component arrays must have identical shapes.")
    return np.asarray(
        [
            flatten_v2_action(int(verb), int(argument), int(target))
            for verb, argument, target in zip(
                selected_verbs.flat,
                selected_arguments.flat,
                selected_targets.flat,
                strict=True,
            )
        ],
        dtype=np.int32,
    ).reshape(selected_verbs.shape)


def choose_factorized_actions(
    *,
    verb_logits: np.ndarray,
    argument_logits: np.ndarray,
    target_logits: np.ndarray,
    flat_masks: np.ndarray,
    inference_mode: FactorizedInferenceMode,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose legal registry actions and return their exact joint log probabilities."""

    masks = _validated_flat_masks(flat_masks)
    verb_values = _validated_logits(verb_logits, masks.shape[0], FACTOR_VERB_COUNT, "verb")
    argument_values = _validated_logits(
        argument_logits,
        masks.shape[0],
        FACTOR_ARGUMENT_COUNT,
        "argument",
    )
    target_values = _validated_logits(
        target_logits,
        masks.shape[0],
        FACTOR_TARGET_COUNT,
        "target",
    )

    selected_verbs, verb_log_probs = _choose_component(
        verb_values,
        verb_masks(masks),
        inference_mode,
        rng,
    )
    selected_arguments, argument_log_probs = _choose_component(
        argument_values,
        argument_masks(masks, selected_verbs),
        inference_mode,
        rng,
    )
    selected_targets, target_log_probs = _choose_component(
        target_values,
        target_masks(masks, selected_verbs, selected_arguments),
        inference_mode,
        rng,
    )
    actions = flatten_components(
        selected_verbs,
        selected_arguments,
        selected_targets,
    )
    if not np.all(masks[np.arange(masks.shape[0]), actions]):
        raise AssertionError("Factorized sampler produced an illegal flattened action.")
    return actions, (verb_log_probs + argument_log_probs + target_log_probs).astype(np.float32)


def _choose_component(
    logits: np.ndarray,
    masks: np.ndarray,
    inference_mode: FactorizedInferenceMode,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    masked = np.where(masks, logits, -1e9)
    probabilities = np.stack([_probabilities(row) for row in masked], axis=0)
    if inference_mode == "deterministic":
        selected = np.argmax(masked, axis=1).astype(np.int32)
    elif inference_mode == "seeded_stochastic":
        selected = np.asarray(
            [rng.choice(masked.shape[1], p=row) for row in probabilities],
            dtype=np.int32,
        )
    else:
        raise ValueError(f"Unknown inference mode: {inference_mode!r}.")
    log_probs = np.log(probabilities[np.arange(masked.shape[0]), selected])
    return selected, log_probs.astype(np.float32)


def _probabilities(logits: np.ndarray) -> np.ndarray:
    stable = logits - float(np.max(logits))
    probabilities = np.exp(stable)
    return probabilities / np.sum(probabilities)


def _validated_flat_masks(flat_masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(flat_masks, dtype=np.bool_)
    if masks.ndim != 2 or masks.shape[1] != V2_FLAT_ACTION_COUNT:
        raise ValueError(
            f"Flattened masks must have shape (batch, {V2_FLAT_ACTION_COUNT}), got {masks.shape}."
        )
    if np.any(~np.any(masks, axis=1)):
        raise ValueError("Every flattened mask row must allow at least one action.")
    return masks


def _validated_components(
    values: np.ndarray,
    batch_size: int,
    component_count: int,
    name: str,
) -> np.ndarray:
    components = np.asarray(values, dtype=np.int64)
    if components.shape != (batch_size,):
        raise ValueError(f"{name} values must have shape ({batch_size},).")
    if np.any(components < 0) or np.any(components >= component_count):
        raise ValueError(f"{name} values are out of range.")
    return components


def _validated_logits(
    logits: np.ndarray,
    batch_size: int,
    component_count: int,
    name: str,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    expected = (batch_size, component_count)
    if values.shape != expected:
        raise ValueError(f"{name} logits must have shape {expected}, got {values.shape}.")
    return values


def _require_legal_component(masks: np.ndarray, name: str) -> None:
    if np.any(~np.any(masks, axis=1)):
        raise ValueError(f"Selected prefix has no legal {name}.")
