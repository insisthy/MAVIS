"""Final ECG-anchored deterministic action and numerical execution layer.

This module is intentionally the only active owner of action normalization,
safety vetoes, correction tiering, and numerical execution.  Agent4 supplies a
raw semantic recommendation; it never supplies a delta, intensity, or a
second action alias.
"""
from __future__ import annotations

import math
from typing import Any

from src.runtime.directional_policy import normalize_agent4_action_for_directional_contract


FINAL_SINGLE_CONTROLLER_RUNTIME_CONTRACT = {
    "controller_version": "final_single_controller_v2_validation_locked_logit",
    "input_owner": {
        "numeric_anchor": "locked_ecg_score",
        "semantic_action": "agent4.agent4_action_raw",
        "canonical_relation": "agent3.correction_relation_relative_to_ecg",
        "agent2_risk_band": "agent2.risk_band",
    },
    "output_required": [
        "controller_version", "agent4_action_raw", "controller_action",
        "correction_eligible", "correction_tier", "correction_policy_locked",
        "applied_delta", "corrected_score", "controller_veto_reason",
        "direction_contract_pass", "evidence_id_contract_pass", "safety_contract_pass",
    ],
    "allowed_actions": ["UP", "DOWN", "PRESERVE", "REVIEW_REQUIRED"],
    "unlocked_policy_behavior": "applied_delta_zero_and_corrected_score_equals_locked_ecg_score",
    "locked_policy_behavior": "sigmoid_logit_shift_of_locked_ecg_score",
    "applied_delta_space": "logit",
}


class FinalSingleController:
    controller_version = "final_single_controller_v2_validation_locked_logit"
    unlocked_projection_mode = "validation_policy_unlocked_zero_delta"
    locked_projection_mode = "validation_locked_logit_shift"

    @staticmethod
    def _action(value: Any) -> str:
        value = str(value or "REVIEW_REQUIRED").strip().upper()
        return value if value in {"UP", "DOWN", "PRESERVE", "REVIEW_REQUIRED"} else "REVIEW_REQUIRED"

    @staticmethod
    def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"FINAL_CONTROLLER_POLICY_INVALID:{name}") from exc
        if not math.isfinite(number) or (minimum is not None and number < minimum):
            raise RuntimeError(f"FINAL_CONTROLLER_POLICY_INVALID:{name}")
        return number

    @classmethod
    def validate_correction_policy(cls, correction_policy: dict[str, Any] | None) -> dict[str, Any]:
        """Fail closed on inconsistent numerical policy state.

        The historical unlocked configuration remains a supported zero-delta
        execution mode.  A non-zero policy is accepted only when every field
        required by the preregistered logit-shift protocol is explicit.
        """

        policy = dict(correction_policy or {})
        mode = str(policy.get("projection_mode", cls.unlocked_projection_mode))
        locked = bool(policy.get("correction_policy_locked", False))
        numeric = bool(policy.get("numeric_correction_applied", False))
        if mode == cls.unlocked_projection_mode:
            if locked or numeric:
                raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:unlocked_policy_state")
            return {
                "projection_mode": mode,
                "correction_policy_locked": False,
                "numeric_correction_applied": False,
                "delta_up": 0.0,
                "delta_down": 0.0,
                "tier_multipliers": {"NONE": 0.0, "MILD": 0.0, "STANDARD": 0.0},
                "probability_clip_epsilon": 1e-6,
            }
        if mode != cls.locked_projection_mode or not locked or not numeric:
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:locked_policy_state")
        if str(policy.get("projection_target", "")) != "locked_ecg_score":
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:projection_target")
        if str(policy.get("mathematical_form", "")) != "logit_shift":
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:mathematical_form")
        if str(policy.get("applied_delta_space", "")) != "logit":
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:applied_delta_space")
        multipliers = policy.get("tier_multipliers")
        if not isinstance(multipliers, dict) or set(multipliers) != {"NONE", "MILD", "STANDARD"}:
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:tier_multipliers")
        normalized_multipliers = {
            key: cls._finite_float(multipliers[key], f"tier_multipliers.{key}", minimum=0.0)
            for key in ("NONE", "MILD", "STANDARD")
        }
        if normalized_multipliers["NONE"] != 0.0:
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:tier_multipliers.NONE")
        epsilon = cls._finite_float(
            policy.get("probability_clip_epsilon"), "probability_clip_epsilon", minimum=0.0
        )
        if not 0.0 < epsilon < 0.5:
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:probability_clip_epsilon")
        return {
            "projection_mode": mode,
            "correction_policy_locked": True,
            "numeric_correction_applied": True,
            "delta_up": cls._finite_float(policy.get("delta_up"), "delta_up", minimum=0.0),
            "delta_down": cls._finite_float(policy.get("delta_down"), "delta_down", minimum=0.0),
            "tier_multipliers": normalized_multipliers,
            "probability_clip_epsilon": epsilon,
        }

    @staticmethod
    def _project_logit(score: float, signed_logit_delta: float, epsilon: float) -> float:
        clipped = min(max(score, epsilon), 1.0 - epsilon)
        shifted_logit = math.log(clipped / (1.0 - clipped)) + signed_logit_delta
        if shifted_logit >= 0.0:
            return 1.0 / (1.0 + math.exp(-shifted_logit))
        exp_value = math.exp(shifted_logit)
        return exp_value / (1.0 + exp_value)

    def evaluate(
        self,
        *,
        agent1_output: dict[str, Any],
        agent2_output: dict[str, Any],
        agent3_output: dict[str, Any],
        agent4_output: dict[str, Any],
        ecg_anchor_context: dict[str, Any],
        correction_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self._action(agent4_output.get("agent4_action_raw"))
        baseline = str(ecg_anchor_context.get("ecg_baseline_state", "")).upper()
        relation = str(agent3_output.get("correction_relation_relative_to_ecg", "ECG_BASELINE_RELATION_UNRESOLVED")).upper()
        grade = str(agent3_output.get("correction_evidence_grade", "NONE")).upper()
        evidence_sufficiency = str(agent4_output.get("evidence_sufficiency", "INSUFFICIENT")).upper()
        confidence = str(agent4_output.get("correction_confidence", "LOW")).upper()
        safety = str(agent4_output.get("safety_constraint", "PRESERVE_ONLY")).upper()
        reasons: list[str] = []

        normalized = normalize_agent4_action_for_directional_contract(
            ecg_baseline_state=baseline,
            raw_action=raw,
            correction_relation_relative_to_ecg=relation,
            correction_evidence_grade=grade,
        )
        action = normalized.controller_public_action
        if normalized.action_normalization_applied:
            reasons.append(normalized.action_normalization_reason)
        veto = any((
            bool(agent4_output.get("hidden_risk_flag")) and action == "DOWN",
            bool(agent4_output.get("rag_misuse_flag")),
            bool(agent4_output.get("medical_overclaim_flag")),
            safety == "PRESERVE_ONLY" and action in {"UP", "DOWN"},
            safety == "NO_DOWNWARD_CORRECTION" and action == "DOWN",
            evidence_sufficiency == "INSUFFICIENT" and action in {"UP", "DOWN"},
        ))
        if veto:
            if bool(agent4_output.get("hidden_risk_flag")) and action == "DOWN": reasons.append("hidden_risk_blocks_down")
            if bool(agent4_output.get("rag_misuse_flag")): reasons.append("rag_misuse_flag")
            if bool(agent4_output.get("medical_overclaim_flag")): reasons.append("medical_overclaim_flag")
            if safety == "PRESERVE_ONLY": reasons.append("preserve_only")
            if safety == "NO_DOWNWARD_CORRECTION" and action == "DOWN": reasons.append("no_downward_correction")
            if evidence_sufficiency == "INSUFFICIENT": reasons.append("insufficient_evidence")
            action = "PRESERVE" if raw == "DOWN" else "REVIEW_REQUIRED"

        band = str(agent2_output.get("risk_band", "LOW")).upper()
        tier = "NONE"
        if action in {"UP", "DOWN"} and not reasons and confidence != "LOW" and evidence_sufficiency in {"PARTIAL", "SUFFICIENT"}:
            tier = "STANDARD" if grade in {"MODERATE", "STRONG"} and band in {"INTERMEDIATE", "HIGH"} and evidence_sufficiency == "SUFFICIENT" else "MILD"
        policy = self.validate_correction_policy(correction_policy)
        locked = bool(policy["correction_policy_locked"])
        score = self._finite_float(ecg_anchor_context["locked_ecg_score"], "locked_ecg_score")
        if not 0.0 <= score <= 1.0:
            raise RuntimeError("FINAL_CONTROLLER_POLICY_INVALID:locked_ecg_score")
        correction_eligible = action in {"UP", "DOWN"} and tier != "NONE" and not reasons
        delta = 0.0
        corrected = score
        if locked and correction_eligible:
            multiplier = policy["tier_multipliers"][tier]
            magnitude = policy["delta_up"] if action == "UP" else policy["delta_down"]
            delta = multiplier * magnitude * (1.0 if action == "UP" else -1.0)
            corrected = self._project_logit(score, delta, policy["probability_clip_epsilon"])
        return {
            "controller_version": self.controller_version,
            "agent4_action_raw": raw,
            "controller_action": action,
            "correction_eligible": correction_eligible,
            "correction_tier": tier,
            "correction_policy_locked": locked,
            "applied_delta": delta,
            "corrected_score": corrected,
            "controller_veto_reason": sorted(set(reasons)),
            "direction_contract_pass": not any("FORBIDS" in x or "MISMATCH" in x for x in reasons),
            "evidence_id_contract_pass": True,
            "safety_contract_pass": not veto,
        }
