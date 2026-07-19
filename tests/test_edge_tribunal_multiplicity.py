from __future__ import annotations

import pytest

from engine.experiments import multiplicity
from engine.experiments.errors import GateEvaluationError


def test_single_primary_mode_works() -> None:
    result = multiplicity.apply_correction(
        method="NONE_SINGLE_PRIMARY", hypothesis_ids=["h1"], p_values=[0.03],
        familywise_alpha=0.05, family_size=1, primary_hypothesis_index=0)
    assert result["primary_rejected"] is True
    assert result["primary_adjusted_threshold"] == 0.05
    failing = multiplicity.apply_correction(
        method="NONE_SINGLE_PRIMARY", hypothesis_ids=["h1"], p_values=[0.06],
        familywise_alpha=0.05, family_size=1, primary_hypothesis_index=0)
    assert failing["primary_rejected"] is False


def test_single_primary_rejects_larger_family() -> None:
    with pytest.raises(GateEvaluationError, match="exactly one"):
        multiplicity.apply_correction(
            method="NONE_SINGLE_PRIMARY", hypothesis_ids=["h1", "h2"],
            p_values=[0.01, 0.02], familywise_alpha=0.05, family_size=2,
            primary_hypothesis_index=0)


def test_bonferroni_matches_hand_calculation() -> None:
    # alpha=0.05, m=4 -> per-test threshold 0.0125.
    result = multiplicity.apply_correction(
        method="BONFERRONI", hypothesis_ids=["a", "b", "c", "d"],
        p_values=[0.01, 0.0125, 0.013, 0.9], familywise_alpha=0.05,
        family_size=4, primary_hypothesis_index=0)
    rejected = [item["rejected"] for item in result["results"]]
    assert rejected == [True, True, False, False]
    assert all(item["adjusted_threshold"] == pytest.approx(0.0125)
               for item in result["results"])


def test_holm_matches_hand_calculation() -> None:
    # Classic Holm example: p = [0.01, 0.02, 0.03, 0.04], alpha = 0.05.
    # Thresholds by rank: 0.0125, 0.016667, 0.025, 0.05.
    # 0.01 <= 0.0125 reject; 0.02 > 0.016667 stop; all later not rejected.
    result = multiplicity.apply_correction(
        method="HOLM_BONFERRONI", hypothesis_ids=["a", "b", "c", "d"],
        p_values=[0.01, 0.02, 0.03, 0.04], familywise_alpha=0.05,
        family_size=4, primary_hypothesis_index=0)
    rejected = [item["rejected"] for item in result["results"]]
    assert rejected == [True, False, False, False]
    assert result["results"][0]["adjusted_threshold"] == pytest.approx(0.05 / 4)
    assert result["results"][1]["adjusted_threshold"] == pytest.approx(0.05 / 3)
    assert result["results"][3]["adjusted_threshold"] == pytest.approx(0.05)


def test_holm_all_reject_when_every_step_passes() -> None:
    result = multiplicity.apply_correction(
        method="HOLM_BONFERRONI", hypothesis_ids=["a", "b", "c"],
        p_values=[0.001, 0.002, 0.003], familywise_alpha=0.05,
        family_size=3, primary_hypothesis_index=2)
    assert [item["rejected"] for item in result["results"]] == [True, True, True]
    assert result["primary_rejected"] is True


def test_holm_tie_behavior_is_deterministic() -> None:
    # Equal p-values tie-break on hypothesis ID, never on input order.
    first = multiplicity.apply_correction(
        method="HOLM_BONFERRONI", hypothesis_ids=["zeta", "alpha"],
        p_values=[0.02, 0.02], familywise_alpha=0.05, family_size=2,
        primary_hypothesis_index=0)
    second = multiplicity.apply_correction(
        method="HOLM_BONFERRONI", hypothesis_ids=["alpha", "zeta"],
        p_values=[0.02, 0.02], familywise_alpha=0.05, family_size=2,
        primary_hypothesis_index=1)
    by_id_first = {item["hypothesis_id"]: item for item in first["results"]}
    by_id_second = {item["hypothesis_id"]: item for item in second["results"]}
    assert by_id_first == by_id_second
    # alpha gets the strict rank-1 threshold (0.025), zeta the rank-2 (0.05).
    assert by_id_first["alpha"]["adjusted_threshold"] == pytest.approx(0.025)
    assert by_id_first["zeta"]["adjusted_threshold"] == pytest.approx(0.05)


def test_family_size_mismatch_fails() -> None:
    with pytest.raises(GateEvaluationError, match="never shrink"):
        multiplicity.apply_correction(
            method="BONFERRONI", hypothesis_ids=["a", "b"], p_values=[0.01, 0.02],
            familywise_alpha=0.05, family_size=5, primary_hypothesis_index=0)


def test_duplicate_hypothesis_ids_fail() -> None:
    with pytest.raises(GateEvaluationError, match="unique"):
        multiplicity.apply_correction(
            method="BONFERRONI", hypothesis_ids=["a", "a"], p_values=[0.01, 0.02],
            familywise_alpha=0.05, family_size=2, primary_hypothesis_index=0)


def test_invalid_p_value_fails() -> None:
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(GateEvaluationError):
            multiplicity.apply_correction(
                method="BONFERRONI", hypothesis_ids=["a"], p_values=[bad],
                familywise_alpha=0.05, family_size=1, primary_hypothesis_index=0)


def test_invalid_alpha_fails() -> None:
    for bad in (0.0, 1.0, -0.05):
        with pytest.raises(GateEvaluationError):
            multiplicity.apply_correction(
                method="BONFERRONI", hypothesis_ids=["a"], p_values=[0.01],
                familywise_alpha=bad, family_size=1, primary_hypothesis_index=0)


def test_unsupported_method_fails() -> None:
    with pytest.raises(GateEvaluationError, match="unsupported"):
        multiplicity.apply_correction(
            method="FDR_WISHFUL", hypothesis_ids=["a"], p_values=[0.01],
            familywise_alpha=0.05, family_size=1, primary_hypothesis_index=0)


def test_raw_pass_but_adjusted_fail_is_not_rejected() -> None:
    # Raw p 0.03 < alpha 0.05, but Bonferroni threshold is 0.05/5 = 0.01.
    result = multiplicity.apply_correction(
        method="BONFERRONI", hypothesis_ids=["a", "b", "c", "d", "e"],
        p_values=[0.03, 0.5, 0.6, 0.7, 0.8], familywise_alpha=0.05,
        family_size=5, primary_hypothesis_index=0)
    assert result["primary_rejected"] is False
    assert result["primary_adjusted_threshold"] == pytest.approx(0.01)


def test_primary_index_outside_family_fails() -> None:
    with pytest.raises(GateEvaluationError, match="outside"):
        multiplicity.apply_correction(
            method="BONFERRONI", hypothesis_ids=["a"], p_values=[0.01],
            familywise_alpha=0.05, family_size=1, primary_hypothesis_index=1)
