import pytest

from src.ai import daily_summary
from src.ai.retraining_explainer import (
    explain_retraining,
    explanation_html,
    explanation_text_lines,
)


def _result(promoted=True, criteria=None, regressions=None, **overrides):
    criteria = criteria or {
        "macro_f1_not_worse": True,
        "bootstrap_noninferior": True,
        "class_recall_guard": True,
    }
    result = {
        "retrained": True,
        "run_at": "20261006_031200",
        "run_id": 7,
        "promoted": promoted,
        "corrections_used": 12,
        "eval_holdout_size": 180,
        "baseline_accuracy": 0.914,
        "new_accuracy": 0.925,
        "baseline_recall_macro": 0.88,
        "new_recall_macro": 0.9,
        "baseline_f1": 0.89,
        "new_f1": 0.905,
        "training_duration": 420,
        "promotion_assessment": {
            "f1_delta": 0.015,
            "f1_delta_ci95_lower": -0.004,
            "f1_delta_ci95_upper": 0.031,
            "per_class_recall": {
                "delivery": {"support": 40, "baseline": 0.95, "candidate": 0.97, "delta": 0.02},
                "commercial": {"support": 50, "baseline": 0.90, "candidate": 0.78, "delta": -0.12},
                "social": {"support": 4, "baseline": 0.75, "candidate": 0.75, "delta": 0.0},
                "tech": {"support": 0, "baseline": 0.0, "candidate": 0.0, "delta": 0.0},
            },
            "recall_regressions": regressions or {},
            "promotion_criteria": criteria,
        },
    }
    result.update(overrides)
    return result


def test_promoted_run_says_new_version_is_in_use():
    explanation = explain_retraining(_result())

    assert explanation["headline"] == "Model update: new version in use"
    assert "12 of your corrections" in explanation["intro"]
    assert "180 emails" in explanation["intro"]
    assert explanation["overall"] == (
        "New version: 92 of 100 test emails sorted into the right folder "
        "(version in use before: 91)."
    )
    assert "RESTORE MODEL" in explanation["next_step"]


@pytest.mark.parametrize(
    ("failed", "regressions", "expected_reason", "expected_short"),
    [
        ("class_recall_guard", {"commercial": -0.12}, "noticeably worse at recognising Commercial emails", "worse at Commercial"),
        ("macro_f1_not_worse", None, "Averaged over all folders", "worse overall"),
        ("bootstrap_noninferior", None, "played it safe", "result too uncertain"),
    ],
)
def test_rejected_run_explains_the_failed_check(failed, regressions, expected_reason, expected_short):
    criteria = {
        "macro_f1_not_worse": True,
        "bootstrap_noninferior": True,
        "class_recall_guard": True,
        failed: False,
    }
    explanation = explain_retraining(_result(False, criteria, regressions))

    assert explanation["headline"] == "Model update: new version not used"
    assert len(explanation["reasons"]) == 1
    assert expected_reason in explanation["reasons"][0]
    assert explanation["short_reason"] == expected_short
    assert "try again" in explanation["next_step"]


def test_several_failed_checks_are_all_listed():
    criteria = {"macro_f1_not_worse": False, "bootstrap_noninferior": False, "class_recall_guard": False}
    explanation = explain_retraining(
        _result(False, criteria, {"commercial": -0.12, "delivery": -0.2})
    )

    assert len(explanation["reasons"]) == 3
    assert "Commercial and Delivery" in explanation["reasons"][0]


def test_runs_stored_before_checks_were_recorded_still_explain():
    result = _result(False)
    del result["promotion_assessment"]
    del result["corrections_used"]

    explanation = explain_retraining(result)

    assert explanation["reasons"] == ["It did not pass SMIO's quality checks."]
    assert "tested it on 180 emails" in explanation["intro"]
    assert explanation["folders"] == []


def test_folder_lines_use_plain_words():
    lines = explanation_text_lines(explain_retraining(_result()))

    assert "- Commercial: finds 78 of 100 (was 90) ⚠️ worse" in lines
    assert "- Delivery: finds 97 of 100 (was 95) ✅ better" in lines
    assert "- Social: finds 75 of 100 (was 75) ➖ about the same - only 4 test emails, rough estimate" in lines
    assert "- Tech: no test emails yet" in lines
    assert not any("F1" in line or "recall" in line.lower() for line in lines)


def test_no_run_yet():
    assert explain_retraining(None) is None
    assert explanation_text_lines(None) == ["No model update has run yet."]
    assert "No model update" in explanation_html(None)


def test_html_escapes_and_shows_verdict():
    output = explanation_html(explain_retraining(_result()))

    assert "Model update: new version in use" in output
    assert "<b>97 of 100</b>" in output
    assert "What to do:" in output


def _retraining(result):
    return {
        "last_run_at": result["run_at"],
        "last_promoted_at": result["run_at"] if result["promoted"] else None,
        "last_run_id": result["run_id"],
        "last_metrics": result,
        "history": [result],
        "promoted_runs": 1,
    }


def test_daily_summary_puts_plain_text_first_and_jargon_last():
    lines = daily_summary._model_and_footer_lines(_retraining(_result()))

    technical = lines.index("Technical details (for troubleshooting)")
    commands = lines.index("Email commands")
    assert lines[0] == "Model update: new version in use"
    assert commands < technical
    assert all("F1" not in line for line in lines[:commands])
    assert any("Macro F1" in line for line in lines[technical:])
    assert "Last model update: 06/10/2026 05:12" in lines


def test_daily_summary_html_renders_plain_and_technical_sections():
    result = _result(False, {"macro_f1_not_worse": False, "bootstrap_noninferior": True, "class_recall_guard": True})
    summary = {
        "period_start": "2026-10-06T06:00:00",
        "period_end": "2026-10-07T06:00:00",
        "new_counts": {"delivery": 2, "other": 1},
        "unread_counts": {"delivery": 1, "other": 0},
        "removed_counts": {"delivery": 0, "other": 0},
        "db": {"total_entries": 10, "pending_corrections": 3, "correction_threshold": 20, "retrain_due": False},
        "deliveries": {"today_count": 0, "upcoming": []},
        "retraining": _retraining(result),
    }

    output = daily_summary._format_html_summary(summary)

    assert output.index("Model update: new version not used") < output.index("Technical details")
    assert "<td>Not used</td><td>worse overall</td><td>92 of 100</td>" in output
    assert "Macro F1 delta 95% paired-bootstrap CI" in output
