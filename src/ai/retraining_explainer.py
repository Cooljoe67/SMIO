"""Explains a retraining result in plain language for non-technical readers.

The retraining code stores pass/fail checks and per-folder scores; this module
turns them into a verdict, a reason, "X of 100" figures and a next step.
Technical numbers stay available separately for troubleshooting.
"""

import html

from src.ai.promotion_rules import (
    PROMOTION_F1_NONINFERIORITY_MARGIN,
    PROMOTION_MAX_CLASS_RECALL_DROP,
    PROMOTION_MIN_CLASS_SUPPORT,
)

# Per-folder changes smaller than this (2 of 100) count as "about the same".
SAME_THRESHOLD = 0.02

TREND_SYMBOLS = {"better": "✅", "same": "➖", "worse": "⚠️"}
TREND_WORDS = {"better": "better", "same": "about the same", "worse": "worse"}


def _float(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _of_100(value):
    value = _float(value)
    return None if value is None else round(value * 100)


def _folder_name(label):
    return str(label).replace("_", " ").title()


def _join_names(names):
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _trend(delta):
    delta = _float(delta)
    if delta is None:
        return None
    if delta >= SAME_THRESHOLD:
        return "better"
    if delta <= -SAME_THRESHOLD:
        return "worse"
    return "same"


def _folders(assessment):
    folders = []
    for label, values in sorted((assessment.get("per_class_recall") or {}).items()):
        support = int(values.get("support") or 0)
        folders.append({
            "label": label,
            "name": _folder_name(label),
            "before": _of_100(values.get("baseline")) if support else None,
            "after": _of_100(values.get("candidate")) if support else None,
            "trend": _trend(values.get("delta")) if support else None,
            "test_emails": support,
            "reliable": support >= PROMOTION_MIN_CLASS_SUPPORT,
        })
    return folders


def _rejection_reasons(result, assessment):
    criteria = assessment.get("promotion_criteria") or {}
    drop_points = round(PROMOTION_MAX_CLASS_RECALL_DROP * 100)
    reasons = []

    if criteria.get("class_recall_guard") is False:
        worse = _join_names(
            _folder_name(label)
            for label in sorted(assessment.get("recall_regressions") or {})
        ) or "one folder"
        reasons.append((
            f"worse at {worse}",
            f"It got noticeably worse at recognising {worse} emails "
            f"(more than {drop_points} of 100 fewer found).",
        ))
    if criteria.get("macro_f1_not_worse") is False:
        reasons.append((
            "worse overall",
            "Averaged over all folders, it sorted the test emails slightly "
            "worse than the version in use.",
        ))
    if criteria.get("bootstrap_noninferior") is False:
        reasons.append((
            "result too uncertain",
            "The test could not show clearly enough that it is not worse, "
            "so SMIO played it safe.",
        ))
    if not reasons:
        # Runs stored before the checks were recorded only know the outcome.
        reasons.append((
            "did not pass the checks",
            "It did not pass SMIO's quality checks.",
        ))
    return reasons


def explain_retraining(result):
    """Return a plain-language explanation of one retraining result, or None."""
    if not result or result.get("promoted") is None:
        return None

    promoted = bool(result.get("promoted"))
    assessment = result.get("promotion_assessment") or {}
    test_emails = result.get("eval_holdout_size")
    corrections = result.get("corrections_used")

    if corrections is not None and test_emails:
        intro = (
            f"SMIO practised on {corrections} of your corrections and tested the "
            f"new version on {test_emails} emails it had never seen."
        )
    elif test_emails:
        intro = (
            "SMIO trained a new version from your corrections and tested it on "
            f"{test_emails} emails it had never seen."
        )
    else:
        intro = "SMIO trained a new version from your corrections and tested it."

    before = _of_100(result.get("baseline_accuracy"))
    new_accuracy = result.get("new_accuracy")
    after = _of_100(new_accuracy if new_accuracy is not None else result.get("accuracy"))
    if before is not None and after is not None:
        overall = (
            f"New version: {after} of 100 test emails sorted into the right folder "
            f"(version in use before: {before})."
        )
    elif after is not None:
        overall = f"New version: {after} of 100 test emails sorted into the right folder."
    else:
        overall = None

    if promoted:
        reasons = [("passed all checks", "It did at least as well as the old version, so it is now in use.")]
        headline = "Model update: new version in use"
        next_step = (
            "Nothing. If sorting feels worse, reply RESTORE MODEL "
            "within 7 days to switch back."
        )
    else:
        reasons = _rejection_reasons(result, assessment)
        headline = "Model update: new version not used"
        next_step = (
            "Nothing. Your corrections are kept; SMIO keeps using the current "
            "version and will try again when more corrections come in."
        )

    return {
        "promoted": promoted,
        "headline": headline,
        "intro": intro,
        "overall": overall,
        "reasons": [sentence for _, sentence in reasons],
        "short_reason": "; ".join(short for short, _ in reasons),
        "folders": _folders(assessment),
        "next_step": next_step,
        "technical": technical_lines(result),
    }


def _folder_line(folder):
    if folder["before"] is None or folder["after"] is None:
        return f"{folder['name']}: no test emails yet"
    line = (
        f"{folder['name']}: finds {folder['after']} of 100 (was {folder['before']}) "
        f"{TREND_SYMBOLS[folder['trend']]} {TREND_WORDS[folder['trend']]}"
    )
    if not folder["reliable"]:
        line += f" - only {folder['test_emails']} test emails, rough estimate"
    return line


def technical_lines(result):
    """Detailed figures for troubleshooting, in the original technical terms."""
    lines = []
    for label, before_key, after_key in (
        ("Accuracy", "baseline_accuracy", "new_accuracy"),
        ("Macro recall", "baseline_recall_macro", "new_recall_macro"),
        ("Macro F1", "baseline_f1", "new_f1"),
    ):
        before, after = _float(result.get(before_key)), _float(result.get(after_key))
        if after is not None:
            before_text = "-" if before is None else f"{before:.3f}"
            lines.append(f"{label}: {before_text} -> {after:.3f}")

    assessment = result.get("promotion_assessment") or {}
    lower = _float(assessment.get("f1_delta_ci95_lower"))
    upper = _float(assessment.get("f1_delta_ci95_upper"))
    if lower is not None and upper is not None:
        lines.append(f"Macro F1 delta 95% paired-bootstrap CI: {lower:+.3f} to {upper:+.3f}")
    criteria = assessment.get("promotion_criteria") or {}
    if criteria:
        lines.append(
            "Promotion checks: "
            + ", ".join(f"{name} {'passed' if ok else 'failed'}" for name, ok in criteria.items())
        )
    lines.append(
        "Promotion gate: macro F1 not lower; CI lower bound >= "
        f"-{PROMOTION_F1_NONINFERIORITY_MARGIN:.3f}; for classes with at least "
        f"{PROMOTION_MIN_CLASS_SUPPORT} holdout examples, recall drop <= "
        f"{PROMOTION_MAX_CLASS_RECALL_DROP:.3f}."
    )
    for label, values in sorted((assessment.get("per_class_recall") or {}).items()):
        baseline, candidate = _float(values.get("baseline")), _float(values.get("candidate"))
        if baseline is None or candidate is None:
            continue
        lines.append(
            f"Recall {label}: {baseline:.3f} -> {candidate:.3f} "
            f"(holdout n={values.get('support', 0)})"
        )
    return lines


def explanation_text_lines(explanation):
    """Plain-text lines for the main part of an email."""
    if explanation is None:
        return ["No model update has run yet."]
    lines = [explanation["headline"], explanation["intro"]]
    if explanation["overall"]:
        lines.append(explanation["overall"])
    lines.extend(explanation["reasons"])
    if explanation["folders"]:
        lines.append("How well each folder is recognised (out of 100 emails):")
        lines.extend(f"- {_folder_line(folder)}" for folder in explanation["folders"])
    lines.append(f"What to do: {explanation['next_step']}")
    return lines


def explanation_html(explanation):
    """HTML fragment for the main part of an email (inline styles only)."""
    if explanation is None:
        return "<p>No model update has run yet.</p>"

    colour = "#2e7d32" if explanation["promoted"] else "#b54708"
    rows = []
    for folder in explanation["folders"]:
        if folder["before"] is None or folder["after"] is None:
            rows.append(
                f"<tr><td>{html.escape(folder['name'])}</td>"
                '<td colspan="3">no test emails yet</td></tr>'
            )
            continue
        note = "" if folder["reliable"] else (
            f'<br><span style="color:#667085;font-size:12px">only '
            f"{folder['test_emails']} test emails, rough estimate</span>"
        )
        rows.append(
            f"<tr><td>{html.escape(folder['name'])}{note}</td>"
            f"<td>{folder['before']} of 100</td>"
            f"<td><b>{folder['after']} of 100</b></td>"
            f"<td>{TREND_SYMBOLS[folder['trend']]} {TREND_WORDS[folder['trend']]}</td></tr>"
        )
    table = ""
    if rows:
        table = (
            '<table style="width:100%;border-collapse:collapse">'
            '<tr><th align="left">Folder</th><th align="left">Before</th>'
            '<th align="left">New version</th><th align="left"></th></tr>'
            + "".join(rows)
            + "</table>"
        )
    paragraphs = [explanation["intro"]]
    if explanation["overall"]:
        paragraphs.append(explanation["overall"])
    paragraphs.extend(explanation["reasons"])
    return (
        f'<p style="font-size:16px;color:{colour}"><b>{html.escape(explanation["headline"])}</b></p>'
        + "".join(f"<p>{html.escape(text)}</p>" for text in paragraphs)
        + (
            '<p style="margin-bottom:4px"><b>How well each folder is recognised</b> '
            "(out of 100 emails)</p>" + table if table else ""
        )
        + '<p style="background:#f4f6f8;padding:10px 12px;border-left:4px solid #173f5f">'
        f"<b>What to do:</b> {html.escape(explanation['next_step'])}</p>"
    )
