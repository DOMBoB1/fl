from typing import Dict, List, Tuple

import config


SUGGESTION_MESSAGES = {
    "stable": {
        "ui": "Student state is stable.",
        "raport": "The student remained within normal monitoring parameters.",
        "needs_alert": False,
        "needs_tracking": False,
        "category": "stable",
    },
    "monitor": {
        "ui": "Student should be monitored.",
        "raport": "The student showed mild deviations and should continue to be monitored.",
        "needs_alert": False,
        "needs_tracking": True,
        "category": "monitor",
    },
    "refocus": {
        "ui": "Low attention detected.",
        "raport": "The student showed low attention without major fatigue signs.",
        "needs_alert": False,
        "needs_tracking": True,
        "category": "attention_issue",
    },
    "fatigue_risk": {
        "ui": "Fatigue signs detected.",
        "raport": "The student showed fatigue signs that may affect class activity.",
        "needs_alert": False,
        "needs_tracking": True,
        "category": "fatigue_issue",
    },
    "warning": {
        "ui": "Student alert detected.",
        "raport": "The student showed a problematic combination of attention and fatigue indicators.",
        "needs_alert": True,
        "needs_tracking": True,
        "category": "warning",
    },
    "critical": {
        "ui": "Critical student state detected.",
        "raport": "The student reached a critical state based on attention and/or fatigue indicators.",
        "needs_alert": True,
        "needs_tracking": True,
        "category": "critical",
    },
}

DECISION_MESSAGES = SUGGESTION_MESSAGES

DECISION_MATRIX = {
    "ideal": {
        "crit": "refocus",
        "low": "refocus",
        "ok": "monitor",
        "good": "stable",
        "ideal": "stable",
    },
    "good": {
        "crit": "refocus",
        "low": "monitor",
        "ok": "stable",
        "good": "stable",
        "ideal": "stable",
    },
    "ok": {
        "crit": "warning",
        "low": "fatigue_risk",
        "ok": "monitor",
        "good": "stable",
        "ideal": "stable",
    },
    "low": {
        "crit": "critical",
        "low": "warning",
        "ok": "fatigue_risk",
        "good": "monitor",
        "ideal": "monitor",
    },
    "crit": {
        "crit": "critical",
        "low": "critical",
        "ok": "warning",
        "good": "fatigue_risk",
        "ideal": "fatigue_risk",
    },
}


def _clamp_score(score: float) -> float:
    try:
        return max(0.0, min(100.0, float(score)))
    except Exception:
        return 0.0


def _classify_by_thresholds(score: float, thresholds: Dict[str, Tuple[int, int]]) -> str:
    score = _clamp_score(score)

    for level, (min_v, max_v) in thresholds.items():
        if min_v <= score <= max_v:
            return level

    keys = list(thresholds.keys())
    if not keys:
        return "ok"

    return keys[0] if score <= 0 else keys[-1]


def classify_attention(attention_score: float) -> str:
    return _classify_by_thresholds(attention_score, config.ATTENTION_THRESHOLDS)


def classify_fatigue(fatigue_score: float) -> str:
    return _classify_by_thresholds(fatigue_score, config.FATIGUE_THRESHOLDS)


def get_decision(attention_level: str, fatigue_level: str) -> str:
    fatigue_map = DECISION_MATRIX.get(fatigue_level)

    if fatigue_map is None:
        return "monitor"

    return fatigue_map.get(attention_level, "monitor")


def _decision_to_severity_and_alert_type(decision: str) -> tuple[str, str]:
    if decision == "critical":
        return "critical", "fatigue_attention"

    if decision == "warning":
        return "warning", "fatigue_attention"

    if decision == "fatigue_risk":
        return "warning", "fatigue"

    if decision == "refocus":
        return "warning", "attention"

    return "none", "none"


def evaluate_student_state(attention_score: float, fatigue_score: float) -> dict:
    attention_score = _clamp_score(attention_score)
    fatigue_score = _clamp_score(fatigue_score)

    attention_level = classify_attention(attention_score)
    fatigue_level = classify_fatigue(fatigue_score)

    decision = get_decision(attention_level, fatigue_level)
    meta = DECISION_MESSAGES[decision]

    color_map = getattr(config, "DECISION_COLORS", {})
    priority_map = getattr(config, "DECISION_PRIORITY", {})

    report_message = meta["raport"]
    severity, alert_type = _decision_to_severity_and_alert_type(decision)

    return {
        "attention_score": round(attention_score, 2),
        "fatigue_score": round(fatigue_score, 2),
        "attention_level": attention_level,
        "fatigue_level": fatigue_level,
        "decision": decision,
        "priority": priority_map.get(decision, 3),
        "decision_priority": priority_map.get(decision, 3),
        "color": color_map.get(decision, "#999999"),
        "decision_color": color_map.get(decision, "#999999"),
        "ui_message": meta["ui"],
        "raport_message": report_message,
        "report_message": report_message,
        "decision_message": report_message,
        "needs_alert": meta["needs_alert"],
        "needs_tracking": meta["needs_tracking"],
        "category": meta["category"],
        "severity": severity,
        "alert_type": alert_type,
    }


def get_ui_thresholds() -> dict:
    return {
        "class_fatigue_on": int(getattr(config, "ALERT_CLASS_FATIGUE_ON", 50)),
        "class_fatigue_off": int(getattr(config, "ALERT_CLASS_FATIGUE_OFF", 45)),
        "class_attention_on": int(getattr(config, "ALERT_CLASS_ATTENTION_ON", 50)),
        "class_attention_off": int(getattr(config, "ALERT_CLASS_ATTENTION_OFF", 55)),
        "student_fatigue_on": int(getattr(config, "ALERT_STUDENT_FATIGUE_ON", 60)),
        "student_fatigue_off": int(getattr(config, "ALERT_STUDENT_FATIGUE_OFF", 52)),
        "student_attention_on": int(getattr(config, "ALERT_STUDENT_ATTENTION_ON", 50)),
        "student_attention_off": int(getattr(config, "ALERT_STUDENT_ATTENTION_OFF", 55)),
        "student_fatigue_critical": int(getattr(config, "ALERT_STUDENT_FATIGUE_CRITICAL", 70)),
        "student_attention_critical": int(getattr(config, "ALERT_STUDENT_ATTENTION_CRITICAL", 30)),
        "min_active_students_for_class_alert": int(getattr(config, "MIN_ACTIVE_STUDENTS_FOR_CLASS_ALERT", 1)),
    }


def build_class_suggestion(
    heads: int,
    fatigue: float,
    attention: float,
    fatigue_alert_active: bool,
    attention_alert_active: bool,
) -> dict:
    heads = int(max(0, heads or 0))
    fatigue = round(_clamp_score(fatigue), 1)
    attention = round(_clamp_score(attention), 1)
    thresholds = get_ui_thresholds()
    suggestions: List[str] = []

    if heads <= 0:
        return {
            "summary": "No student is currently detected, so the class state cannot be evaluated yet.",
            "suggestions": ["Check camera position and wait until at least one student is visible."],
            "source": "backend",
            "scope": "class",
            "thresholds": thresholds,
        }

    if fatigue_alert_active and attention_alert_active:
        summary = f"Class alert: {heads} detected, average fatigue is {fatigue}% and average attention is {attention}%."
        suggestions.append("Use a short pause, stretch moment, or easier transition task.")
        suggestions.append("Ask a short recap question before continuing the lesson.")
    elif fatigue_alert_active:
        summary = f"Class fatigue alert: {heads} detected and average fatigue is {fatigue}%."
        suggestions.append("Use a short pause, stretch moment, or easier transition task.")
    elif attention_alert_active:
        summary = f"Class attention alert: {heads} detected and average attention is {attention}%."
        suggestions.append("Change the rhythm of the lesson and ask a direct question to the class.")
    else:
        summary = f"Class state is stable. Detected people: {heads}, average fatigue: {fatigue}%, average attention: {attention}%."
        suggestions.append("Continue the current teaching flow and monitor the next changes.")

    return {
        "summary": summary,
        "suggestions": suggestions,
        "source": "backend",
        "scope": "class",
        "thresholds": thresholds,
    }


def build_student_suggestion(student_alerts: List[dict]) -> dict:
    student_alerts = student_alerts or []
    suggestions: List[str] = []

    if not student_alerts:
        return {
            "summary": "No student-level alert is currently active.",
            "suggestions": [],
            "source": "backend",
            "scope": "student",
        }

    critical = [
        item for item in student_alerts
        if str(item.get("severity", "")).lower() == "critical"
    ]

    fatigue_alerts = [
        item for item in student_alerts
        if bool(item.get("fatigue_alert_active"))
    ]

    attention_alerts = [
        item for item in student_alerts
        if bool(item.get("attention_alert_active"))
    ]

    count = len(student_alerts)
    summary = f"{count} student-level alert{' is' if count == 1 else 's are'} currently active."

    if critical:
        suggestions.append("Prioritize the critical student alert first." if len(critical) == 1 else "Prioritize the critical student alerts first.")

    if fatigue_alerts:
        suggestions.append("For fatigue alerts, use a short pause or a low-pressure question.")

    if attention_alerts:
        suggestions.append("For attention alerts, use a direct interaction or change the activity rhythm.")

    return {
        "summary": summary,
        "suggestions": list(dict.fromkeys(suggestions)),
        "source": "backend",
        "scope": "student",
    }


def build_group_decision(
    heads: int,
    fatigue: float,
    attention: float,
    fatigue_alert_active: bool,
    attention_alert_active: bool,
) -> dict:
    return build_class_suggestion(
        heads=heads,
        fatigue=fatigue,
        attention=attention,
        fatigue_alert_active=fatigue_alert_active,
        attention_alert_active=attention_alert_active,
    )


def build_individual_decision(student_alerts: List[dict]) -> dict:
    return build_student_suggestion(student_alerts)


def build_ui_decisions(
    heads: int,
    fatigue: float,
    attention: float,
    fatigue_alert_active: bool,
    attention_alert_active: bool,
    student_alerts: List[dict],
) -> dict:
    return {
        "group": build_class_suggestion(
            heads=heads,
            fatigue=fatigue,
            attention=attention,
            fatigue_alert_active=fatigue_alert_active,
            attention_alert_active=attention_alert_active,
        ),
        "individual": build_student_suggestion(student_alerts or []),
        "class": build_class_suggestion(
            heads=heads,
            fatigue=fatigue,
            attention=attention,
            fatigue_alert_active=fatigue_alert_active,
            attention_alert_active=attention_alert_active,
        ),
        "student": build_student_suggestion(student_alerts or []),
    }