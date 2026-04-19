from typing import Dict, Tuple

import config


DECISION_MESSAGES = {
    "stable": {
        "ui": "Studentul este in parametri normali.",
        "raport": "Studentul se afla in parametri normali.",
        "needs_alert": False,
        "needs_tracking": False,
        "category": "stable",
    },
    "monitor": {
        "ui": "Studentul necesita monitorizare.",
        "raport": "Studentul prezinta abateri usoare si trebuie monitorizat.",
        "needs_alert": False,
        "needs_tracking": True,
        "category": "monitor",
    },
    "refocus": {
        "ui": "Atentie scazuta detectata.",
        "raport": "Studentul prezinta un nivel scazut de atentie, fara semne majore de oboseala.",
        "needs_alert": False,
        "needs_tracking": True,
        "category": "attention_issue",
    },
    "fatigue_risk": {
        "ui": "Semne de oboseala detectate.",
        "raport": "Studentul prezinta semne de oboseala care pot afecta activitatea.",
        "needs_alert": False,
        "needs_tracking": True,
        "category": "fatigue_issue",
    },
    "warning": {
        "ui": "Studentul necesita atentie.",
        "raport": "Studentul prezinta o combinatie problematica de atentie si oboseala.",
        "needs_alert": True,
        "needs_tracking": True,
        "category": "warning",
    },
    "critical": {
        "ui": "Stare critica detectata.",
        "raport": "Studentul se afla intr-o stare critica din punct de vedere al atentiei si/sau oboselii.",
        "needs_alert": True,
        "needs_tracking": True,
        "category": "critical",
    },
}


# linii = fatigue, coloane = attention
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

    raport_message = meta["raport"]
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
        "raport_message": raport_message,
        "report_message": raport_message,
        "decision_message": raport_message,
        "needs_alert": meta["needs_alert"],
        "needs_tracking": meta["needs_tracking"],
        "category": meta["category"],
        "severity": severity,
        "alert_type": alert_type,
    }