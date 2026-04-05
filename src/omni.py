"""
src/omni.py - VIPER Forensic Engine executive-language explainers.

This module translates structured forensic outputs into concise, non-technical
English that fits a live demo, dashboard, or judge-facing report card.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_MODEL_NAME = "VIPER"

_GENERIC_VERDICTS = {
    "synthetic artifact pattern detected",
    "natural capture signature favored",
}

_LEVEL_ALIASES = {
    "low": "low",
    "mild": "low",
    "limited": "low",
    "weak": "low",
    "small": "low",
    "medium": "medium",
    "moderate": "medium",
    "mid": "medium",
    "mixed": "medium",
    "elevated": "high",
    "detected": "high",
    "strong": "high",
    "high": "high",
}

_LEVEL_TO_SCORE = {
    "low": 0.25,
    "medium": 0.58,
    "high": 0.85,
}

_ANOMALY_PHRASES = {
    "fft": "elevated high-frequency FFT activity",
    "prnu": "an irregular sensor-noise pattern",
    "lab": "unnatural color saturation",
    "gradcam": "concentrated attention on localized artifact regions",
}

_NATURAL_PHRASES = {
    "fft": "a frequency profile close to the natural baseline",
    "prnu": "a comparatively stable noise signature",
    "lab": "restrained color saturation",
    "gradcam": "diffuse attention rather than concentrated artifact hotspots",
}

__all__ = [
    "explain_forensic_report_card",
    "explain_report_card",
    "generate_error_insight",
    "explain_error_breakdown",
]


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_probability(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed /= 100.0
    return _clamp(parsed, 0.0, 1.0)


def _parse_score(value: Any) -> float | None:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    if parsed > 1.0 and parsed <= 100.0:
        parsed /= 100.0
    return _clamp(parsed, 0.0, 1.0)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _oxford_join(parts: Sequence[str]) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _normalize_level(value: Any) -> str | None:
    text = _clean_text(value).lower().replace("-", " ")
    if not text:
        return None
    return _LEVEL_ALIASES.get(text)


def _canonical_signal_id(raw_value: Any) -> str | None:
    text = _clean_text(raw_value).lower().replace("-", "").replace("_", "")
    if not text:
        return None
    if "fft" in text:
        return "fft"
    if "prnu" in text or "noise" in text:
        return "prnu"
    if text.startswith("lab") or "chroma" in text or "saturation" in text:
        return "lab"
    if "gradcam" in text or text == "cam":
        return "gradcam"
    return None


def _prediction_label(report_card: Mapping[str, Any]) -> str | None:
    predicted_index = _safe_float(report_card.get("predicted_index"))
    if predicted_index is not None:
        return "ai" if int(round(predicted_index)) == 1 else "real"

    raw_prediction = _clean_text(
        report_card.get("prediction")
        or report_card.get("predicted_label")
        or report_card.get("class")
        or report_card.get("label")
    ).lower()
    if raw_prediction:
        if "ai" in raw_prediction or "synthetic" in raw_prediction or "generated" in raw_prediction:
            return "ai"
        if "real" in raw_prediction or "authentic" in raw_prediction or "natural" in raw_prediction:
            return "real"

    ai_probability = _parse_probability(
        report_card.get("ai_probability")
        or report_card.get("p_ai_generated")
        or report_card.get("p_ai")
    )
    if ai_probability is not None:
        return "ai" if ai_probability >= 0.5 else "real"
    return None


def _predicted_confidence(report_card: Mapping[str, Any], prediction: str | None) -> float | None:
    confidence = _parse_probability(
        report_card.get("confidence_score")
        or report_card.get("confidence")
        or report_card.get("confidence_pct")
    )
    if confidence is not None:
        return confidence

    ai_probability = _parse_probability(
        report_card.get("ai_probability")
        or report_card.get("p_ai_generated")
        or report_card.get("p_ai")
    )
    if ai_probability is None:
        return None
    if prediction == "ai":
        return ai_probability
    if prediction == "real":
        return 1.0 - ai_probability
    return max(ai_probability, 1.0 - ai_probability)


def _generic_verdict(verdict: str) -> bool:
    return verdict.lower().rstrip(".") in _GENERIC_VERDICTS


def _normalized_evidence_from_breakdown(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        signal_id = _canonical_signal_id(item.get("id") or item.get("label"))
        label = _clean_text(item.get("label"))
        detail = _clean_text(item.get("detail"))
        score = _parse_score(item.get("score"))
        level = _normalize_level(item.get("status"))
        if score is None and level is not None:
            score = _LEVEL_TO_SCORE[level]
        if score is None:
            continue
        normalized.append(
            {
                "id": signal_id or label.lower() or "signal",
                "label": label or "Signal",
                "score": score,
                "detail": detail,
            }
        )
    return normalized


def _extract_signal_from_fields(report_card: Mapping[str, Any], signal_id: str) -> dict[str, Any] | None:
    direct_value = report_card.get(signal_id)
    score = None
    level = None
    detail = ""

    if isinstance(direct_value, Mapping):
        score = _parse_score(direct_value.get("score"))
        level = _normalize_level(
            direct_value.get("label")
            or direct_value.get("status")
            or direct_value.get("bucket")
        )
        detail = _clean_text(direct_value.get("detail"))
    else:
        level = _normalize_level(direct_value)

    score = score if score is not None else _parse_score(report_card.get(f"{signal_id}_score"))
    level = level or _normalize_level(
        report_card.get(f"{signal_id}_label")
        or report_card.get(f"{signal_id}_status")
        or report_card.get(f"{signal_id}_bucket")
    )
    detail = detail or _clean_text(report_card.get(f"{signal_id}_detail"))

    if score is None and level is None:
        return None
    if score is None and level is not None:
        score = _LEVEL_TO_SCORE[level]

    return {
        "id": signal_id,
        "label": signal_id.upper() if signal_id != "gradcam" else "Grad-CAM",
        "score": score,
        "detail": detail,
    }


def _collect_evidence(report_card: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = _normalized_evidence_from_breakdown(report_card.get("evidence_breakdown"))
    if evidence:
        return evidence

    fallback: list[dict[str, Any]] = []
    for signal_id in ("fft", "prnu", "lab", "gradcam"):
        item = _extract_signal_from_fields(report_card, signal_id)
        if item is not None:
            fallback.append(item)
    return fallback


def _supporting_phrases(
    evidence: Sequence[dict[str, Any]],
    prediction: str | None,
) -> list[str]:
    if not evidence:
        return []

    phrases: list[str] = []
    if prediction == "real":
        candidates = [item for item in evidence if item["id"] != "gradcam"] or list(evidence)
        ranked = sorted(candidates, key=lambda item: float(item["score"]))
        for item in ranked:
            if float(item["score"]) > 0.65 and phrases:
                break
            phrases.append(_NATURAL_PHRASES.get(item["id"], item["label"].lower()))
            if len(phrases) == 2:
                break
    else:
        ranked = sorted(evidence, key=lambda item: float(item["score"]), reverse=True)
        for item in ranked:
            if float(item["score"]) < 0.45 and phrases:
                break
            phrases.append(_ANOMALY_PHRASES.get(item["id"], item["label"].lower()))
            if len(phrases) == 2:
                break

    deduped: list[str] = []
    for phrase in phrases:
        if phrase not in deduped:
            deduped.append(phrase)
    return deduped


def explain_forensic_report_card(report_card: Mapping[str, Any]) -> str:
    """
    Turn a forensic report dictionary into a short executive-language insight.

    The function accepts either the current backend prediction payload or a
    lighter-weight report card containing prediction/confidence plus FFT, PRNU,
    and LAB signal scores.
    """
    if not isinstance(report_card, Mapping):
        raise TypeError("report_card must be a mapping-like object.")

    model_name = _clean_text(report_card.get("model_name")) or DEFAULT_MODEL_NAME
    prediction = _prediction_label(report_card)
    confidence = _predicted_confidence(report_card, prediction)
    verdict = _clean_text(report_card.get("final_verdict") or report_card.get("verdict"))
    evidence = _collect_evidence(report_card)
    phrases = _supporting_phrases(evidence, prediction)

    if prediction == "real":
        decision_label = "authentic rather than AI-generated"
    elif prediction == "ai":
        decision_label = "AI-generated"
    else:
        decision_label = "forensically suspicious"

    if confidence is None:
        intro = f"{model_name}'s forensic read favors a {decision_label} interpretation"
    elif confidence < 0.68:
        if prediction == "real":
            intro = (
                f"{model_name} leans authentic at {confidence:.0%} confidence, "
                "so this is a close call rather than a decisive clearance"
            )
        elif prediction == "ai":
            intro = (
                f"{model_name} leans AI-generated at {confidence:.0%} confidence, "
                "so this is a close call rather than a decisive flag"
            )
        else:
            intro = (
                f"{model_name} shows only a modest signal at {confidence:.0%} confidence, "
                "so this case sits near the decision boundary"
            )
    else:
        intro = f"{model_name} is {confidence:.0%} confident this image is {decision_label}"

    if confidence is not None and confidence < 0.68:
        if phrases:
            return (
                f"{intro}, with the strongest cues coming from {_oxford_join(phrases)}. "
                "The supporting evidence remains mixed."
            )
        return f"{intro}. The supporting evidence remains mixed."

    if phrases:
        connector = "supported primarily by" if prediction == "real" else "driven primarily by"
        return f"{intro}, {connector} {_oxford_join(phrases)}."

    if verdict and not _generic_verdict(verdict):
        verdict = verdict.rstrip(".")
        return f"{intro}. Overall assessment: {_sentence_case(verdict)}."

    return f"{intro}."


def explain_report_card(report_card: Mapping[str, Any]) -> str:
    """Alias for explain_forensic_report_card()."""
    return explain_forensic_report_card(report_card)


def _count_from_value(value: Any) -> int | None:
    parsed = _safe_float(value)
    if parsed is not None:
        return max(int(round(parsed)), 0)

    if isinstance(value, Mapping):
        if "count" in value:
            nested = _count_from_value(value.get("count"))
            if nested is not None:
                return nested
        subtotal = 0
        found = False
        for nested_value in value.values():
            nested_count = _count_from_value(nested_value)
            if nested_count is None:
                continue
            subtotal += nested_count
            found = True
        return subtotal if found else None

    return None


def _first_count(stats: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        if key in stats:
            count = _count_from_value(stats.get(key))
            if count is not None:
                return count
    return None


def _normalize_bucket_name(raw_key: Any) -> str | None:
    key = _clean_text(raw_key).lower().replace("-", "_").replace(" ", "_")
    if not key:
        return None
    if "low" in key:
        return "low"
    if "medium" in key or "mid" in key or "moderate" in key:
        return "medium"
    if "high" in key:
        return "high"
    return None


def _bucket_counts_from_mapping(payload: Any) -> dict[str, int]:
    buckets = {"low": 0, "medium": 0, "high": 0}
    if not isinstance(payload, Mapping):
        return buckets

    for key, value in payload.items():
        bucket_name = _normalize_bucket_name(key)
        if bucket_name is None:
            continue
        count = _count_from_value(value)
        if count is None:
            continue
        buckets[bucket_name] += count
    return buckets


def _extract_confidence_buckets(stats: Mapping[str, Any]) -> dict[str, int]:
    direct_buckets = _bucket_counts_from_mapping(stats.get("confidence_buckets"))
    if any(direct_buckets.values()):
        return direct_buckets

    nested_totals = {"low": 0, "medium": 0, "high": 0}
    found_nested = False
    for key in ("false_positives", "false_negatives", "fp", "fn"):
        nested = _bucket_counts_from_mapping(stats.get(key))
        if any(nested.values()):
            found_nested = True
            for bucket_name, count in nested.items():
                nested_totals[bucket_name] += count
    if found_nested:
        return nested_totals

    flat_totals = {"low": 0, "medium": 0, "high": 0}
    for key, value in stats.items():
        normalized_key = _clean_text(key).lower().replace("-", "_").replace(" ", "_")
        if "confidence" not in normalized_key:
            continue
        if not any(token in normalized_key for token in ("error", "misclass", "false_positive", "false_negative", "fp", "fn")):
            continue
        bucket_name = _normalize_bucket_name(normalized_key)
        if bucket_name is None:
            continue
        count = _count_from_value(value)
        if count is None:
            continue
        flat_totals[bucket_name] += count
    return flat_totals


def generate_error_insight(error_breakdown: Mapping[str, Any]) -> str:
    """
    Summarize a model error breakdown as one professional sentence.

    The function accepts either flat counters or nested dictionaries such as:
      {"false_positives": 8, "false_negatives": 3}
      {"false_positives": {"low": 5, "high": 2}, "false_negatives": {"low": 1}}
    """
    if not isinstance(error_breakdown, Mapping):
        raise TypeError("error_breakdown must be a mapping-like object.")

    false_positives = _first_count(
        error_breakdown,
        ("false_positives", "fp", "real_as_ai", "real_flagged_as_ai"),
    ) or 0
    false_negatives = _first_count(
        error_breakdown,
        ("false_negatives", "fn", "ai_as_real", "ai_missed_as_real"),
    ) or 0

    total_errors = false_positives + false_negatives
    if total_errors == 0:
        total_errors = _first_count(
            error_breakdown,
            ("total_errors", "errors", "misclassified", "misclassifications"),
        ) or 0

    if total_errors == 0:
        return (
            "The current error breakdown does not yet expose a stable blind spot, "
            "either because errors are absent or the counts are not populated."
        )

    count_clause = ""
    if false_positives or false_negatives:
        count_clause = f" ({false_positives} false positives vs {false_negatives} false negatives)"

    if false_positives >= max(false_negatives * 1.25, false_negatives + 2):
        direction_clause = (
            "the model's main blind spot is over-flagging authentic artwork as AI-generated"
        )
    elif false_negatives >= max(false_positives * 1.25, false_positives + 2):
        direction_clause = (
            "the model's main blind spot is missing AI-generated work by treating it as authentic"
        )
    else:
        direction_clause = (
            "the error profile is fairly balanced between over-flagging real work and missing synthetic work"
        )

    buckets = _extract_confidence_buckets(error_breakdown)
    low_errors = buckets["low"]
    medium_errors = buckets["medium"]
    high_errors = buckets["high"]

    if high_errors >= max(low_errors * 1.25, low_errors + 2) and high_errors > 0:
        confidence_clause = (
            "many of those misses occur at high confidence, which suggests a systematic bias rather than simple uncertainty"
        )
    elif low_errors >= max(high_errors * 1.25, high_errors + 2) and low_errors > 0:
        confidence_clause = (
            "most of those misses cluster in low-confidence cases, which suggests ambiguity near the decision boundary"
        )
    elif medium_errors > max(low_errors, high_errors):
        confidence_clause = (
            "most of those misses sit in the mid-confidence band, which suggests recurring ambiguity instead of a single sharp failure mode"
        )
    else:
        confidence_clause = (
            "the confidence mix is broad, so the weakness appears distributed rather than concentrated in one certainty band"
        )

    prefix = "In the current sample, " if total_errors < 4 else ""
    return _sentence_case(f"{prefix}{direction_clause}{count_clause}, and {confidence_clause}.")


def explain_error_breakdown(error_breakdown: Mapping[str, Any]) -> str:
    """Alias for generate_error_insight()."""
    return generate_error_insight(error_breakdown)
