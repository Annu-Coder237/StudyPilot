import json
from pathlib import Path


def evaluate_retrieval(test_cases, retrieve_fn, chunks):
    """Small transparent benchmark: context precision/recall over known source locations."""
    rows = []
    for case in test_cases:
        results = retrieve_fn(case["question"], chunks, top_k=case.get("top_k", 5))
        predicted = {(r.get("source"), r.get("page")) for r in results}
        expected = {tuple(x) for x in case.get("expected_sources", [])}
        hit = predicted & expected
        precision = len(hit) / len(predicted) if predicted else 0.0
        recall = len(hit) / len(expected) if expected else 1.0
        rows.append({
            "question": case["question"],
            "context_precision": precision,
            "context_recall": recall,
            "hit": bool(hit),
        })
    if not rows:
        return {"rows": [], "context_precision": 0.0, "context_recall": 0.0}
    return {
        "rows": rows,
        "context_precision": sum(r["context_precision"] for r in rows) / len(rows),
        "context_recall": sum(r["context_recall"] for r in rows) / len(rows),
    }


def personalization_report(before_accuracy, after_accuracy, repeated_questions, total_questions):
    repetition_rate = repeated_questions / total_questions if total_questions else 0.0
    return {
        "accuracy_gain": after_accuracy - before_accuracy,
        "repetition_rate": repetition_rate,
        "mastery_gain": after_accuracy - before_accuracy,
    }


def load_test_set(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
