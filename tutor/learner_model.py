from collections import defaultdict
from datetime import datetime


def _ensure_state(session_state):
    if "learning_attempts" not in session_state:
        session_state["learning_attempts"] = []
    if "topic_mastery" not in session_state:
        session_state["topic_mastery"] = {}
    if "conversation_events" not in session_state:
        session_state["conversation_events"] = []


def _ensure_topic(session_state, topic):
    topic = (topic or "General").strip() or "General"
    mastery = session_state["topic_mastery"]
    if topic not in mastery:
        # Beta(2,2) prior = 50% initial mastery with moderate uncertainty.
        mastery[topic] = {"alpha": 2.0, "beta": 2.0, "attempts": 0}
    return topic, mastery[topic]


def record_attempt(session_state, topic, correct):
    """Update a lightweight Bayesian mastery estimate after an assessment item."""
    _ensure_state(session_state)
    topic, state = _ensure_topic(session_state, topic)
    correct = bool(correct)

    state["alpha"] += 1.0 if correct else 0.0
    state["beta"] += 0.0 if correct else 1.0
    state["attempts"] += 1

    session_state["learning_attempts"].append({
        "topic": topic,
        "correct": correct,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def record_conversation_feedback(session_state, topic, understood):
    """Let tutoring conversations update mastery without pretending correctness was observed."""
    _ensure_state(session_state)
    topic, state = _ensure_topic(session_state, topic)
    if understood:
        state["alpha"] += 0.25
    else:
        state["beta"] += 0.5
    session_state["conversation_events"].append({
        "topic": topic,
        "understood": bool(understood),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def get_learning_summary(session_state):
    _ensure_state(session_state)
    attempts = session_state["learning_attempts"]

    if not attempts:
        return {
            "attempts": 0,
            "accuracy": 0.0,
            "topics": [],
            "topic_accuracy": {},
            "topic_mastery": {},
            "conversation_events": len(session_state["conversation_events"]),
        }

    per_topic = defaultdict(list)
    for attempt in attempts:
        per_topic[attempt["topic"]].append(attempt["correct"])

    accuracy = sum(a["correct"] for a in attempts) / len(attempts)
    topic_accuracy = {
        topic: sum(values) / len(values)
        for topic, values in per_topic.items()
    }
    topic_mastery = {
        topic: round(state["alpha"] / (state["alpha"] + state["beta"]), 3)
        for topic, state in session_state["topic_mastery"].items()
    }
    recent_topics = list(dict.fromkeys(a["topic"] for a in reversed(attempts)))

    return {
        "attempts": len(attempts),
        "accuracy": accuracy,
        "topics": recent_topics,
        "topic_accuracy": topic_accuracy,
        "topic_mastery": topic_mastery,
        "conversation_events": len(session_state["conversation_events"]),
    }


def recommend_topics(session_state, fallback="General"):
    """Recommend lowest-mastery topics first; unseen topics come before mastered ones."""
    summary = get_learning_summary(session_state)
    mastery = summary["topic_mastery"]
    if not mastery:
        return [fallback or "General"]
    return sorted(mastery, key=lambda topic: (mastery[topic], topic))[:3]
