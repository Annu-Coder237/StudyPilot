
import json
import re

from tutor.generation import _chat


def _repair_json_backslashes(raw: str) -> str:
    """Escape raw LaTeX backslashes inside JSON strings without changing valid JSON escapes."""
    out = []
    in_string = False
    escaped = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
            continue
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            i += 1
            continue
        if ch == "\\":
            nxt = raw[i + 1] if i + 1 < len(raw) else ""
            # A LaTeX command is a backslash followed by a word, e.g. \theta,
            # \text, \frac. JSON's \t or \n escapes are only valid when
            # not followed by the rest of an alphabetic command.
            if nxt.isalpha():
                j = i + 1
                while j < len(raw) and raw[j].isalpha():
                    j += 1
                command = raw[i + 1:j]
                if len(command) > 1 or command not in "bfnrt":
                    out.append("\\\\")
                    i += 1
                    continue
                # Valid one-character JSON escape such as \n or \t.
                out.append(ch)
                escaped = True
                i += 1
                continue
            if nxt in '"\\/bfnrt':
                out.append(ch)
                escaped = True
                i += 1
                continue
            if nxt == "u" and i + 5 < len(raw) and re.fullmatch(r"[0-9a-fA-F]{4}", raw[i+2:i+6]):
                out.append(ch)
                escaped = True
                i += 1
                continue
            # Invalid escape: make it a literal backslash.
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_json(text):
    """Extract JSON from model output and repair common unescaped LaTeX commands."""
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The model did not return a JSON object.")
    raw = text[start:end + 1]
    # Repair first: json.loads alone can silently interpret LaTeX commands
    # such as \theta and \frac as JSON tab/form-feed escapes, corrupting math.
    repaired = _repair_json_backslashes(raw)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as repaired_error:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as original_error:
            raise ValueError(
                f"Model output was not valid JSON (after LaTeX repair: {repaired_error}; "
                f"original: {original_error})."
            ) from repaired_error


def _context(chunks):
    mapping = {}
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        source_id = f"C{i}"
        mapping[source_id] = chunk
        parts.append(
            f"[{source_id} | source={chunk.get('source', 'Unknown')} "
            f"| page={chunk.get('page', '?')}]\n"
            f"{chunk.get('text', '')}"
        )

    return "\n\n---\n\n".join(parts), mapping


def _normalise(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _canonicalise_math_markup(text):
    """Use one delimiter style that Streamlit can render consistently."""
    text = str(text).strip()
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)


def _has_readable_math_markup(text):
    """Reject malformed or bare LaTeX before it reaches the assessment UI."""
    text = str(text)
    if text.count("$") % 2:
        return False
    math_parts = re.findall(r"\$\$?(.*?)\$\$?", text, flags=re.DOTALL)
    for part in math_parts:
        if not part.strip() or part.count("{") != part.count("}"):
            return False
    outside_math = re.sub(r"\$\$?.*?\$\$?", "", text, flags=re.DOTALL)
    return not re.search(r"\\(?:frac|sqrt|sum|int|alpha|beta|gamma|theta|pi|sin|cos|log)\b", outside_math)


def _options_are_meaningful(options):
    """Reject duplicated, empty, or placeholder choices, including math variants."""
    keys = [_normalise(option).replace("$", "") for option in options]
    if len(set(keys)) != len(options) or any(not key for key in keys):
        return False
    return not any(re.fullmatch(r"(?:option|choice)\s*[a-d1-4]?", key) for key in keys)



def generate_assessment(
    topic,
    chunks,
    num_questions=5,
    difficulty="Mixed",
    question_history=None,
    model="qwen2.5:3b",
    ollama_url="http://localhost:11434",
):
    if not chunks:
        raise ValueError("No source passages were retrieved for this topic.")

    if not 3 <= num_questions <= 10:
        raise ValueError("Choose between 3 and 10 questions.")

    context, mapping = _context(chunks)
    history = list(question_history or [])[-100:]

    assessment_schema = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["mcq", "short_answer", "numerical"]
                    },
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "topic": {"type": "string"},
                    "source_id": {"type": "string"},
                    "difficulty": {
                        "type": "string",
                        "enum": ["Easy", "Medium", "Hard"]
                    }
                },
                "required": [
                    "type",
                    "question",
                    "options",
                    "answer",
                    "explanation",
                    "topic",
                    "source_id",
                    "difficulty"
                ]
            }
        }
    },
    "required": ["questions"]
}
    verification_schema = {
    "type": "object",
    "properties": {
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string"
                    },
                    "valid": {
                        "type": "boolean"
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": ["id", "valid", "reason"]
            }
        }
    },
    "required": ["checks"]
}

    # For five questions: 2 MCQs, 2 short answers, 1 numerical.
    types = ["mcq", "short_answer", "numerical"]
    base, remainder = divmod(num_questions, len(types))
    target = {kind: base for kind in types}
    for kind in types[:remainder]:
        target[kind] += 1

    accepted = []
    seen = {_normalise(q) for q in history}
    last_error = None

    # Retry incomplete generations and failed verifications, but never forever.
    next_qid = 1
    for round_number in range(1, 4):
        current_counts = {
            kind: sum(q["type"] == kind for q in accepted)
            for kind in types
        }
        deficits = {
            kind: target[kind] - current_counts[kind]
            for kind in types
            if target[kind] - current_counts[kind] > 0
        }

        if not deficits:
            break

        required_types = [
            kind
            for kind, count in deficits.items()
            for _ in range(count)
        ]

        # Assign difficulty levels explicitly instead of hardcoding Medium.
        if difficulty.strip().lower() == "mixed":
            difficulty_cycle = ["Easy", "Medium", "Hard"]
            difficulty_levels = [
                difficulty_cycle[i % len(difficulty_cycle)]
                for i in range(len(required_types))
            ]
        else:
            selected_level = difficulty.strip().capitalize()
            if selected_level not in {"Easy", "Medium", "Hard"}:
                raise ValueError(f"Unsupported difficulty: {difficulty}")
            difficulty_levels = [
                selected_level
            ] * len(required_types)

        output_template = []
        for i, kind in enumerate(required_types):
            output_template.append({
                "type": kind,
                "question": "REPLACE_WITH_A_UNIQUE_QUESTION",
                "options": (
                    ["Option A", "Option B", "Option C", "Option D"]
                    if kind == "mcq" else []
                ),
                "answer": "REPLACE_WITH_CORRECT_ANSWER",
                "explanation": "REPLACE_WITH_REASONING",
                "topic": topic,
                "source_id": "C1",
                "difficulty": difficulty_levels[i],
            })


        prompt = f"""
Create assessment questions for the topic: {topic}
Requested difficulty: {difficulty}

Generate ONLY the missing question types and counts specified here:
{json.dumps(deficits)}

Use only the supplied study material. Each question must test a meaningful
concept, not just recall a definition.

Every question must contain:
- type: mcq, short_answer, or numerical
- question: clear, self-contained wording
- options: exactly four distinct options for MCQ; [] otherwise
- answer: correct option text for MCQ; reference answer otherwise
- explanation: reasoning, including calculation steps for numerical questions
- topic: specific subtopic
- source_id: a valid source ID such as C1
- difficulty: Easy, Medium, or Hard

Math formatting rules (apply to the question, every option, answer, and explanation):
- Put every mathematical expression in standard LaTeX delimiters: $...$ inline
  or $$...$$ as a display expression. Do not use raw \\( ... \\), \\[ ... \\], or a
  bare LaTeX command such as \\frac outside delimiters.
- Balance all braces and dollar delimiters. Use readable notation such as
  "$x^2$", "$\\frac{{a}}{{b}}$", and "$\\sqrt{{x}}$".
- MCQ choices must be meaningfully different mathematical values, expressions,
  or claims. Do not use placeholders and do not make choices differ only by
  spacing, punctuation, or an option label.

MCQs must have exactly one correct answer. Numerical questions must be
solvable using the supplied material and state necessary assumptions.
Short-answer questions must have a specific, gradable reference answer.
Do not make up facts, formulas, page numbers, or source IDs.

Return valid JSON only.

The "questions" array MUST contain exactly {len(required_types)} question objects.

The required question types, in order, are:
{json.dumps(required_types)}

Follow this JSON structure. Replace every placeholder with a real,
unique question and its correct answer. Do not copy the placeholders.

{json.dumps({"questions": output_template}, ensure_ascii=False, indent=2)}

Do not return fewer questions than requested. Do not change the required
question types. For MCQs, provide four distinct options and set "answer"
to the exact text of the correct option. For short-answer and numerical
questions, "options" must be an empty list.

Avoid repeating these questions:
{json.dumps(history + [q["question"] for q in accepted], ensure_ascii=False)}

The difficulty field in each question must exactly match the
difficulty assigned to that question in the JSON template.

For Mixed difficulty, use the assigned Easy, Medium, and Hard levels.
Do not label every question Medium.

STUDY MATERIAL:
{context}
"""

        try:
            generated = _extract_json(_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You create accurate, source-grounded university "
                            "assessments. Follow the requested counts and JSON "
                            "schema exactly. Return no prose outside JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model,
                ollama_url=ollama_url,
                temperature=0.1,
                response_format=assessment_schema,
            ))
        except Exception as exc:
            last_error = f"Generation round {round_number}: {exc}"
            continue

        if not isinstance(generated, dict):
            last_error = "The generator did not return a JSON object."
            continue

        candidates = generated.get("questions", [])
        if not isinstance(candidates, list):
            last_error = "The questions field was not a list."
            continue

        pending = []
        pending_counts = current_counts.copy()

        rejections = {
            "invalid_type_or_missing_text": 0,
            "invalid_source_id": 0,
            "duplicate": 0,
            "invalid_options_or_answer": 0,
            "invalid_structure": 0,
        }

        for item in candidates:
            if not isinstance(item, dict):
                rejections["invalid_structure"] += 1
                continue

            qtype = str(item.get("type", "")).strip().lower()
            qtype = qtype.replace("-", "_").replace(" ", "_")

            # Accept common alternative names from the model.
            aliases = {
                "multiple_choice": "mcq",
                "multiplechoice": "mcq",
                "numeric": "numerical",
                "calculation": "numerical",
                "shortanswer": "short_answer",
            }
            qtype = aliases.get(qtype, qtype)

            question = _canonicalise_math_markup(item.get("question", ""))
            answer = _canonicalise_math_markup(item.get("answer", ""))
            source_id = item.get("source_id")

            if (
                qtype not in deficits
                or not question
                or not answer
            ):
                rejections["invalid_type_or_missing_text"] += 1
                continue

            if not isinstance(source_id, str) or source_id not in mapping:
                rejections["invalid_source_id"] += 1
                continue

            if not _has_readable_math_markup(question) or not _has_readable_math_markup(answer):
                rejections["invalid_type_or_missing_text"] += 1
                continue

            if _normalise(question) in seen:
                rejections["duplicate"] += 1
                continue

            options = item.get("options", [])
            if not isinstance(options, list):
                rejections["invalid_structure"] += 1
                continue

            options = [_canonicalise_math_markup(option) for option in options]

            if qtype == "mcq":
                if (
                    len(options) != 4
                    or any(not option for option in options)
                    or not _options_are_meaningful(options)
                    or any(not _has_readable_math_markup(option) for option in options)
                ):
                    rejections["invalid_options_or_answer"] += 1
                    continue

                match = re.match(
                    r"^([A-D])(?:[.)]|\s|$)",
                    answer,
                    re.IGNORECASE,
                )
                if match:
                    answer = options[
                        ord(match.group(1).upper()) - ord("A")
                    ]

                if _normalise(answer) not in {
                    _normalise(option) for option in options
                }:
                    rejections["invalid_options_or_answer"] += 1
                    continue
            else:
                options = []

            chunk = mapping[source_id]
            qid = f"Q{next_qid}"
            next_qid += 1

            pending.append({
                "id": qid,
                "type": qtype,
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": _canonicalise_math_markup(item.get("explanation", "")),
                "topic": str(item.get("topic") or topic).strip(),
                "difficulty": str(item.get("difficulty") or difficulty),
                "source": chunk.get("source", "Unknown"),
                "page": chunk.get("page", "?"),
                "source_text": chunk.get("text", ""),
            })

            pending_counts[qtype] += 1

        if not pending:
            last_error = (
                f"Round {round_number} produced no acceptable candidates. "
                f"Candidate count: {len(candidates)}. "
                f"Rejection counts: {rejections}"
            )
            print(f"[ASSESSMENT DEBUG] {last_error}", flush=True)
            continue

        verification_sources = []
        seen_source_keys = set()
        for q in pending:
            key = (q.get("source"), q.get("page"))
            if key in seen_source_keys:
                continue
            seen_source_keys.add(key)
            verification_sources.append(
                f"[source={q.get('source', 'Unknown')} | page={q.get('page', '?')}]\n"
                f"{q.get('source_text', '')}"
            )
        verification_context = "\n\n---\n\n".join(verification_sources)

        verify_prompt = f"""
Independently verify every candidate against the supplied source excerpts.
Solve the question yourself; do not assume the proposed answer is correct.

For MCQs, check that exactly one option is correct and the answer key matches.
For numerical questions, check the calculation. For short answers, check that
the reference answer is accurate and gradable.

Mark valid=true only when the question, answer, explanation, and source support
are sound. Otherwise mark valid=false and explain why.

Return JSON only:
{{"checks":[
  {{"id":"Q1","valid":true,"reason":"brief reason"}}
]}}

SOURCE EXCERPTS:
{verification_context}

CANDIDATE QUESTIONS:
{json.dumps(pending, ensure_ascii=False)}
"""

        try:
            checked = _extract_json(_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are an independent assessment verifier. "
                            "Return valid JSON and check every candidate."
                        ),
                    },
                    {"role": "user", "content": verify_prompt},
                ],
                model=model,
                ollama_url=ollama_url,
                temperature=0,
                response_format=verification_schema,
            ))
        except Exception as exc:
            last_error = f"Verification round {round_number}: {exc}"
            print(f"[ASSESSMENT WARNING] Verification failed: {exc}", flush=True)

            # Do not regenerate a complete assessment just because the verifier
            # failed. Preserve the generated questions but mark them unverified.
            for q in pending:
                q["verified"] = False
                q["verification_note"] = "Verification could not be completed."
                accepted.append(q)
                seen.add(_normalise(q["question"]))
            break

        if not isinstance(checked, dict):
            last_error = "The verifier did not return a JSON object."
            continue

        checks_list = checked.get("checks", [])
        if not isinstance(checks_list, list):
            last_error = "The verifier did not return a checks list."
            continue

        checks = {
            item.get("id"): item
            for item in checks_list
            if isinstance(item, dict)
        }

        for q in pending:
            check = checks.get(q["id"], {})

            if check.get("valid") is True:
                q["verified"] = True
                q["verification_note"] = str(
                    check.get("reason", "Passed verification.")
                )
                accepted.append(q)
                seen.add(_normalise(q["question"]))
            elif check.get("valid") is not False:
                # If the verifier omitted this ID, retain the structurally valid
                # question with an explicit unverified flag rather than silently
                # losing it and ending with an empty assessment.
                q["verified"] = False
                q["verification_note"] = "Verifier did not return a decision for this question."
                accepted.append(q)
                seen.add(_normalise(q["question"]))

    final_counts = {
        kind: sum(q["type"] == kind for q in accepted)
        for kind in types
    }

    if len(accepted) != num_questions or final_counts != target:
        raise ValueError(
            f"Could not generate the required assessment after 3 attempts. "
            f"Requested {num_questions} questions with distribution {target}; "
            f"verified {len(accepted)} with distribution {final_counts}. "
            f"Try again or use more relevant source material. "
            f"Last issue: {last_error or 'Some questions failed validation.'}"
        )

    return accepted


def evaluate_assessment_answers(
    questions,
    student_answers,
    model="qwen2.5:3b",
    ollama_url="http://localhost:11434",
):
    results = []
    subjective = []

    # 1. Grade MCQs deterministically; send other valid responses for grading.
    for q in questions:
        student = str(student_answers.get(q["id"], "")).strip()
        qtype = q["type"]

        # Handle unanswered questions safely.
        if not student:
            results.append({
                "question_id": q["id"],
                "correct": False,
                "student_answer": student,
                "reference_answer": q["answer"],
                "feedback": (
                    f"No answer was provided. "
                    f"Reference answer: {q['answer']}. "
                    f"{q.get('explanation', '')}"
                ),
                "topic": q["topic"],
                "source": q["source"],
                "page": q["page"],
            })
            continue

        if qtype == "mcq":
            correct = (
                _normalise(student) == _normalise(q["answer"])
            )

            results.append({
                "question_id": q["id"],
                "correct": correct,
                "student_answer": student,
                "reference_answer": q["answer"],
                "feedback": (
                    "Correct. " + q.get("explanation", "")
                    if correct
                    else (
                        f"Incorrect. The correct answer is "
                        f"{q['answer']}. "
                        f"{q.get('explanation', '')}"
                    )
                ),
                "topic": q["topic"],
                "source": q["source"],
                "page": q["page"],
            })

        elif (
            qtype == "numerical"
            and re.fullmatch(r"[A-Da-d]", student)
        ):
            # A bare MCQ letter is not a valid numerical solution.
            results.append({
                "question_id": q["id"],
                "correct": False,
                "student_answer": student,
                "reference_answer": q["answer"],
                "feedback": (
                    "Incorrect. Enter the numerical result or "
                    "mathematical expression, not an option letter. "
                    f"Reference answer: {q['answer']}. "
                    f"{q.get('explanation', '')}"
                ),
                "topic": q["topic"],
                "source": q["source"],
                "page": q["page"],
            })

        else:
            # IMPORTANT: collect responses for model-based grading.
            subjective.append((q, student))

    # 2. Grade short-answer and numerical questions in one model call.
    if subjective:
        grading_items = [
            {
                "question_id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "reference_answer": q["answer"],
                "explanation": q.get("explanation", ""),
                "student_answer": student,
                "source": q["source"],
                "page": q["page"],
                "source_text": q.get("source_text", ""),
            }
            for q, student in subjective
        ]

        prompt = f"""
Grade every student response against its question, reference answer,
explanation, and supplied source material.

Rules:
- Accept mathematically equivalent expressions and correct reasoning.
- Do not mark an answer correct merely because it sounds plausible.
- For numerical answers, check the mathematical result and reasoning.
- If the answer is incomplete or mathematically incorrect, mark it false.
- Give concise feedback explaining the decision.
- Return exactly one result for every supplied question_id.
- The "correct" field MUST be a JSON boolean: true or false.

Return valid JSON only, in this format:
{{
  "results": [
    {{
      "question_id": "Q2",
      "correct": false,
      "feedback": "Explain the result.",
      "reference_answer": "The expected answer"
    }}
  ]
}}

QUESTIONS AND STUDENT ANSWERS:
{json.dumps(grading_items, ensure_ascii=False)}
"""

        graded = _extract_json(_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a careful university examiner. "
                        "Grade each response independently and accurately. "
                        "Return valid JSON only, with Boolean correctness."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            ollama_url=ollama_url,
            temperature=0,
        ))

        if not isinstance(graded, dict):
            raise ValueError("The grading model returned invalid JSON.")

        returned = graded.get("results", [])
        if not isinstance(returned, list):
            raise ValueError("The grading model did not return a results list.")

        by_id = {
            item.get("question_id"): item
            for item in returned
            if isinstance(item, dict)
        }

        for q, student in subjective:
            item = by_id.get(q["id"])

            if (
                item is None
                or not isinstance(item.get("correct"), bool)
                or not isinstance(item.get("feedback"), str)
            ):
                raise ValueError(
                    f"Invalid or missing grading result for {q['id']}. "
                    "Please submit the assessment again."
                )

            results.append({
                "question_id": q["id"],
                "correct": item["correct"],
                "student_answer": student,
                "reference_answer": item.get(
                    "reference_answer", q["answer"]
                ),
                "feedback": item["feedback"],
                "topic": q["topic"],
                "source": q["source"],
                "page": q["page"],
            })

    # 3. Restore the original question order and check completeness.
    order = {q["id"]: i for i, q in enumerate(questions)}
    results.sort(key=lambda r: order[r["question_id"]])

    if len(results) != len(questions):
        raise ValueError(
            f"Grading incomplete: expected {len(questions)} results, "
            f"received {len(results)}."
        )

    return results
