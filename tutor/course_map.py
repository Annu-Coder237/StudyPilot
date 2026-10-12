
import re
from collections import Counter


def _clean_line(value):
    """Normalize a line and reject obvious OCR fragments."""
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = value.strip(" \t•·-–—|:")

    if not value:
        return ""

    letters = re.findall(r"[A-Za-z]", value)
    words = re.findall(r"[A-Za-z]{2,}", value)

    # Reject labels with too little meaningful language.
    if len(letters) < 5 or len(words) == 0:
        return ""

    # Reject short uppercase fragments such as "E MEK" or "YEN".
    if len(value) <= 24 and value.upper() == value:
        return ""

    # Reject lines dominated by symbols or isolated characters.
    if len(words) < 2 and len(value) < 16:
        return ""

    # Reject likely OCR noise: too many isolated single-letter tokens.
    tokens = value.split()
    if len(tokens) >= 4:
        single_letter_ratio = sum(
            1 for token in tokens if re.fullmatch(r"[A-Za-z]", token)
        ) / len(tokens)
        if single_letter_ratio >= 0.5:
            return ""

    return value

def clean_course_map_text(text):
    """Remove formatting commands and obvious OCR noise from display text."""
    text = str(text or "")

    # Remove LaTeX environments and document-formatting commands.
    text = re.sub(
        r"\\(?:begin|end)\{(?:equation|align|gather|displaymath)\*?\}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\\(?:section|subsection|subsubsection)\*?\{([^{}]*)\}",
        r"\1",
        text,
    )

    # Convert common text wrappers into readable text.
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", text)

    # Remove remaining LaTeX commands that are mostly formatting.
    text = re.sub(
        r"\\(?:left|right|displaystyle|textstyle|quad|qquad|"
        r"big|Big|bigg|Bigg)\b",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()



def build_course_map(chunks):
    """
    Build a readable course map from extracted chunks.
    Prefer meaningful headings and concepts over arbitrary OCR fragments.
    """
    topic_counts = Counter()
    concepts = {}
    topic_pages = {}

    heading_pattern = re.compile(
        r"^(?:lecture\s+\d+|chapter\s+\d+|"
        r"\d+(?:\.\d+)*[.)]?\s+[A-Z][\w\s,&()/+-]{3,})$",
        re.IGNORECASE,
    )

    for chunk in chunks:
        text = clean_course_map_text(chunk.get("text", ""))
        if not text:
            continue

        lines = [
            cleaned
            for raw_line in text.splitlines()
            if (cleaned := _clean_line(raw_line))
        ]

        if not lines:
            continue

        # Prefer a recognizable lecture/chapter heading.
        heading = next(
            (
                line for line in lines[:12]
                if heading_pattern.match(line)
                and len(line) >= 8
            ),
            None,
        )

        # Otherwise use a meaningful descriptive line, not arbitrary uppercase text.
        if heading is None:
            heading = next(
                (
                    line for line in lines[:8]
                    if len(line) >= 18
                    and len(re.findall(r"[A-Za-z]{2,}", line)) >= 3
                    and not line.lower().startswith(
                        ("example:", "problem:", "solution:")
                    )
                ),
                None,
            )

        if heading is None:
            # Do not invent a topic from a noisy page.
            continue

        heading = heading.rstrip(" :.")
        topic_counts[heading] += 1
        concepts.setdefault(heading, [])
        topic_pages.setdefault(heading, set())

        page = chunk.get("page")
        if page is not None:
            topic_pages[heading].add(page)

        for line in lines:
            line = clean_course_map_text(line)

            if line == heading or len(line) < 18:
                continue

            # Ignore lines that still contain raw LaTeX markup.
            if re.search(
                r"\\(?:begin|end|text|frac|sum|int|section|subsection)\b",
                line,
            ):
                continue

            # Require enough meaningful words to be a useful concept.
            words = re.findall(r"[A-Za-z]{2,}", line)
            if len(words) < 3:
                continue

            # Skip likely OCR fragments dominated by isolated characters.
            tokens = line.split()
            if tokens:
                isolated = sum(
                    len(token) == 1 and token.isalpha()
                    for token in tokens
                )
                if isolated / len(tokens) > 0.4:
                    continue

            # Avoid repeating the same concept.
            if line.casefold() not in {
                existing.casefold() for existing in concepts[heading]
            }:
                concepts[heading].append(line[:180])

    topics = []
    for name, count in topic_counts.most_common(12):
        topics.append({
            "topic": name,
            "concepts": concepts[name][:4],
            "occurrences": count,
            "pages": sorted(
                topic_pages[name],
                key=lambda value: (
                    0, int(value)
                ) if str(value).isdigit() else (1, str(value)),
            ),
            "prerequisites": [],
        })

    return topics

