import requests
import re


def check_ollama_connection(ollama_url="http://localhost:11434", model="qwen3.5:4b"):
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = [item.get("name", "") for item in response.json().get("models", [])]
        if not models:
            return False, "Ollama is responding, but no models are installed. Run `ollama pull qwen3.5:4b`."
        if model not in models and f"{model}:latest" not in models:
            return False, f"Ollama is running, but '{model}' was not found. Installed models: {', '.join(models)}"
        return True, f"Ollama is ready and '{model}' is installed."
    except requests.RequestException as exc:
        return False, f"Cannot reach Ollama at {ollama_url}. Open Ollama or run `ollama serve`. Details: {exc}"

def normalize_math_delimiters(text: str) -> str:
    """Convert LaTeX math delimiters to Streamlit-friendly delimiters."""
    text = text.replace(r"\[", "$$")
    text = text.replace(r"\]", "$$")
    text = text.replace(r"\(", "$")
    text = text.replace(r"\)", "$")
    return text


def _chat(
    messages,
    model,
    ollama_url,
    temperature=0.2,
    response_format=None,
):
    url = f"{ollama_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature},
    }

    if response_format is not None:
        payload["format"] = response_format

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=240,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not call local Ollama model '{model}' at {ollama_url}. "
            f"Check that Ollama is running and the model is installed. Details: {exc}"
        ) from exc

    answer = response.json().get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    return answer


def generate_tutor_answer(question, retrieved_chunks, model="qwen3.5:4b", ollama_url="http://localhost:11434"):
    if not retrieved_chunks:
        raise ValueError("No retrieved source passages were provided.")
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk.get('source', 'Unknown source')}, page {chunk.get('page', '?')}]\n"
            f"{chunk.get('text', '')}"
        )
    context = "\n\n---\n\n".join(context_parts)
    
    system = (
        "You are a careful university-level AI study tutor. "
        "Use the provided study material as the primary source. "
        "Answer the student's exact question and explain the reasoning "
        "step by step when useful.\n\n"

        "MATHEMATICAL ACCURACY:\n"
        "- Preserve the distinction between discrete-time and continuous-time signals.\n"
        "- Check equations for mathematical consistency before explaining them.\n"
        "- Define symbols when needed and explain why each equality holds.\n"
        "- Never invent a formula, derivation, source quotation, or page reference.\n"
        "- If OCR or source notation is ambiguous, explicitly identify the uncertainty.\n"
        "- If the source appears incorrect or incomplete, say so rather than silently "
        "correcting it or presenting it as verified.\n\n"

        "FORMATTING:\n"
        "- Return a well-organized answer using valid Markdown.\n"
        "- Use ## headings for meaningful sections, not for every short paragraph.\n"
        "- Use numbered steps for derivations and bullet points for lists.\n"
        "- Put standalone equations on separate lines between $$ delimiters.\n"
        "- Use $...$ for inline mathematical expressions.\n"
        "- Never wrap display equations in square brackets.\n"
        "- Use valid LaTeX commands, such as \\delta, \\sum, \\infty, "
        "\\text{}, \\frac{}{}, and \\begin{aligned} where appropriate.\n"
        "- Explain equations in ordinary prose outside the math delimiters.\n"
        "- End with a short key takeaway when it helps learning.\n\n"
        "- Use ONLY $...$ for inline mathematics.\n"
        "- Use ONLY $$...$$ for display equations.\n"
        "- Never use \\(...\\) or \\[...\\] as math delimiters.\n"
        "- Every equation must have balanced delimiters and valid LaTeX.\n"
        "- Write x[n] with both square brackets intact.\n"
        "- For the discrete-time sifting property, the formula is:\n"
        "  $$x[n]\\delta[n-k]=x[k]\\delta[n-k]$$\n"
        "- Do not change mathematical symbols or omit indices.\n"

        "SOURCES:\n"
        "- Cite only the supplied source filenames and page numbers.\n"
        "- Clearly distinguish source-supported statements from general background.\n"
        "- Do not claim that a page or equation was visually verified unless "
        "the supplied context supports that claim."
    )

    user = (
    f"STUDY MATERIAL:\n{context}\n\n"
    f"STUDENT QUESTION:\n{question}\n\n"
    "Write a clear, accurate, well-formatted study explanation. "
    "Check mathematical notation carefully and cite the relevant source pages. "
    "If the supplied material does not establish a formula, state that limitation."
)
    answer = _chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        model=model,
        ollama_url=ollama_url,
    )

    return normalize_math_delimiters(answer)
