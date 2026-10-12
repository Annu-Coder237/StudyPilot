import os
import re
from pathlib import Path

import fitz
import requests
import time



def should_recover_page(text: str, has_images: bool = False) -> bool:
    """Use vision only when native PDF text extraction is clearly inadequate."""
    text = (text or "").strip()

    # Empty or nearly empty pages are strong recovery candidates.
    if len(text) < 30:
        return True

    # Replacement characters indicate potentially corrupted text extraction.
    if "\ufffd" in text:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = re.findall(r"[A-Za-z0-9]{2,}", text)

    # Flag pages where text is fragmented into many isolated glyphs.
    if (
        len(lines) >= 12
        and len([line for line in lines if len(line) <= 5]) / len(lines) >= 0.70
    ):
        return True

    # Do not invoke vision merely because a page contains an image.
    # Sparse text on an image-heavy page is not, by itself, proof of failure.
    return False



def _ordered_text_blocks(page):
    """Return readable text blocks in page order without flattening paragraphs."""
    blocks = []
    for index, block in enumerate(page.get_text("blocks", sort=True)):
        x0, y0, x1, y1, text, block_no, block_type = block[:7]
        if block_type != 0 or not text or not text.strip():
            continue
        # Keep line breaks within a text block, but remove extraction-only
        # whitespace around them.  Blocks remain separated by a blank line.
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if cleaned:
            blocks.append({
                "index": index,
                "bbox": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)],
                "text": cleaned,
                "pdf_block_number": block_no,
            })
    return blocks


def _ollama_models(ollama_url: str) -> set[str]:
    response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
    response.raise_for_status()
    return {item.get("name", "") for item in response.json().get("models", [])}


def recover_page_with_vision(page, model: str, ollama_url: str = "http://localhost:11434", dpi: int = 100,timeout_seconds: int = 90,) -> str:
    """Render a PDF page and transcribe it using a local Ollama vision model."""
    scale = dpi / 72
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image_b64 = __import__("base64").b64encode(pixmap.tobytes("png")).decode("ascii")
    
    prompt = """
        You are transcribing a page from an academic Signals and Systems lecture.

        Transcribe the visible page faithfully. Do not summarize, explain, or invent text.

        Rules:
        1. Read the page in its natural reading order, preserving paragraph boundaries.
        2. Preserve equations, subscripts, superscripts, Greek letters, indices,
        inequalities, and mathematical operators exactly as visible.
        3. Write equations in valid LaTeX, enclosed in \\( ... \\) for inline
        mathematics or \\[ ... \\] for display equations.
        4. Preserve headings, numbered questions, examples, and solution steps.
        5. Keep diagram labels near their relevant figure descriptions.
        6. Do not turn unclear symbols into guessed words. Use [uncertain]
        when a symbol cannot be read confidently.
        7. Do not merge separate text fragments into meaningless sentences.
        8. Avoid repeating the same line.

        Return only the transcription. Do not include commentary.
        """

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "options": {"temperature": 0},
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    text = response.json().get("message", {}).get("content", "").strip()
    if not text:
        raise ValueError("The local vision model returned empty text.")
    return text


def extract_pdf_pages(
    pdf_input,
    source_name=None,
    client=None,  # Kept only for compatibility with earlier calls; never used.
    model=None,
    vision_mode="none",
    dpi=160,
    ollama_url=None,
    vision_model=None,
):
    """
    Extract each PDF page locally.

    vision_mode='none': PyMuPDF text extraction only.
    vision_mode='auto': use local vision only on likely equation/problem pages.
    vision_mode='all': try local vision on every page.
    """
    if vision_mode not in {"auto", "all", "none"}:
        raise ValueError("vision_mode must be 'auto', 'all', or 'none'.")

    if isinstance(pdf_input, (bytes, bytearray)):
        pdf_bytes = bytes(pdf_input)
    elif hasattr(pdf_input, "getvalue"):
        pdf_bytes = pdf_input.getvalue()
    elif isinstance(pdf_input, (str, Path)):
        path = Path(pdf_input)
        pdf_bytes = path.read_bytes()
        source_name = source_name or path.name
    elif hasattr(pdf_input, "read"):
        pdf_bytes = pdf_input.read()
    else:
        raise TypeError("pdf_input must be PDF bytes, a file path, or an uploaded file object.")

    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    source_name = source_name or getattr(pdf_input, "name", "uploaded.pdf")
    ollama_url = (ollama_url or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    vision_model = vision_model or model or os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")

    available_models = set()
    vision_available = False
    if vision_mode != "none":
        try:
            available_models = _ollama_models(ollama_url)
            vision_available = (
                vision_model in available_models
                or f"{vision_model}:latest" in available_models
            )
        except requests.RequestException:
            # Ordinary text extraction should still work if Ollama is offline.
            vision_available = False

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    try:
        total_pages = len(document)

        print(
            f"\n[INGESTION] Starting {source_name}: "
            f"{total_pages} pages | mode={vision_mode} | "
            f"model={vision_model} | vision_available={vision_available}",
            flush=True,
            )

        for number, pdf_page in enumerate(document, start=1):
            page_start = time.perf_counter()

            # raw_text is retained verbatim for audit/review.  The ordered form
            # is used for display and retrieval so paragraphs do not collapse.
            raw_text = pdf_page.get_text("text", sort=True).strip()
            text_blocks = _ordered_text_blocks(pdf_page)
            ordered_text = "\n\n".join(block["text"] for block in text_blocks).strip()
            extracted_text = ordered_text or raw_text

            print(f"\n[DEBUG] Page {number} native text:")
            print(extracted_text[:1500])

            has_images = bool(pdf_page.get_images(full=True))

            should_try = (
                vision_mode == "all"
                or (
                    vision_mode == "auto"
                    and should_recover_page(
                        extracted_text,
                        has_images=has_images,
                    )
                )
            )

            print(
                f"[INGESTION] Page {number}/{total_pages}: "
                f"{len(extracted_text)} extracted characters | "
                f"vision_required={should_try}",
                flush=True,
            )
            record = {
                "page": number,
                "source": source_name,
                "raw_text": raw_text,
                "text": extracted_text,
                "text_blocks": text_blocks,
                "extraction_metadata": {
                    "text_block_count": len(text_blocks),
                    "has_images": has_images,
                    "text_order": "top-to-bottom PDF blocks",
                },
                "vision_text": "",
                "extraction_method": "pdf_text",
                "recovery_status": "not_needed",
                "vision_error": "",
            }

            if should_try:
                if not vision_available:
                    record["recovery_status"] = "skipped_model_unavailable"
                    record["vision_error"] = (
                        f"Local vision model '{vision_model}' is not available or Ollama is offline. "
                        "Original PDF text was retained."
                        )
                else:
                    vision_start = time.perf_counter()

                    print(
                        f"[VISION START] Page {number}/{total_pages}",
                        flush=True,
                    )

                    try:
                        
                        recovered = None
                        first_error = None

                        # First attempt uses the configured DPI.
                        # If it fails, retry once with a smaller image.
                        attempts = [
                            (dpi, 150),
                            (max(90, dpi // 2), 150),
                        ]

                        for attempt_number, (attempt_dpi, timeout) in enumerate(
                            attempts, start=1
                        ):
                            try:
                                print(
                                    f"[VISION ATTEMPT] Page {number}/{total_pages} "
                                    f"| attempt={attempt_number} "
                                    f"| dpi={attempt_dpi}",
                                    flush=True,
                                )

                                recovered = recover_page_with_vision(
                                    pdf_page,
                                    model=vision_model,
                                    ollama_url=ollama_url,
                                    dpi=attempt_dpi,
                                    timeout_seconds=timeout,
                                )

                                print(f"\n[DEBUG] Page {number} vision text:")
                                print(recovered[:1500])

                                if attempt_number > 1:
                                    print(
                                        f"[VISION RECOVERED] Page {number} "
                                        "succeeded on retry.",
                                        flush=True,
                                    )
                                break

                            except Exception as exc:
                                if first_error is None:
                                    first_error = exc

                                print(
                                    f"[VISION ATTEMPT FAILED] Page {number} "
                                    f"| attempt={attempt_number} "
                                    f"| {type(exc).__name__}: {exc}",
                                    flush=True,
                                )

                        if recovered:
                            record["vision_text"] = recovered
                            record["extraction_method"] = "pdf_text+ollama_vision"
                            record["recovery_status"] = "success"
                                                    
                        else:
                            record["recovery_status"] = "failed"
                            record["vision_error"] = (
                                f"Vision recovery failed after {len(attempts)} attempts. "
                                f"First error: {type(first_error).__name__}: {first_error}. "
                                "Original PDF text was retained."
                            )


                    except Exception as e:
                        record["recovery_status"] = "failed"
                        record["vision_error"] = f"{type(e).__name__}: {e}"

                        print(
                            f"[VISION ERROR] Page {number}/{total_pages}: "
                            f"{type(e).__name__}: {e}",
                            flush=True,
                        )

                    finally:
                        elapsed = time.perf_counter() - vision_start

                        print(
                            f"[VISION END] Page {number}/{total_pages} | "
                            f"elapsed={elapsed:.1f}s | "
                            f"status={record['recovery_status']}",
                            flush=True,
                        )
            pages.append(record)
    finally:
        document.close()
    return pages


def chunk_pages(pages, chunk_size=1200, overlap=200):
    """Split page text into overlapping chunks while retaining source/page metadata."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    chunks = []
    for page in pages:
        base_text = (page.get("text") or "").strip()
        vision_text = (page.get("vision_text") or "").strip()

        
        text = vision_text if vision_text else base_text

        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": len(chunks),
                    "text": piece,
                    "page": page.get("page", "?"),
                    "source": page.get("source", "Unknown source"),
                    "char_count": len(piece),
                })
            if end == len(text):
                break
            start = end - overlap
    return chunks

