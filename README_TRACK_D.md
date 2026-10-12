# StudyPilot AI — Track D upgrade

This prototype is aligned to the Multimodal AI Hackathon 2026 Track D requirements.

## What is implemented

- **Multimodal knowledge base:** PDF/text extraction with optional local vision recovery; lecture-video frame sampling with timestamps via FFmpeg + Ollama vision.
- **Source preservation:** page/source metadata is retained on chunks and questions. The Sources tab renders the exact PDF page for inspection.
- **Topic organization:** lightweight local course-map extraction from source headings/concepts.
- **Source-grounded tutor:** retrieval-first tutor prompts with explicit unsupported-content handling and source/page citations.
- **Adaptive assessment:** MCQ, short-answer, and numerical questions; topic/source/difficulty tags; duplicate history; answer verification; answer feedback.
- **Learner model:** Bayesian-style Beta posterior mastery estimate per topic, updated after assessment items and tutor comprehension feedback.
- **New learner flow:** diagnostic assessment prompt when no history exists.
- **Dashboard:** course status, mastery, weak-topic recommendations, sources, assessment reports, and evaluation.
- **System evaluation:** transparent retrieval precision/recall benchmark and personalization gain/repetition metrics, with a path to formal RAGAS/DeepEval/TruLens reporting.

## Run

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For video ingestion, install FFmpeg and make sure `ffmpeg` and `ffprobe` are on PATH.

For local models:

```bash
ollama pull qwen3.5:4b
ollama pull qwen2.5vl:3b
```

## Important demo notes

- Use **PDF vision mode = auto** by default. It only sends pages with poor text extraction to the vision model; equation-heavy pages do not automatically trigger expensive vision calls.
- Video ingestion samples frames rather than transcribing every second. This keeps a local demo practical and preserves timestamped source locations.
- The evaluation panel is a transparent prototype benchmark. For the final hackathon report, run a fixed team-built test set and report formal framework metrics where available.
