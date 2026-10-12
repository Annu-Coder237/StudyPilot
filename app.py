import json
import re
from pathlib import Path

import fitz
import streamlit as st

from tutor.ingestion import extract_pdf_pages, chunk_pages
from tutor.retrieval import retrieve_relevant_chunks
from tutor.generation import generate_tutor_answer, check_ollama_connection
from tutor.assessment import generate_assessment, evaluate_assessment_answers
from tutor.learner_model import (
    record_attempt,
    record_conversation_feedback,
    get_learning_summary,
    recommend_topics,
)
from tutor.course_map import build_course_map
from tutor.evaluation import evaluate_retrieval, personalization_report

try:
    from tutor.video_ingestion import ingest_video
    VIDEO_AVAILABLE = True
except Exception:
    VIDEO_AVAILABLE = False

st.set_page_config(page_title="StudyPilot AI", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# ---------- Theme ----------
st.markdown("""
<style>
:root { --accent:#6d5dfc; --soft:#f5f3ff; }
.block-container { max-width: 1450px; padding-top: 1.4rem; }
.hero { padding: 1.4rem 1.6rem; border-radius: 20px; background: linear-gradient(135deg,#f4f1ff,#eef7ff); border:1px solid #e6e1ff; margin-bottom:1rem; }
.hero h1 { margin:0; font-size:2.25rem; }
.hero p { margin:.35rem 0 0; color:#5d6470; font-size:1.02rem; }
.card { padding:1rem 1.15rem; border:1px solid #e8e8ee; border-radius:16px; background:white; min-height:105px; }
.card-title { color:#656b76; font-size:.88rem; }
.card-value { font-size:1.65rem; font-weight:700; margin-top:.25rem; }
.source-chip { display:inline-block; padding:.2rem .55rem; border-radius:999px; background:#f1efff; color:#5548c8; font-size:.78rem; margin:.1rem; }
.weak { border-left:4px solid #e45757; padding:.6rem .8rem; background:#fff7f7; border-radius:8px; margin:.4rem 0; }
.good { border-left:4px solid #3ca66b; padding:.6rem .8rem; background:#f5fff9; border-radius:8px; margin:.4rem 0; }
.small { color:#6b7280; font-size:.85rem; }

/* Course map: compact, consistent typography */
.map-topic {
    font-size: 1.05rem;
    font-weight: 650;
    line-height: 1.4;
    letter-spacing: 0;
    overflow-wrap: anywhere;
    margin: 0 0 0.8rem 0;
    color: var(--text-color);
}

.map-concept {
    font-size: 0.91rem;
    font-weight: 400;
    line-height: 1.55;
    letter-spacing: 0;
    overflow-wrap: anywhere;
    margin: 0.45rem 0;
    color: var(--text-color);
}

.map-dot {
    color: #8b82ff;
    font-weight: 700;
    margin-right: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


def reset_document_state():
    for key in (
        "study_pages", "study_chunks", "study_filename", "pdf_bytes", "retrieved_chunks",
        "tutor_answer", "retrieval_question", "active_assessment", "assessment_results",
        "assessment_topic_active", "assessment_chunks", "course_map", "video_records",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith("answer_Q"):
            del st.session_state[key]


def render_source_page(pages, source, page_no):
    for page in pages:
        if page.get("source") == source and str(page.get("page")) == str(page_no):
            try:
                doc = fitz.open(stream=st.session_state.get("pdf_bytes", b""), filetype="pdf")
                pix = doc[int(page_no) - 1].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
                st.image(pix.tobytes("png"), caption=f"{source} — page {page_no}")
                doc.close()
            except Exception as exc:
                st.warning(f"Could not render the page preview: {exc}")
            return


def source_badge(chunk):
    return f"{chunk.get('source','Unknown')} · p.{chunk.get('page','?')}"


def math_markdown(text):
    """Normalise common LaTeX delimiters before handing text to Streamlit."""
    text = str(text or "")
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text, flags=re.DOTALL)
    return re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.DOTALL)


def render_rich_text(text):
    """Render paragraphs and valid LaTeX instead of showing escaped source text."""
    text = math_markdown(text).strip()
    if not text:
        st.caption("No text was extracted from this page.")
        return
    for paragraph in re.split(r"\n\s*\n", text):
        st.markdown(paragraph)


def option_letters(count):
    return [chr(ord("A") + index) for index in range(count)]


# ---------- State ----------
if "learning_attempts" not in st.session_state:
    st.session_state["learning_attempts"] = []
if "topic_mastery" not in st.session_state:
    st.session_state["topic_mastery"] = {}

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎓 StudyPilot AI")
    st.caption("Personalized, source-grounded learning")
    st.divider()

    subject = st.selectbox("Course", ["Signals & Systems", "Digital System Design", "Device Electronics", "Other"], key="course")
    ollama_url = st.text_input("Ollama URL", "http://localhost:11434")
    text_model = st.text_input("Text model", "qwen3.5:4b")
    vision_model = st.text_input("Vision model", "qwen2.5vl:3b")

    if st.button("Check local AI", use_container_width=True):
        ok, msg = check_ollama_connection(ollama_url, text_model)
        (st.success if ok else st.error)(msg)

    summary = get_learning_summary(st.session_state)
    st.divider()
    st.markdown("### Your progress")
    if summary["attempts"] == 0:
        st.info("No quiz history yet. Start with a diagnostic assessment.")
    else:
        st.metric("Mastery", f"{summary['accuracy']:.0%}")
        st.caption(f"{summary['attempts']} answered questions")
        for topic, mastery in sorted(summary["topic_mastery"].items(), key=lambda x: x[1])[:3]:
            st.progress(float(mastery), text=f"{topic[:28]} · {mastery:.0%}")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <h1>🎓 StudyPilot AI</h1>
  <p>Your course material → grounded tutor → adaptive practice → measurable mastery.</p>
</div>
""", unsafe_allow_html=True)

pages = st.session_state.get("study_pages", [])
chunks = st.session_state.get("study_chunks", [])
summary = get_learning_summary(st.session_state)


ready_notice = st.session_state.pop("document_ready_notice", None)

if ready_notice:
    st.success(ready_notice)


# ---------- Dashboard ----------
tab_home, tab_tutor, tab_assess, tab_sources, tab_progress, tab_eval = st.tabs([
    "🏠 Dashboard", "💬 Tutor", "📝 Assess", "📚 Sources", "📈 Progress", "🧪 Evaluation"
])

with tab_home:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="card"><div class="card-title">Course material</div><div class="card-value">{len(pages)} pages</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card"><div class="card-title">Knowledge chunks</div><div class="card-value">{len(chunks)}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card"><div class="card-title">Questions answered</div><div class="card-value">{summary["attempts"]}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card"><div class="card-title">Overall accuracy</div><div class="card-value">{summary["accuracy"]:.0%}</div></div>', unsafe_allow_html=True)

    st.markdown("### Build your course knowledge base")
    st.caption("Upload slides/textbooks now. Video frames can be sampled into timestamped source units when FFmpeg is available.")
    col1, col2 = st.columns(2)
    with col1:
        pdf = st.file_uploader("📄 Lecture slides / textbook", type=["pdf"], key="pdf_upload")
        vision_mode = st.selectbox(
            "PDF visual recovery",
            ["none", "auto", "all"],
            index=0,
            help="None: use native PDF text only. Auto: recover pages with poor extraction. All: run vision on every page."
        )
        
        if pdf and st.button(
            "Process course PDF",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner("Building your source-grounded knowledge base..."):
                    pdf_bytes_new = pdf.getvalue()

                    pages_new = extract_pdf_pages(
                        pdf_bytes_new,
                        source_name=pdf.name,
                        vision_mode=vision_mode,
                        ollama_url=ollama_url,
                        vision_model=vision_model,
                    )
                    chunks_new = chunk_pages(pages_new)
                    course_map_new = build_course_map(chunks_new)

                if not chunks_new:
                    st.error(
                        "PDF processing finished, but no usable text chunks "
                        "were extracted. Check the PDF extraction logs."
                    )
                else:
                    reset_document_state()

                    st.session_state.update({
                        "study_pages": pages_new,
                        "study_chunks": chunks_new,
                        "study_filename": pdf.name,
                        "pdf_bytes": pdf_bytes_new,
                        "course_map": course_map_new,
                        "document_ready_notice": (
                            f"Knowledge base ready: {len(pages_new)} pages, "
                            f"{len(chunks_new)} chunks."
                        ),
                    })

                    # Rerun the app so every tab reads the newly stored data.
                    st.rerun()

            except Exception as exc:
                st.error(f"PDF ingestion failed: {type(exc).__name__}: {exc}")


    with col2:
        video = st.file_uploader("🎥 Lecture video", type=["mp4", "mov", "mkv", "webm"], key="video_upload")
        interval = st.slider("Frame sampling interval (seconds)", 10, 90, 30)
        if video and st.button("Ingest video frames", use_container_width=True):
            if not VIDEO_AVAILABLE:
                st.error("Video ingestion module is unavailable.")
            else:
                try:
                    with st.spinner("Sampling lecture frames and extracting timestamped content..."):
                        records = ingest_video(video.getvalue(), video.name, vision_model, ollama_url, interval=interval, max_frames=12)
                    video_chunks = [dict(r, char_count=len(r.get("text", ""))) for r in records if r.get("text")]
                    st.session_state["video_records"] = records
                    st.session_state["study_chunks"] = chunks + video_chunks
                    st.success(f"Added {len(video_chunks)} timestamped video source units.")
                except Exception as exc:
                    st.error(f"Video ingestion failed: {type(exc).__name__}: {exc}. Install FFmpeg and ensure the vision model is available.")

    
    if pages:
        st.markdown("### Course map")
        st.caption(
            "A structured overview of topics and concepts found in your uploaded material."
        )

        cmap = st.session_state.get("course_map") or build_course_map(chunks)

        def clean_map_text(value):
            """Remove obvious OCR fragments before displaying map labels."""
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            text = text.strip(" •·-–—|:")
            if not text:
                return ""

            # Reject short, mostly uppercase OCR fragments such as "E MEK".
            letters = re.sub(r"[^A-Za-z]", "", text)
            if len(letters) < 4:
                return ""
            if len(text) <= 18 and letters.isupper():
                return ""

            # Avoid labels that are mostly symbols or stray punctuation.
            if sum(ch.isalpha() for ch in text) / max(len(text), 1) < 0.45:
                return ""

            return text

        if not cmap:
            st.info(
                "A reliable course map could not be generated from the extracted "
                "text. Check the source pages for extraction quality."
            )
        else:
            # Use a two-column layout for more readable cards.
            for start in range(0, min(len(cmap), 8), 2):
                row_items = cmap[start:start + 2]
                columns = st.columns(2, gap="medium")

                for col, item in zip(columns, row_items):
                    with col:
                        topic = clean_map_text(item.get("topic", ""))
                        concepts = [
                            clean_map_text(concept)
                            for concept in item.get("concepts", [])
                        ]
                        concepts = list(dict.fromkeys(
                            concept for concept in concepts if concept
                        ))[:4]

                        # Do not show cards with no trustworthy content.
                        if not topic and not concepts:
                            continue

                        with st.container(border=True):
                            if topic:
                                st.markdown(
                                    f'<div class="map-topic">{topic}</div>',
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown("**Extracted concepts**")

                            if concepts:
                                for concept in concepts:
                                    st.markdown(
                                        f'<div class="map-concept">'
                                        f'<span class="map-dot">•</span> '
                                        f'{concept}</div>',
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption(
                                    "No clear concepts were detected for this section."
                                )


    if not pages and not chunks:
        st.info("Start by uploading your lecture slides or textbook PDF. The Tutor, Assess, Sources, and Progress areas will unlock automatically.")

with tab_tutor:
    if not chunks:
        st.info("Upload course material from the Dashboard first.")
    else:
        st.markdown("### Ask your course, not a generic chatbot")
        question = st.text_area("What are you trying to understand?", placeholder="Explain convolution and derive the result for this example.", height=110, key="tutor_question")
        top_k = st.slider("Evidence passages", 2, 6, 4)
        if st.button("Find evidence", type="primary"):
            if not question.strip():
                st.warning("Ask a question first.")
            else:
                results = retrieve_relevant_chunks(question, chunks, top_k=top_k)
                st.session_state["retrieved_chunks"] = results
                st.session_state["retrieval_question"] = question
                st.session_state["tutor_answer"] = ""

        results = st.session_state.get("retrieved_chunks", [])
        if results:
            st.markdown("#### Evidence used")
            for i, chunk in enumerate(results, 1):
                with st.expander(f"{i}. {source_badge(chunk)} · relevance {chunk.get('retrieval_score',0):.2f}"):
                    render_rich_text(chunk.get("text", ""))

            if st.button("Explain with StudyPilot", type="primary"):
                try:
                    with st.spinner("Generating a source-grounded explanation..."):
                        answer = generate_tutor_answer(question=st.session_state.get("retrieval_question", question), retrieved_chunks=results, model=text_model, ollama_url=ollama_url)
                    st.session_state["tutor_answer"] = answer
                except Exception as exc:
                    st.error(f"Tutor failed: {type(exc).__name__}: {exc}")

        if st.session_state.get("tutor_answer"):
            st.markdown("### Tutor")
            st.markdown(st.session_state["tutor_answer"])
            st.caption("Grounding rule: if the uploaded material does not support an answer, the tutor should say so instead of silently guessing.")
            feedback_topic = st.session_state.get("assessment_topic", subject)
            f1, f2 = st.columns(2)
            if f1.button("✓ I understand", use_container_width=True):
                record_conversation_feedback(st.session_state, feedback_topic, True)
                st.success("Nice. Your mastery estimate was updated slightly upward.")
            if f2.button("↻ Still confused", use_container_width=True):
                record_conversation_feedback(st.session_state, feedback_topic, False)
                st.info("Your next recommendation will prioritize this topic.")

with tab_assess:
    if not chunks:
        st.info("Upload course material from the Dashboard first.")
    else:
        st.markdown("### Adaptive assessment")
        recommended = recommend_topics(st.session_state, fallback=subject)
        if summary["attempts"] == 0:
            st.info("🩺 New learner detected. Start with a diagnostic quiz to establish your initial mastery profile.")

        a1, a2, a3 = st.columns(3)
        topic = a1.text_input("Scope / topic", value=recommended[0] if recommended else subject, key="assessment_topic")
        count = a2.slider("Questions", 3, 8, 5, key="assessment_count")
        diff = a3.selectbox("Difficulty", ["Mixed", "Easy", "Medium", "Hard"], key="assessment_difficulty")

        if st.button("Generate adaptive assessment", type="primary", use_container_width=True):
            for key in ("active_assessment", "assessment_results"):
                st.session_state.pop(key, None)
            try:
                assessment_chunks = retrieve_relevant_chunks(topic, chunks, top_k=5)
                if not assessment_chunks:
                    st.error("No relevant source material was found for that scope.")
                else:
                    with st.spinner("Generating and verifying questions from your course material..."):
                        questions = generate_assessment(topic, assessment_chunks, count, diff, st.session_state.get("question_history", []), text_model, ollama_url)
                    st.session_state["active_assessment"] = questions
                    st.session_state["assessment_chunks"] = assessment_chunks
                    st.session_state["question_history"] = st.session_state.get("question_history", []) + [q["question"] for q in questions]
                    st.success(f"Ready: {len(questions)} source-tagged questions.")
            except Exception as exc:
                st.error(f"Assessment generation failed: {type(exc).__name__}: {exc}")

        questions = st.session_state.get("active_assessment", [])
        if questions:
            with st.form("adaptive_assessment"):
                for q in questions:
                    st.markdown(f"#### {q['id']}")
                    render_rich_text(q["question"])
                    st.caption(f"{q['topic']} · {q['difficulty']} · {q['source']}, p.{q['page']}")
                    key = f"answer_{q['id']}"
                    if q["type"] == "mcq":
                        letters = option_letters(len(q["options"]))
                        # Radio labels stay short and stable; equations are rendered
                        # separately so Streamlit does not show raw LaTeX in a widget.
                        st.radio("Select an option", letters, index=None, key=key, horizontal=True)
                        for letter, option in zip(letters, q["options"]):
                            st.markdown(f"**{letter}.**")
                            render_rich_text(option)
                    elif q["type"] == "numerical":
                        st.text_input("Numerical answer", key=key)
                    else:
                        st.text_area("Your answer", key=key)
                submit = st.form_submit_button("Submit assessment", type="primary")

            if submit:
                answers = {}
                for q in questions:
                    selected = st.session_state.get(f"answer_{q['id']}", "")
                    if q["type"] == "mcq" and selected in option_letters(len(q["options"])):
                        selected = q["options"][ord(selected) - ord("A")]
                    answers[q["id"]] = selected
                if any(not str(v).strip() for v in answers.values()):
                    st.warning("Please answer every question before submitting.")
                else:
                    try:
                        with st.spinner("Grading and analysing misconceptions..."):
                            results = evaluate_assessment_answers(questions, answers, text_model, ollama_url)
                        st.session_state["assessment_results"] = results
                        for item in results:
                            record_attempt(st.session_state, item["topic"], item["correct"])
                        st.success("Assessment complete. Your mastery model has been updated.")
                    except Exception as exc:
                        st.error(f"Grading failed: {type(exc).__name__}: {exc}")

        results = st.session_state.get("assessment_results")
        if results:
            correct = sum(bool(x["correct"]) for x in results)
            st.markdown("### Assessment report")
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{correct}/{len(results)}")
            c2.metric("Accuracy", f"{correct/len(results):.0%}")
            weak = {}
            for item in results:
                weak.setdefault(item["topic"], [0,0])
                weak[item["topic"]][1] += 1
                weak[item["topic"]][0] += int(item["correct"])
            weak_topics = sorted(weak.items(), key=lambda x: x[1][0]/x[1][1])
            c3.metric("Weak topics", sum(1 for _,v in weak_topics if v[0] < v[1]))
            for q, item in zip(questions, results):
                with st.expander(f"{q['id']} · {'Correct' if item['correct'] else 'Needs review'}"):
                    st.write(item.get("feedback", ""))
                    st.caption(f"Source: {item['source']}, page {item['page']}")
            st.markdown("#### Next best topics")
            for name, (right, total) in weak_topics[:3]:
                if right < total:
                    st.markdown(f'<div class="weak"><b>{name}</b> · {right}/{total} correct · recommended for revision</div>', unsafe_allow_html=True)

with tab_sources:
    if not pages and not chunks:
        st.info("Your source library is empty.")
    else:
        st.markdown("### Source library")
        st.caption("Every tutor/assessment item carries its source and location. Select a location below to inspect the exact page.")
        all_source_pages = [(p.get("source"), str(p.get("page"))) for p in pages]
        if all_source_pages:
            selected = st.selectbox("Open source page", all_source_pages, format_func=lambda x: f"{x[0]} · page {x[1]}")
            render_source_page(pages, selected[0], selected[1])
            page_data = next((p for p in pages if p.get("source") == selected[0] and str(p.get("page")) == selected[1]), None)
            if page_data:
                st.markdown("#### Extracted content")
                metadata = page_data.get("extraction_metadata", {})
                st.caption(
                    f"Source: {page_data.get('source', 'Unknown')} · page {page_data.get('page', '?')} · "
                    f"{metadata.get('text_block_count', 0)} ordered text blocks"
                )
                render_rich_text(page_data.get("text", ""))
                with st.expander("Original PDF text (unformatted)"):
                    st.code(page_data.get("raw_text", ""), language=None)
                if page_data.get("vision_text"):
                    st.markdown("#### Vision-recovered content")
                    render_rich_text(page_data["vision_text"])
        if chunks:
            st.markdown("### Source index")
            for chunk in chunks[:40]:
                st.markdown(f'<span class="source-chip">{source_badge(chunk)}</span>', unsafe_allow_html=True)

with tab_progress:
    st.markdown("### Your learner model")
    if summary["attempts"] == 0:
        st.info("Complete a diagnostic/practice assessment to initialize topic mastery.")
    else:
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Overall accuracy", f"{summary['accuracy']:.0%}")
        pc2.metric("Questions answered", summary["attempts"])
        pc3.metric("Tutor feedback events", summary.get("conversation_events", 0))
        st.markdown("#### Topic mastery")
        for topic, mastery in sorted(summary["topic_mastery"].items(), key=lambda x: x[1]):
            st.write(f"**{topic}** · {mastery:.0%}")
            st.progress(float(mastery))
        st.markdown("#### Recommended revision")
        for topic in recommend_topics(st.session_state, fallback=subject):
            st.markdown(f'<div class="weak">📌 <b>{topic}</b> — prioritised from your current mastery estimate.</div>', unsafe_allow_html=True)

with tab_eval:
    st.markdown("### System evaluation")
    st.caption("Track D asks for retrieval/generation evaluation plus simulated personalization. This panel gives you a transparent local benchmark and keeps the test set inspectable.")
    default_cases = []
    if chunks:
        first = chunks[0]
        default_cases = [{"question": "What is the main concept on this passage?", "expected_sources": [[first.get("source"), first.get("page")]]}]
    test_json = st.text_area("Retrieval test set (JSON)", value=json.dumps(default_cases, indent=2), height=180)
    if st.button("Run retrieval benchmark", disabled=not chunks):
        try:
            cases = json.loads(test_json)
            report = evaluate_retrieval(cases, retrieve_relevant_chunks, chunks)
            e1,e2 = st.columns(2)
            e1.metric("Context precision", f"{report['context_precision']:.2%}")
            e2.metric("Context recall", f"{report['context_recall']:.2%}")
            st.dataframe(report["rows"], use_container_width=True)
        except Exception as exc:
            st.error(f"Evaluation failed: {type(exc).__name__}: {exc}")

    st.markdown("#### Personalization benchmark")
    before = st.number_input("Baseline accuracy", 0.0, 1.0, 0.50, 0.01)
    after = st.number_input("Post-personalization accuracy", 0.0, 1.0, float(summary["accuracy"]), 0.01)
    repeated = st.number_input("Repeated questions", 0, 1000, 0)
    total = st.number_input("Total generated questions", 1, 1000, max(1, summary["attempts"]))
    if st.button("Calculate personalization metrics"):
        rep = personalization_report(before, after, repeated, total)
        e1,e2,e3 = st.columns(3)
        e1.metric("Mastery/accuracy gain", f"{rep['accuracy_gain']:+.1%}")
        e2.metric("Question repetition rate", f"{rep['repetition_rate']:.1%}")
        e3.metric("Personalization gain", f"{rep['mastery_gain']:+.1%}")

    st.info("For the final submission, report these results on a fixed team-built test set and include off-material questions. If you install RAGAS/DeepEval/TruLens, the same test cases can be fed into that framework for formal faithfulness and answer-relevancy scores.")

st.caption("Local-first prototype · Course material remains the grounding source · Ollama provides local generation")
