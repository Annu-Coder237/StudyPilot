# StudyPilot AI — Multimodal Adaptive Tutor

**StudyPilot AI** is a local-first, source-grounded learning assistant built for the Multimodal AI Hackathon 2026, Track D.

It helps students learn from their own course materials by combining document ingestion, visual-content recovery, retrieval-augmented generation (RAG), adaptive assessments, and learner-progress tracking.

## The Problem

Students often study using a mixture of lecture slides, textbooks, diagrams, and recorded lectures. Finding relevant explanations, checking understanding, and deciding what to revise can be time-consuming.

StudyPilot aims to bring these activities into one learning workflow while keeping answers grounded in the student's uploaded course materials.

## Features

- **PDF knowledge base:** Extracts text from lecture slides and textbooks, with optional vision-based recovery for pages that have poor text extraction or relevant visual content.
- **Video ingestion:** Samples lecture-video frames and extracts timestamped content when FFmpeg and the required local vision model are available.
- **Source-grounded tutoring:** Retrieves relevant course passages before generating explanations and retains source information for inspection.
- **Course map:** Organizes extracted material into a lightweight overview of topics and concepts.
- **Adaptive assessments:** Generates practice questions with topic, difficulty, and source metadata. Supported question formats include MCQs, short-answer questions, and numerical questions.
- **Learner model:** Tracks performance by topic and updates mastery estimates using assessment results and tutor feedback.
- **Personalized revision:** Uses learner progress to recommend topics for further practice.
- **Source inspection:** Lets learners inspect extracted content and view the original PDF page associated with a source.
- **Evaluation dashboard:** Provides retrieval precision/recall benchmarks and prototype personalization metrics.

## How It Works

1. **Upload:** The learner provides course PDFs or, optionally, a lecture video.
2. **Ingest:** StudyPilot extracts document text and selectively uses a local vision model for visual recovery. Video ingestion samples frames when the required tools are installed.
3. **Retrieve:** The system searches the extracted course content for passages relevant to a question or assessment topic.
4. **Generate:** A locally hosted language model generates a response or practice questions using retrieved material as context.
5. **Assess:** The learner answers questions and receives grading feedback.
6. **Personalize:** Assessment results and comprehension feedback update topic-level mastery estimates and revision recommendations.

## Technology

- Python
- Streamlit
- PyMuPDF for PDF processing
- Ollama for local language-model and vision-model inference
- FFmpeg for optional video-frame extraction
- Retrieval, assessment, learner-model, course-map, and evaluation modules implemented in Python

Exact Python package dependencies are listed in `requirements.txt`.

## Prerequisites

- Python installed on your computer.
- Git, if cloning the repository.
- Ollama installed and running locally.
- Sufficient RAM and disk space for the selected local models.
- FFmpeg installed and available on `PATH` for video ingestion.

Video ingestion is optional. PDF-based learning can be used without the video-processing tools, subject to the installed Python dependencies and availability of the configured local models.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Annu-Coder237/StudyPilot.git
cd StudyPilot
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly instead.

**Linux / Ubuntu:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Start Ollama

Install Ollama from its official website and ensure the local service is running.

Pull the models configured in the application. For the default configuration:

```bash
ollama pull qwen3.5:4b
ollama pull qwen2.5vl:3b
```

If you change the model names in the application, pull the corresponding models instead. Model availability and hardware performance depend on your local Ollama installation.

### 5. Run StudyPilot

From the repository root:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal by Streamlit, usually `http://localhost:8501`.

## Using the Application

1. Start StudyPilot and confirm that Ollama is available.
2. Upload a course PDF from the Dashboard.
3. Keep PDF visual recovery set to **Auto** for selective recovery of weak-text or visual pages.
4. Wait for ingestion and inspect the generated course map and source library.
5. Open the Tutor tab, retrieve evidence, and ask a question about the uploaded material.
6. Open Assess to generate and complete a practice assessment.
7. Review your assessment report, topic mastery, and revision recommendations.
8. Use the Evaluation tab to inspect the available prototype benchmarks.

For video ingestion, install FFmpeg and ensure both `ffmpeg` and `ffprobe` are accessible from the terminal before starting the application.

## Project Structure

```text
StudyPilot/
├── app.py
├── requirements.txt
├── README.md
└── tutor/
    ├── ingestion.py
    ├── video_ingestion.py
    ├── retrieval.py
    ├── generation.py
    ├── assessment.py
    ├── learner_model.py
    ├── course_map.py
    └── evaluation.py
```

This is the intended core structure; optional files may also be present in the repository.

## Privacy and Local Inference

StudyPilot is designed around local inference through Ollama. Uploaded course content is processed by the application and supplied to the configured local models.

Local inference does not itself guarantee that every dependency or component is offline. Users should verify their configuration and avoid uploading confidential material unless they are comfortable with how their local environment processes it.

## Current Limitations

- Extraction quality depends on the source PDF, scan quality, mathematical notation, and the configured vision model.
- Video ingestion requires additional system dependencies and can take time for large videos.
- Generated explanations and assessments may contain mistakes; learners should verify important information against the source material.
- The learner model and evaluation dashboard are prototype implementations. Their metrics should not be interpreted as a validated measure of learning outcomes.
- Runtime speed depends on the selected model, hardware, and available memory.

## Hackathon Context

StudyPilot was developed for the **Multimodal AI Hackathon 2026 — Track D: AI Tutoring**. Its goal is to explore how multimodal course-material ingestion, retrieval-grounded explanations, assessment, and learner modeling can support a more personalized study experience.

## License

No license is specified yet. Add an appropriate open-source license before redistributing the project under specific reuse terms.

