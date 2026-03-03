# AI-Powered CRM Email Generation System

> An end-to-end AI system that generates, evaluates, and previews personalised CRM emails for global live-service games — covering the full pipeline from retrieval-augmented generation to quality scoring, safety guardrails, and performance uplift prediction.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Getting Started](#getting-started)
6. [Configuration](#configuration)
7. [Demo Walkthrough](#demo-walkthrough)
8. [Evaluation Methodology](#evaluation-methodology)
9. [Limitations & Future Work](#limitations--future-work)

---

## Project Overview

Global live-service games require high-volume, personalised CRM campaigns across multiple player segments. Manual email copywriting is slow, inconsistent, and difficult to scale. This system addresses that challenge by combining retrieval-augmented generation (RAG) with LLM-based content creation and automated quality evaluation.

**Key capabilities:**

- Retrieves the most relevant historical high-performing emails as generation context via a hybrid BM25 + dense retrieval pipeline with RRF fusion and semantic reranking
- Generates three tone-differentiated email variants (Urgency / Storytelling / Direct Value) from a single Campaign Brief using DeepSeek Chat
- Scores each variant across four quality dimensions using LLM-as-Judge and flags policy violations via a rule-based safety guardrail engine
- Predicts performance uplift relative to industry baselines via Monte Carlo simulation and estimates operational efficiency savings
- Renders campaign-themed HTML email previews and surfaces all results in a 5-page interactive Streamlit demo

**Target role alignment:** Designed to demonstrate direct relevance to Tencent IEGG CRM & SmartLink product requirements, covering content ingestion pipelines, LLM-based generation, automated quality scoring, safety compliance, and end-to-end demo delivery.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Campaign Brief (UI)                 │
│  Game · Campaign Type · Segment · Tone · KPI · Context│
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Ingestion Pipeline                      │
│  JSON/CSV Loader → Validate → Normalise → Score      │
│  Corpus: 80 synthetic historical gaming CRM emails   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Hybrid Retriever                        │
│                                                      │
│  ┌─────────────────┐    ┌──────────────────────────┐│
│  │  BM25 Sparse    │    │  Qwen text-embedding-v3  ││
│  │  (rank-bm25)    │    │  Dense Retrieval         ││
│  │  keyword match  │    │  semantic similarity     ││
│  └────────┬────────┘    └──────────┬───────────────┘│
│           │                        │                 │
│           └───────────┬────────────┘                 │
│                       ▼                              │
│           RRF Fusion (k=60, w=0.4/0.6)               │
│                       ▼                              │
│           Qwen gte-rerank (Top-10 → Top-3)           │
└──────────────────────┬──────────────────────────────┘
                       │ RAG context (Top-3 emails)
                       ▼
┌─────────────────────────────────────────────────────┐
│           DeepSeek Chat — Email Generator            │
│                                                      │
│  Variant A: Urgency       (FOMO, deadlines, scarcity)│
│  Variant B: Storytelling  (narrative, emotional arc) │
│  Variant C: Direct Value  (facts, bullets, clarity)  │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Quality Scoring & Safety Guardrails          │
│                                                      │
│  LLM-as-Judge (DeepSeek):                           │
│    Relevance (30%) · Tone (25%) · Compliance (25%)  │
│    Creativity (20%) → Weighted Overall Score        │
│                                                      │
│  Rule-based Guardrail Engine:                        │
│    Banned phrases · Over-promises · Spam triggers   │
│    Subject line length · CTA verb validation        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Rendering & Evaluation                       │
│                                                      │
│  Jinja2 HTML Renderer  → Campaign-themed email preview│
│  Monte Carlo Estimator → Uplift vs baseline + 95% CI │
│  Efficiency Calculator → Hours saved, FTE equivalent │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           Streamlit Demo UI (5 pages)                │
│                                                      │
│  1. Campaign Brief    — input form                   │
│  2. Email Output      — content + HTML preview       │
│  3. Variant Comparison — side-by-side with scores   │
│  4. Quality Scores    — dimension breakdown + flags  │
│  5. Uplift Analytics  — performance + efficiency     │
└─────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| LLM Generation | DeepSeek Chat (`deepseek-chat`) | Email generation + LLM-as-Judge scoring |
| Embedding | Qwen `text-embedding-v3` (DashScope) | 1024-dim semantic vectors for dense retrieval |
| Reranking | Qwen `gte-rerank` (DashScope) | Cross-encoder semantic reranking |
| Sparse Retrieval | BM25 (`rank-bm25`) | Keyword-based candidate recall |
| Fusion | RRF — Reciprocal Rank Fusion | Combining sparse + dense ranked lists |
| Vector Store | NumPy cosine similarity | Lightweight in-memory dense search |
| HTML Rendering | Jinja2 | Campaign-themed email template rendering |
| UI Framework | Streamlit | 5-page interactive demo |
| Data Validation | Pydantic v2 | Typed data models and config management |
| Config | PyYAML + python-dotenv | Layered configuration and API key management |
| Testing | pytest | 19 unit tests covering core data models and pipeline |

---

## Project Structure

```
crm_email_system/
├── app.py                          # Streamlit entry point (5-page demo)
├── config.yaml                     # System configuration (models, weights, thresholds)
├── .env.example                    # API key template
├── run.bat                         # Windows one-click launcher
├── requirements.txt
│
├── core/
│   ├── types.py                    # Pydantic data models (CampaignBrief, EmailVariant, etc.)
│   ├── base.py                     # Abstract base classes for all pluggable components
│   ├── config.py                   # Settings loader + API key management
│   └── pipeline.py                 # End-to-end orchestrator (lazy-loaded components)
│
├── data/
│   ├── mock_generator.py           # Synthetic corpus generator (80 emails, calibrated to Mailchimp benchmarks)
│   └── processed/
│       ├── historical_emails.json  # Generated corpus
│       ├── golden_test_set.json    # 10-case evaluation set
│       └── vector_index.pkl        # Cached vector index (auto-generated, gitignored)
│
├── ingestion/
│   └── pipeline.py                 # Load → Validate → Normalise → Score transforms
│
├── retrieval/
│   ├── vector_store.py             # Qwen embedding client + NumPy vector store
│   ├── bm25_retriever.py           # BM25 sparse retrieval
│   └── hybrid_retriever.py         # RRF fusion + Qwen reranker
│
├── generation/
│   ├── prompt_templates.py         # Prompt library (per campaign type × tone style)
│   └── email_generator.py          # DeepSeek API integration + mock fallback
│
├── scoring/
│   ├── quality_scorer.py           # LLM-as-Judge scorer (4 dimensions)
│   └── safety_guardrail.py         # Rule-based safety and compliance checks
│
├── templating/
│   └── html_renderer.py            # Jinja2 HTML email renderer (6 campaign themes)
│
├── evaluation/
│   └── uplift_estimator.py         # Monte Carlo uplift simulation + efficiency calculator
│
└── tests/
    └── unit/test_day1.py           # 19 unit tests (types, config, ingestion pipeline)
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ZhangRuoqi-39/crm_email_system.git
cd crm_email_system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
copy .env.example .env      # Windows
cp .env.example .env        # macOS / Linux

# Edit .env and fill in your keys:
# DEEPSEEK_API_KEY=sk-...
# DASHSCOPE_API_KEY=sk-...
```

### Generate Synthetic Data (first run only)

```bash
python data/mock_generator.py
```

This creates `data/processed/historical_emails.json` (80 emails) and `data/processed/golden_test_set.json` (10 evaluation cases).

### Launch the Demo

**Windows (recommended):**
```
Double-click run.bat
```

**macOS / Linux:**
```bash
# Load environment variables and start
export $(cat .env | grep -v '#' | xargs)
streamlit run app.py
```

**Manual (any OS):**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> **Note:** Both API keys are optional. The system runs in **mock mode** without them — embeddings fall back to deterministic random vectors and email generation uses pre-defined templates. All 5 UI pages remain fully functional for demonstration purposes.

### Running Tests

```bash
pytest tests/unit/test_day1.py -v
```

---

## Configuration

All system parameters are managed in `config.yaml`:

```yaml
llm:
  provider: "deepseek"
  model: "deepseek-chat"
  temperature: 0.7

embedding:
  provider: "qwen"
  model: "text-embedding-v3"

rerank:
  provider: "qwen"
  model: "gte-rerank"
  top_n: 3

retrieval:
  top_k: 10          # candidates before reranking
  bm25_weight: 0.4
  dense_weight: 0.6
  rrf_k: 60

scoring:
  weights:
    relevance: 0.30
    tone: 0.25
    compliance: 0.25
    creativity: 0.20
```

To switch providers, update the `provider` field and set the corresponding environment variable — no code changes required.

---

## Demo Walkthrough

| Page | What it shows |
|---|---|
| **Campaign Brief** | Input form: game, campaign type, target segment, tone style, KPI, context |
| **Email Output** | Full email content (subject, preheader, body, CTA) + live HTML preview |
| **Variant Comparison** | Side-by-side comparison of all 3 tone variants with scores and guardrail status |
| **Quality Scores** | Dimension-level score table + per-variant guardrail report |
| **Uplift Analytics** | Predicted open rate / CTR uplift with 95% CI + operational efficiency savings |

**Sidebar API status indicators** show whether each provider is running in live or mock mode in real time.

---

## Evaluation Methodology

### Quality Scoring (LLM-as-Judge)

Each generated variant is evaluated by DeepSeek on four dimensions:

| Dimension | Weight | What is measured |
|---|---|---|
| Relevance | 30% | Alignment with campaign type, player segment, and KPI |
| Tone Match | 25% | Consistency between copy style and requested tone |
| Compliance | 25% | Absence of misleading claims, banned phrases, policy violations |
| Creativity | 20% | Originality, engagement quality, memorability |

**Overall = 0.30 × Relevance + 0.25 × Tone + 0.25 × Compliance + 0.20 × Creativity**

When the API key is unavailable, a heuristic rule-based scorer is used as fallback.

### Safety Guardrail Engine

Rule-based checks applied to every variant before approval:

- Globally banned phrases (e.g. "guaranteed", "risk-free", "act now or else")
- Campaign-specific forbidden keywords (user-defined per brief)
- Over-promise patterns (regex-based, e.g. "500% bonus")
- Spam trigger language (e.g. "WINNER", "URGENT URGENT")
- Subject line length enforcement (≤ 60 characters)
- CTA action verb validation

### Uplift Simulation

Monte Carlo simulation (n=100) using quality score as a proxy signal:

```
uplift_factor = 1.0 + (quality_score − 0.5) × 0.50
predicted_open_rate = baseline (21%) × uplift_factor × campaign_noise
95% CI = [percentile(2.5%), percentile(97.5%)] over 100 simulations
```

Baseline values are calibrated to Mailchimp Gaming Industry benchmarks (Open Rate: 21%, CTR: 3.0%).

> **Important:** Uplift figures are directional estimates for planning purposes, not statistically validated predictions. Production deployment would require recalibration against real send data.

---

## Limitations & Future Work

### Current Limitations

| Area | Limitation |
|---|---|
| Corpus | 80 synthetic emails; no real Tencent campaign data |
| Uplift model | Linear proxy mapping; not validated against real send outcomes |
| Scoring weights | Set by heuristic; not derived from regression on historical performance |
| Variant comparison | Pre-generation quality screening only; not a live A/B test with real users |
| Evaluation coverage | Unit tests cover Day 1 components only; no integration tests for retrieval/generation |

### Planned Improvements

**Short-term**
- Replace synthetic corpus with real historical campaign data via CRM system integration
- Add integration tests covering the full retrieval → generation → scoring pipeline
- Implement trace logging (JSON Lines) for pipeline observability and debugging

**Medium-term**
- Calibrate scoring weights via regression analysis on historical open rate / CTR data
- Connect to a real email sending platform (e.g. SendGrid, Tencent Enterprise Mail) to enable true A/B testing with live user cohorts
- Add multi-language support for localised campaign markets (SEA, LATAM)

**Long-term**
- Build a feedback loop: collect real send performance data → retrain uplift model → continuously improve generation quality
- Extend to other CRM content types: push notifications, in-app messages, SMS
- Integrate player behavioural signals (login frequency, spend history) as additional personalisation inputs to the Campaign Brief

---

## Acknowledgements

System architecture inspired by the RAG pipeline design patterns documented in internal DEV_SPEC.md, adapted for the CRM email generation domain.

Baseline performance metrics sourced from Mailchimp Email Marketing Benchmarks (Gaming Industry, 2024).