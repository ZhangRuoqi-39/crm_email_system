"""
app.py — CRM Email Generation System · Streamlit Demo
======================================================
Tencent IEGG Capstone Project

Run: streamlit run app.py
Requires: DEEPSEEK_API_KEY and/or DASHSCOPE_API_KEY in environment
          (falls back to mock mode if keys are absent)
"""

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="CRM Email AI",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Layout ── */
  [data-testid="stAppViewContainer"] { background: #0d0d1a; }
  [data-testid="stSidebar"] { background: #12122a !important; border-right: 1px solid #2a2a4a; }
  .block-container { padding-top: 1.5rem; }

  /* ── Global text: force white on all sidebar elements ── */
  [data-testid="stSidebar"] * { color: #e8e8f0 !important; }
  [data-testid="stSidebar"] .stMarkdown p  { color: #e8e8f0 !important; }
  [data-testid="stSidebar"] .stMarkdown h3 { color: #ffffff !important; font-size: 1.1rem; }
  [data-testid="stSidebar"] small,
  [data-testid="stSidebar"] caption        { color: #aaaacc !important; }

  /* ── Sidebar radio nav ── */
  [data-testid="stSidebar"] [data-testid="stRadio"] label        { color: #d0d0ee !important; font-size: 0.95rem; padding: 4px 0; }
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover  { color: #ffffff !important; }
  [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] + div p { color: #a78bfa !important; font-weight: 700; }

  /* ── API status dots ── */
  [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #ccccee !important; font-size: 0.88rem; line-height: 1.8; }

  /* ── Main page headings & body ── */
  h1, h2, h3 { color: #ffffff !important; }
  p, li, label { color: #d8d8f0; }

  /* ── Streamlit native widgets ── */
  [data-testid="stSelectbox"] label,
  [data-testid="stSlider"]    label,
  [data-testid="stTextInput"] label,
  [data-testid="stTextArea"]  label { color: #c8c8e8 !important; font-weight: 600; }

  /* selectbox dropdown text */
  [data-testid="stSelectbox"] div[data-baseweb="select"] { background: #1e1e38; border-color: #3a3a5c; }
  [data-testid="stSelectbox"] div[data-baseweb="select"] span { color: #e8e8f0 !important; }

  /* text inputs */
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"]  textarea { background: #1a1a30 !important; color: #e8e8f0 !important; border-color: #3a3a5c !important; }

  /* ── Metrics ── */
  [data-testid="stMetric"] label { color: #aaaacc !important; font-size: 0.8rem; }
  [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.6rem; font-weight: 800; }
  [data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #a78bfa !important; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { background: #1a1a30; }

  /* ── Alert / info / success / warning boxes ── */
  [data-testid="stAlert"] p { color: #ffffff !important; }
  [data-testid="stAlert"] a { color: #a78bfa !important; font-weight: 700; }

  /* ── Score cards (custom HTML) ── */
  .score-card { background: #1a1a30; border: 1px solid #2a2a50; border-radius: 10px; padding: 16px; text-align: center; margin: 4px; }
  .score-val  { font-size: 28px; font-weight: 800; margin: 4px 0; color: #ffffff; }
  .score-lbl  { font-size: 12px; color: #9999bb; text-transform: uppercase; letter-spacing: 1px; }

  /* ── Section headers ── */
  .section-header { background: linear-gradient(90deg, #4f46e5, #7c3aed); color: white; padding: 10px 16px; border-radius: 8px; font-weight: 700; margin-bottom: 12px; }

  /* ── Variant badges ── */
  .variant-badge   { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; margin-right: 6px; }
  .badge-urgency      { background: #dc262625; color: #fca5a5; border: 1px solid #dc262655; }
  .badge-storytelling { background: #7c3aed25; color: #d8b4fe; border: 1px solid #7c3aed55; }
  .badge-direct_value { background: #05966925; color: #6ee7b7; border: 1px solid #05966955; }

  /* ── Uplift colours ── */
  .uplift-positive { color: #34d399; font-weight: 700; }
  .uplift-negative { color: #f87171; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Pipeline singleton (cached across reruns)
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="🔧 Initialising AI pipeline...")
def get_pipeline():
    from core.pipeline import CRMEmailPipeline
    return CRMEmailPipeline()


# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────

from core.types import CampaignType, ToneStyle, PlayerSegment, GameTitle

PAGES = {
    "🎯 Campaign Brief": "brief",
    "✉️ Email Output":    "output",
    "🔀 Variant Comparison": "ab",
    "📊 Quality Scores":  "quality",
    "📈 Uplift Analytics": "analytics",
}

with st.sidebar:
    st.markdown("### 🎮 CRM Email AI")
    st.caption("Capstone Demo")
    st.divider()
    page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()

    # API status indicator
    import os
    ds_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    qs_key = bool(os.environ.get("DASHSCOPE_API_KEY"))
    st.markdown("**API Status**")
    st.markdown(f"{'🟢' if ds_key else '🟡'} DeepSeek {'Live' if ds_key else 'Mock mode'}")
    st.markdown(f"{'🟢' if qs_key else '🟡'} Qwen {'Live' if qs_key else 'Mock mode'}")
    st.divider()
    st.caption("Pipeline: DeepSeek Chat · Qwen text-embedding-v3 · gte-rerank · BM25 RRF")

page_key = PAGES[page]

# ══════════════════════════════════════════════
# PAGE 1: Campaign Brief
# ══════════════════════════════════════════════

if page_key == "brief":
    st.markdown("## 🎯 Campaign Brief")
    st.caption("Define your campaign parameters. The AI will generate 3 email variants in different tone styles for comparison and pre-screening.")

    with st.form("campaign_form"):
        col1, col2 = st.columns(2)

        with col1:
            game = st.selectbox("Game Title", [g.value for g in GameTitle])
            campaign_type = st.selectbox(
                "Campaign Type",
                [c.value for c in CampaignType],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            target_segment = st.selectbox("Target Segment", [s.value for s in PlayerSegment])

        with col2:
            tone_style = st.selectbox(
                "Primary Tone Style",
                [t.value for t in ToneStyle],
                format_func=lambda x: x.replace("_", " ").title(),
            )
            kpi = st.selectbox("KPI Focus", ["Open Rate", "CTR", "Conversion Rate", "Unsubscribe Reduction"])
            num_variants = st.slider("Number of Variants", 1, 3, 3)

        context = st.text_area(
            "Campaign Context",
            placeholder="e.g. Players inactive for 30-90 days. New season launched last week. 500 coin comeback bonus available.",
            height=100,
        )

        col3, col4 = st.columns(2)
        with col3:
            brand_kw_raw = st.text_input(
                "Brand Keywords (comma-separated)",
                placeholder="comeback, exclusive, legendary",
            )
        with col4:
            forbidden_kw_raw = st.text_input(
                "Forbidden Words (comma-separated)",
                placeholder="guaranteed, risk-free",
            )

        submitted = st.form_submit_button("🚀 Generate Emails", use_container_width=True, type="primary")

    if submitted:
        if not context.strip():
            st.error("Please provide campaign context before generating.")
        else:
            from core.types import CampaignBrief, CampaignType as CT, ToneStyle as TS, PlayerSegment as PS

            brief = CampaignBrief(
                game=game,
                campaign_type=CT(campaign_type),
                target_segment=PS(target_segment),
                tone_style=TS(tone_style),
                kpi=kpi,
                context=context,
                brand_keywords=[k.strip() for k in brand_kw_raw.split(",") if k.strip()],
                forbidden_keywords=[k.strip() for k in forbidden_kw_raw.split(",") if k.strip()],
            )

            with st.spinner("🤖 Retrieving historical context + generating emails..."):
                pipeline = get_pipeline()
                t0 = time.time()
                result = pipeline.run(brief, num_variants=num_variants)
                elapsed = time.time() - t0

            st.session_state["result"] = result
            st.session_state["brief"] = brief

            st.success(f"✅ Generated {len(result.variants)} variants in {elapsed:.1f}s")
            st.info("👉 Navigate to **✉️ Email Output** or **🔀 Variant Comparison** to review results.")

            # Quick summary
            with st.expander("📊 Retrieved Historical Context", expanded=False):
                for e in result.retrieved_emails:
                    st.markdown(
                        f"**{e.subject_line}** — {e.game} | {e.campaign_type} | "
                        f"open: {e.open_rate:.1%} CTR: {e.ctr:.2%}"
                    )

    elif "brief" not in st.session_state:
        st.info("Fill out the form above and click **Generate Emails** to start.")


# ══════════════════════════════════════════════
# PAGE 2: Email Output
# ══════════════════════════════════════════════

elif page_key == "output":
    st.markdown("## ✉️ Email Output")

    if "result" not in st.session_state:
        st.warning("No emails generated yet. Go to **🎯 Campaign Brief** first.")
        st.stop()

    result = st.session_state["result"]
    brief = st.session_state["brief"]
    best = result.get_best_variant()

    # Variant selector
    variant_labels = [
        f"{v.tone_style.value.replace('_',' ').title()} "
        f"(⭐ {v.quality_score.overall:.0%})" if v.quality_score else v.tone_style.value
        for v in result.variants
    ]
    selected_idx = st.selectbox(
        "Select Variant",
        range(len(result.variants)),
        format_func=lambda i: ("🏆 " if result.variants[i].variant_id == (best.variant_id if best else "") else "") + variant_labels[i],
    )
    v = result.variants[selected_idx]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Email Content")
        tone_badge_class = f"badge-{v.tone_style.value}"
        st.markdown(
            f'<span class="variant-badge {tone_badge_class}">{v.tone_style.value.replace("_"," ").upper()}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Subject Line**")
        st.code(v.subject_line, language=None)
        chars = len(v.subject_line)
        color = "green" if chars <= 50 else "orange" if chars <= 60 else "red"
        st.markdown(f"<small style='color:{color}'>{chars}/60 characters</small>", unsafe_allow_html=True)

        st.markdown(f"**Preheader**")
        st.code(v.preheader, language=None)

        st.markdown(f"**Body Copy**")
        st.text_area("body", v.body_text, height=200, label_visibility="collapsed")

        st.markdown(f"**CTA**")
        st.code(v.cta_text, language=None)

    with col2:
        st.markdown("### 👁️ HTML Preview")
        if v.html_content:
            st.components.v1.html(v.html_content, height=520, scrolling=True)
        else:
            st.info("HTML preview not available")

    # Guardrail results
    if v.quality_score:
        qs = v.quality_score
        if qs.passed_guardrails:
            st.success("✅ All safety guardrails passed")
        else:
            st.error(f"⚠️ {len(qs.guardrail_flags)} guardrail flag(s) detected")
            for flag in qs.guardrail_flags:
                st.markdown(f"- 🚩 {flag}")


# ══════════════════════════════════════════════
# PAGE 3: Variant Comparison
# ══════════════════════════════════════════════

elif page_key == "ab":
    st.markdown("## 🔀 Variant Comparison")

    if "result" not in st.session_state:
        st.warning("No emails generated yet. Go to **🎯 Campaign Brief** first.")
        st.stop()

    result = st.session_state["result"]
    best = result.get_best_variant()

    cols = st.columns(len(result.variants))
    for i, (col, v) in enumerate(zip(cols, result.variants)):
        with col:
            is_best = v.variant_id == (best.variant_id if best else "")
            badge = "🏆 RECOMMENDED" if is_best else f"Variant {i+1}"
            header_color = "#4f46e5" if is_best else "#2a2a4a"
            st.markdown(
                f'<div style="background:{header_color};padding:8px 12px;border-radius:8px;'
                f'text-align:center;font-weight:700;margin-bottom:8px;">{badge}</div>',
                unsafe_allow_html=True,
            )

            tone_class = f"badge-{v.tone_style.value}"
            st.markdown(
                f'<center><span class="variant-badge {tone_class}">'
                f'{v.tone_style.value.replace("_"," ").upper()}</span></center>',
                unsafe_allow_html=True,
            )

            st.markdown(f"**Subject:**")
            st.info(v.subject_line)
            st.markdown(f"**CTA:**")
            st.success(v.cta_text)

            if v.quality_score:
                qs = v.quality_score
                overall_pct = int(qs.overall * 100)
                bar_color = "#34d399" if overall_pct >= 70 else "#fbbf24" if overall_pct >= 50 else "#f87171"
                st.markdown(
                    f'<div style="background:#1a1a30;border-radius:8px;padding:12px;margin-top:8px;">'
                    f'<div style="font-size:11px;color:#8888aa;margin-bottom:6px;">OVERALL SCORE</div>'
                    f'<div style="font-size:32px;font-weight:800;color:{bar_color}">{overall_pct}%</div>'
                    f'<div style="background:#2a2a40;border-radius:4px;height:6px;margin-top:8px;">'
                    f'<div style="background:{bar_color};width:{overall_pct}%;height:6px;border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

                st.markdown("**Score Breakdown**")
                metrics = {
                    "Relevance": qs.relevance,
                    "Tone Match": qs.tone,
                    "Compliance": qs.compliance,
                    "Creativity": qs.creativity,
                }
                for label, val in metrics.items():
                    pct = int(val * 100)
                    c = "#34d399" if pct >= 70 else "#fbbf24" if pct >= 50 else "#f87171"
                    st.markdown(
                        f'<div style="display:flex;justify-content:space-between;margin:3px 0;">'
                        f'<span style="font-size:12px;color:#8888aa;">{label}</span>'
                        f'<span style="font-size:12px;font-weight:700;color:{c}">{pct}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            if v.quality_score and not v.quality_score.passed_guardrails:
                st.warning(f"⚠️ {len(v.quality_score.guardrail_flags)} flag(s)")

    # Side-by-side body text
    st.divider()
    st.markdown("### Body Copy Comparison")
    body_cols = st.columns(len(result.variants))
    for col, v in zip(body_cols, result.variants):
        with col:
            st.caption(v.tone_style.value.replace("_", " ").title())
            st.text_area("body", v.body_text, height=250, label_visibility="collapsed", key=f"body_{v.variant_id}")


# ══════════════════════════════════════════════
# PAGE 4: Quality Scores
# ══════════════════════════════════════════════

elif page_key == "quality":
    st.markdown("## 📊 Quality Scores & Guardrails")

    if "result" not in st.session_state:
        st.warning("No emails generated yet. Go to **🎯 Campaign Brief** first.")
        st.stop()

    result = st.session_state["result"]
    brief = st.session_state["brief"]

    # Radar-style score table
    import numpy as np
    dimensions = ["Relevance", "Tone", "Compliance", "Creativity", "Overall"]
    table_data = {"Dimension": dimensions}

    for v in result.variants:
        tone_label = v.tone_style.value.replace("_", " ").title()
        if v.quality_score:
            qs = v.quality_score
            scores = [qs.relevance, qs.tone, qs.compliance, qs.creativity, qs.overall]
            table_data[tone_label] = [f"{s:.0%}" for s in scores]
        else:
            table_data[tone_label] = ["N/A"] * 5

    import pandas as pd
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-variant guardrail details
    st.markdown("### 🛡️ Guardrail Report")
    for v in result.variants:
        tone_label = v.tone_style.value.replace("_", " ").title()
        with st.expander(f"**{tone_label}** variant", expanded=True):
            if not v.quality_score:
                st.info("No score available")
                continue
            qs = v.quality_score
            if qs.passed_guardrails:
                st.markdown("✅ **All checks passed** — email is safe to send")
            else:
                st.markdown(f"⚠️ **{len(qs.guardrail_flags)} issue(s) found**")
                for flag in qs.guardrail_flags:
                    st.markdown(f"  🚩 {flag}")

            # Score bars
            score_cols = st.columns(4)
            metrics = [
                ("🎯 Relevance", qs.relevance),
                ("🎭 Tone", qs.tone),
                ("✅ Compliance", qs.compliance),
                ("✨ Creativity", qs.creativity),
            ]
            for sc, (label, val) in zip(score_cols, metrics):
                with sc:
                    pct = int(val * 100)
                    c = "normal" if pct >= 70 else "off"
                    sc.metric(label, f"{pct}%")

    # Scoring methodology note
    with st.expander("ℹ️ How scores are calculated"):
        st.markdown("""
        **Scoring Methodology (LLM-as-judge via DeepSeek)**

        | Dimension | Weight | What it measures |
        |---|---|---|
        | Relevance | 30% | Does content match campaign type, segment, and KPI? |
        | Tone | 25% | Does writing style match the requested tone (urgency/story/value)? |
        | Compliance | 25% | Is content professional, non-misleading, and brand-safe? |
        | Creativity | 20% | Is the copy original, engaging, and memorable? |

        **Overall = 0.30×Relevance + 0.25×Tone + 0.25×Compliance + 0.20×Creativity**

        In offline / mock mode, heuristic rule-based scoring is used as fallback.
        """)


# ══════════════════════════════════════════════
# PAGE 5: Uplift Analytics
# ══════════════════════════════════════════════

elif page_key == "analytics":
    st.markdown("## 📈 Uplift Simulation & Efficiency Analytics")

    if "result" not in st.session_state:
        st.warning("No emails generated yet. Go to **🎯 Campaign Brief** first.")
        st.stop()

    result = st.session_state["result"]

    import pandas as pd
    from evaluation.uplift_estimator import UpliftEstimator

    estimator = UpliftEstimator()

    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        monthly_vol = st.slider("Monthly Email Campaigns", 50, 1000, 200, step=50)
    with col_ctrl2:
        manual_mins = st.slider("Manual Minutes per Email", 20, 120, 45, step=5)

    st.divider()

    # Per-variant uplift cards
    st.markdown("### Predicted Performance Uplift")
    cols = st.columns(len(result.variants))
    all_reports = []

    for col, v in zip(cols, result.variants):
        report = estimator.estimate(v, monthly_email_volume=monthly_vol, manual_minutes=manual_mins)
        all_reports.append((v, report))
        with col:
            tone = v.tone_style.value.replace("_", " ").title()
            qs_overall = v.quality_score.overall if v.quality_score else 0.0

            or_color = "uplift-positive" if report.open_rate_uplift_pct >= 0 else "uplift-negative"
            ctr_color = "uplift-positive" if report.ctr_uplift_pct >= 0 else "uplift-negative"
            or_sign = "+" if report.open_rate_uplift_pct >= 0 else ""
            ctr_sign = "+" if report.ctr_uplift_pct >= 0 else ""

            st.markdown(
                f'<div style="background:#1a1a30;border:1px solid #2a2a50;border-radius:10px;padding:16px;">'
                f'<div style="font-weight:700;margin-bottom:10px;">{tone}</div>'
                f'<div style="font-size:11px;color:#8888aa;">Quality Score</div>'
                f'<div style="font-size:22px;font-weight:800;">{int(qs_overall*100)}%</div>'
                f'<hr style="border-color:#2a2a40;margin:10px 0;">'
                f'<div style="font-size:11px;color:#8888aa;">Predicted Open Rate</div>'
                f'<div style="font-size:20px;font-weight:700;">{report.predicted_open_rate:.1%}</div>'
                f'<div class="{or_color}" style="font-size:13px;">{or_sign}{report.open_rate_uplift_pct:.1f}% vs baseline</div>'
                f'<div style="font-size:10px;color:#666;">95% CI: [{report.open_rate_ci_low:.1%}, {report.open_rate_ci_high:.1%}]</div>'
                f'<hr style="border-color:#2a2a40;margin:10px 0;">'
                f'<div style="font-size:11px;color:#8888aa;">Predicted CTR</div>'
                f'<div style="font-size:20px;font-weight:700;">{report.predicted_ctr:.2%}</div>'
                f'<div class="{ctr_color}" style="font-size:13px;">{ctr_sign}{report.ctr_uplift_pct:.1f}% vs baseline</div>'
                f'<div style="font-size:10px;color:#666;">95% CI: [{report.ctr_ci_low:.2%}, {report.ctr_ci_high:.2%}]</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Efficiency savings (use best variant's report)
    best = result.get_best_variant()
    best_report = next((r for v, r in all_reports if v.variant_id == (best.variant_id if best else "")), all_reports[0][1] if all_reports else None)

    if best_report:
        st.markdown("### ⚡ Operational Efficiency Savings")
        eff_cols = st.columns(4)
        eff_cols[0].metric("Manual time / email", f"{int(best_report.manual_minutes_per_email)} min")
        eff_cols[1].metric("AI time / email", f"{int(best_report.ai_minutes_per_email)} min",
                           delta=f"-{int(best_report.manual_minutes_per_email - best_report.ai_minutes_per_email)} min")
        eff_cols[2].metric("Hours saved / month", f"{best_report.time_saved_hours_per_month}h",
                           delta=f"{monthly_vol} emails @ {int(best_report.manual_minutes_per_email)}min each")
        eff_cols[3].metric("FTE equivalent saved", f"{best_report.fte_equivalent_saved}",
                           delta="per month (160hr basis)")

        # Comparison chart
        st.markdown("### 📊 Baseline vs AI — Open Rate Comparison")
        chart_data = pd.DataFrame({
            "Approach": ["Baseline (Human)", "AI — Urgency", "AI — Storytelling", "AI — Direct Value"],
            "Open Rate": [
                best_report.baseline_open_rate,
                *[r.predicted_open_rate for _, r in all_reports],
            ],
        })
        # Pad if fewer than 3 variants
        while len(chart_data) < 4:
            chart_data = pd.concat([chart_data, pd.DataFrame({"Approach": ["N/A"], "Open Rate": [0]})])

        chart_data = chart_data.head(4)
        st.bar_chart(chart_data.set_index("Approach"), color="#4f46e5")

    # Methodology note
    with st.expander("ℹ️ Uplift simulation methodology"):
        st.markdown("""
        **Monte Carlo Uplift Simulation** (n=100 runs)

        1. Quality score is mapped to an uplift multiplier: high quality → positive lift, low quality → negative lift
        2. 100 stochastic simulations inject quality uncertainty (σ=0.05) and campaign variance (σ=0.03)
        3. 95% confidence intervals computed from simulation distribution
        4. Baseline calibrated to Mailchimp Gaming Industry benchmarks: Open Rate = 21%, CTR = 3.0%

        **Efficiency Savings** = (manual_mins − AI_mins) × monthly_volume ÷ 60

        *Note: Uplift figures are projections for planning purposes, not guarantees.*
        """)