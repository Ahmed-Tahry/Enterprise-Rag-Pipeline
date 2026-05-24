"""
ui/app.py — Streamlit UI for Enterprise RAG Pipeline
Premium glassmorphism design
"""

import streamlit as st
import requests
import json
import time
import plotly.graph_objects as go
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Enterprise RAG Pipeline", page_icon="🔍", layout="wide")

# ── Premium CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #172554 100%);
}
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; }
header[data-testid="stHeader"] { background: transparent; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0c0f1d 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.12);
}
section[data-testid="stSidebar"] .stMarkdown { color: #c7d2fe; }

.hero {
    text-align: center; padding: 40px 20px 20px 20px;
}
.hero h1 {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 8px; letter-spacing: -1px;
}
.hero p { color: #94a3b8; font-size: 1.15rem; font-weight: 300; }

.glass {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 16px; padding: 24px; margin: 12px 0; color: #c7d2fe;
}

.metric-glass {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.1);
    border-radius: 14px; padding: 20px; text-align: center;
}
.metric-glass .value {
    font-size: 2rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-glass .label { color: #94a3b8; font-size: 0.85rem; margin-top: 4px; }

.answer-box {
    background: rgba(99,102,241,0.06);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px; padding: 22px; color: #e0e7ff;
    line-height: 1.8; font-size: 1.05rem;
}

.risk-low {
    color: #34d399; font-weight: 700;
    background: rgba(52,211,153,0.1); padding: 2px 10px; border-radius: 12px;
}
.risk-medium {
    color: #fbbf24; font-weight: 700;
    background: rgba(251,191,36,0.1); padding: 2px 10px; border-radius: 12px;
}
.risk-high {
    color: #f87171; font-weight: 700;
    background: rgba(248,113,113,0.1); padding: 2px 10px; border-radius: 12px;
}

.source-card {
    background: rgba(255,255,255,0.03);
    border-left: 4px solid rgba(99,102,241,0.4);
    padding: 14px 18px; margin: 8px 0; border-radius: 8px;
    color: #c7d2fe; font-size: 0.9em;
    transition: transform 0.2s ease;
}
.source-card:hover { transform: translateX(4px); }
.source-cited { border-left-color: #34d399; background: rgba(52,211,153,0.04); }

.score-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 20px; text-align: center;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02); border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] { border-radius: 8px; color: #94a3b8; font-weight: 500; }
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.15) !important; color: #a5b4fc !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #818cf8, #c084fc) !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; transition: all 0.3s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important; color: #c7d2fe !important;
}

.stFileUploader {
    border-color: rgba(99,102,241,0.2) !important;
}

.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 8px !important; color: #94a3b8 !important;
}
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* Streamlit overrides */
footer, #MainMenu, .stDeployButton, div[data-testid="stDecoration"] { display: none !important; }
[data-testid="stAppViewContainer"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebarContent"] { background: transparent !important; }
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
div[data-testid="stMetricValue"] > div { color: #e2e8f0 !important; }
div[data-testid="stMetricDelta"] { color: #94a3b8 !important; }
div[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────
def api_get(endpoint: str):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=10)
        return r.json() if r.ok else None
    except Exception:
        return None

def api_post(endpoint: str, **kwargs):
    try:
        r = requests.post(f"{API_URL}{endpoint}", timeout=120, **kwargs)
        return r.json(), r.ok
    except Exception as e:
        return {"error": str(e)}, False

def risk_badge(level: str) -> str:
    cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(level, "risk-low")
    return f'<span class="{cls}">● {level}</span>'

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Enterprise RAG Pipeline</h1>
    <p>Production-grade retrieval augmented generation with hallucination detection & RAGAS evaluation</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 RAG Dashboard")
    st.caption("System metrics & models")
    st.divider()

    health = api_get("/health")
    if health:
        st.success("🟢 API Connected")
    else:
        st.error("🔴 API Offline")
        st.stop()

    stats = api_get("/stats")
    if stats:
        st.metric("Documents Indexed", stats.get("documents_indexed", 0))
        st.metric("Chunks in Index",   stats.get("chunks_in_index", 0))
        st.metric("Queries Handled",   stats.get("queries_handled", 0))
        avg_lat = stats.get("avg_latency_ms", 0)
        st.metric("Avg Latency",       f"{avg_lat:.0f}ms")

    st.divider()
    st.caption(f"LLM: `{stats.get('llm_model', '—') if stats else '—'}`")
    st.caption(f"Embedder: `{stats.get('embedding_model', '—') if stats else '—'}`")
    st.divider()
    st.caption("Built with OpenAI · FAISS · RAGAS")

    st.divider()
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:12px;padding:16px;margin-top:8px;">
        <div style="font-weight:700;font-size:1rem;color:#e2e8f0;margin-bottom:6px;">👨‍💻 Naresh</div>
        <div style="font-size:0.8rem;color:#94a3b8;line-height:1.6;">
            GenAI Engineer · Full-Stack ML<br>
            <b style="color:#818cf8;">Skills:</b> LLMs · RAG · Fine-tuning · LangChain · FastAPI · Docker<br>
            <b style="color:#818cf8;">Stack:</b> Python · OpenAI · FAISS · Qdrant · Streamlit
        </div>
        <div style="margin-top:10px;font-size:0.75rem;">
            <a href="https://github.com/Naresh1401" style="color:#818cf8;text-decoration:none;">GitHub</a>
        </div>
        <details style="margin-top:10px;">
            <summary style="color:#94a3b8;font-size:0.8rem;cursor:pointer;">🚀 More Projects</summary>
            <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px;line-height:1.8;">
                <a href="https://llm-safety-guardrails.onrender.com" style="color:#7c3aed;text-decoration:none;">LLM Safety Guardrails</a><br>
                <a href="https://text-to-sql-agent-2za9.onrender.com" style="color:#64ffda;text-decoration:none;">Text-to-SQL Agent</a><br>
                <a href="https://intelligent-document-processing-qyo8.onrender.com" style="color:#a855f7;text-decoration:none;">Intelligent Doc Processing</a><br>
                <a href="https://meeting-intelligent-platform.onrender.com" style="color:#38bdf8;text-decoration:none;">Meeting Intelligence</a><br>
                <a href="https://ai-code-review-agent-bon1.onrender.com" style="color:#10b981;text-decoration:none;">AI Code Review Agent</a><br>
                <a href="https://financial-llm-assistant.onrender.com" style="color:#f59e0b;text-decoration:none;">Financial LLM Assistant</a>
            </div>
        </details>
    </div>
    """, unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────
tab_chat, tab_ingest, tab_eval = st.tabs(["💬 Chat with Documents", "📁 Ingest & Index", "📊 RAGAS Evaluation"])

# ═══ TAB 1: CHAT ═══
with tab_chat:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### Ask your knowledge base")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        question = st.text_input(
            "Question", placeholder="What is the refund policy? / Explain the authentication flow...",
            label_visibility="collapsed",
        )
    with col2:
        top_k = st.number_input("Top K chunks", min_value=1, max_value=20, value=5)
    with col3:
        detect_halluc = st.toggle("Hallucination check", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and generating..."):
                t0 = time.time()
                result, ok = api_post(
                    "/query",
                    json={"question": question, "top_k": top_k, "detect_hallucination": detect_halluc},
                )

            if not ok:
                st.error(f"Error: {result.get('detail', result)}")
            else:
                st.markdown("#### Answer")
                if not result.get("has_answer"):
                    st.warning(result["answer"])
                else:
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                if result.get("hallucination"):
                    h = result["hallucination"]
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.markdown(
                            f"**Hallucination Risk** {risk_badge(h['risk_level'])}",
                            unsafe_allow_html=True,
                        )
                    with col_b:
                        st.markdown(
                            f'<div class="metric-glass"><div class="value">{h["faithfulness_score"]:.3f}</div>'
                            f'<div class="label">Faithfulness</div></div>',
                            unsafe_allow_html=True,
                        )
                    with col_c:
                        if h.get("flagged_claims"):
                            with st.expander(f"⚠️ {len(h['flagged_claims'])} flagged claims"):
                                for claim in h["flagged_claims"]:
                                    st.caption(f"• {claim}")

                st.caption(f"⏱ {result['latency_ms']:.0f}ms | Model: {result['model']}")

                st.markdown("#### Sources")
                for src in result.get("sources", []):
                    cited_cls = "source-cited" if src.get("cited") else ""
                    cited_icon = "✅" if src.get("cited") else "📄"
                    page_info = f" · p.{src['page']}" if src.get("page") else ""
                    section_info = f" · {src['section']}" if src.get("section") else ""
                    st.markdown(
                        f'<div class="source-card {cited_cls}">'
                        f'<b>{cited_icon} [{src["rank"]}] {src["filename"]}</b>'
                        f'{page_info}{section_info} '
                        f'<span style="color:#64748b">score: {src["score"]:.4f}</span><br>'
                        f'<i>{src["excerpt"]}</i></div>',
                        unsafe_allow_html=True,
                    )

    if "history" not in st.session_state:
        st.session_state.history = []
    if question and st.button("Add to history"):
        st.session_state.history.append(question)
    if st.session_state.history:
        with st.expander("📜 Query History"):
            for q in reversed(st.session_state.history[-10:]):
                st.caption(f"• {q}")

# ═══ TAB 2: INGEST ═══
with tab_ingest:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📁 Index Documents")
    st.markdown("Supported formats: **PDF, DOCX, HTML, TXT, Markdown**")

    uploaded = st.file_uploader(
        "Drop your documents here",
        type=["pdf", "docx", "doc", "html", "htm", "txt", "md"],
        accept_multiple_files=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded:
        st.write(f"**{len(uploaded)} file(s) selected:**")
        for f in uploaded:
            st.caption(f"  · {f.name}  ({f.size / 1024:.1f} KB)")

        if st.button("⚡ Index Documents", type="primary", use_container_width=True):
            with st.spinner(f"Indexing {len(uploaded)} file(s)..."):
                files_payload = [
                    ("files", (f.name, f.read(), f.type or "application/octet-stream"))
                    for f in uploaded
                ]
                result, ok = api_post("/ingest", files=files_payload)

            if ok:
                st.success(
                    f"✅ Indexed **{result['files_indexed']} files** → "
                    f"**{result['chunks_added']} chunks** in {result['latency_ms']:.0f}ms"
                )
                st.rerun()
            else:
                st.error(f"Indexing failed: {result.get('detail', result)}")

    st.divider()
    if st.button("🗑️ Clear Index", type="secondary"):
        result, ok = api_post("/index")
        if ok:
            st.success("Index cleared.")
            st.rerun()

# ═══ TAB 3: EVALUATION ═══
with tab_eval:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📊 RAGAS Evaluation")
    st.markdown("Measure RAG quality: faithfulness, answer relevancy, context precision")

    default_qs = (
        "What is the main topic of the indexed documents?\n"
        "What are the key concepts discussed?\n"
        "Can you summarise the main findings?"
    )
    questions_text = st.text_area("Enter one question per line", value=default_qs, height=130)
    use_ground_truth = st.checkbox("Include ground truth answers (for context recall)")

    ground_truths_text = ""
    if use_ground_truth:
        ground_truths_text = st.text_area(
            "Ground truth answers (one per line, same order as questions)", height=130,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("▶️ Run Evaluation", type="primary", use_container_width=True):
        questions = [q.strip() for q in questions_text.strip().split("\n") if q.strip()]
        ground_truths = None
        if use_ground_truth and ground_truths_text:
            ground_truths = [g.strip() for g in ground_truths_text.strip().split("\n") if g.strip()]

        if not questions:
            st.warning("Enter at least one question.")
        else:
            with st.spinner(f"Evaluating {len(questions)} questions..."):
                result, ok = api_post(
                    "/evaluate",
                    json={"questions": questions, "ground_truths": ground_truths},
                )

            if not ok:
                st.error(f"Evaluation failed: {result.get('detail', result)}")
            else:
                st.success("Evaluation complete!")

                cols = st.columns(4)
                metrics = [
                    ("Faithfulness",      result.get("faithfulness", 0)),
                    ("Answer Relevancy",  result.get("answer_relevancy", 0)),
                    ("Context Precision", result.get("context_precision", 0)),
                    ("RAGAS Score",       result.get("ragas_score", 0)),
                ]
                for col, (name, val) in zip(cols, metrics):
                    with col:
                        color = "#34d399" if val >= 0.7 else "#fbbf24" if val >= 0.5 else "#f87171"
                        st.markdown(
                            f'<div class="score-card">'
                            f'<div style="font-size:2.2em;font-weight:800;color:{color}">{val:.3f}</div>'
                            f'<div style="color:#94a3b8;font-size:0.85rem">{name}</div></div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")
                col1, col2 = st.columns(2)
                col1.metric("Latency P50", f"{result.get('latency_p50_ms', 0):.0f}ms")
                col2.metric("Latency P95", f"{result.get('latency_p95_ms', 0):.0f}ms")

                fig = go.Figure(go.Scatterpolar(
                    r=[
                        result.get("faithfulness", 0),
                        result.get("answer_relevancy", 0),
                        result.get("context_precision", 0),
                        result.get("faithfulness", 0),
                    ],
                    theta=["Faithfulness", "Answer Relevancy", "Context Precision", "Faithfulness"],
                    fill="toself",
                    line_color="#818cf8",
                    fillcolor="rgba(129,140,248,0.15)",
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.08)", color="#888"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#aaa"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    showlegend=False, title="RAGAS Metric Radar", height=380,
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("📋 Raw JSON Results"):
                    st.json(result)
