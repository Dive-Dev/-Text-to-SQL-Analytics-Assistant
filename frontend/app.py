import os
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Text-to-SQL Analytics Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Custom Modern Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #713F12;
    }
    .badge-info {
        background-color: #E1EFFE;
        color: #1E429F;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .sql-code-box {
        background-color: #0F172A;
        color: #38BDF8;
        padding: 12px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.9rem;
        margin-top: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# Helper Functions to interact with Backend
def check_backend_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_schema():
    try:
        r = requests.get(f"{BACKEND_URL}/schema", timeout=3.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def run_nl_query(question, provider, model, temperature, api_key=None):
    payload = {
        "question": question,
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "api_key": api_key or None,
    }
    try:
        r = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=300.0)
        if r.status_code == 200:
            return r.json()
        return {"success": False, "error": f"API HTTP {r.status_code}: {r.text}"}
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Query generation timed out after 5 minutes. If running a local LLM on CPU, the model may be under heavy load. Switch to 'Local Standby / Heuristic Engine' in the sidebar for instant zero-latency responses, or ensure Ollama is using GPU acceleration.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": f"Cannot connect to FastAPI backend at {BACKEND_URL}. Ensure the backend is running (`python run.py --backend`).",
        }
    except Exception as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}


def execute_raw_sql(sql):
    try:
        r = requests.post(f"{BACKEND_URL}/execute", json={"sql": sql}, timeout=30.0)
        if r.status_code == 200:
            return r.json()
        return {"success": False, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        return {"success": False, "error": f"Error: {str(e)}"}


def upload_file_to_backend(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        r = requests.post(f"{BACKEND_URL}/upload-dataset", files=files, timeout=30.0)
        return r.status_code == 200, r.json()
    except Exception as e:
        return False, {"detail": str(e)}


# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_sample_prompt" not in st.session_state:
    st.session_state.selected_sample_prompt = None

# Header
st.markdown('<div class="main-header">⚡ Text-to-SQL Analytics Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Localized, privacy-first data querying with DuckDB, Ollama <code>duckdb-nsql:7b</code>, and self-correcting QA validation loop</div>',
    unsafe_allow_html=True,
)

# Backend Health Check
health_data = check_backend_health()

# Top Status Indicators
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    if health_data:
        st.markdown(
            '<div class="metric-card"><span class="status-badge badge-success">● Active</span><br><b>Backend API</b><br><small>FastAPI on :8000</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><span class="status-badge badge-warning">○ Standby</span><br><b>Backend API</b><br><small>Start via run.py</small></div>',
            unsafe_allow_html=True,
        )

with col_stat2:
    if health_data and health_data.get("duckdb_connected"):
        st.markdown(
            f'<div class="metric-card"><span class="status-badge badge-success">● Connected</span><br><b>DuckDB Engine</b><br><small>{health_data.get("tables_count", 0)} Tables Loaded</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><span class="status-badge badge-warning">○ Checking</span><br><b>DuckDB Engine</b><br><small>In-Memory / Parquet</small></div>',
            unsafe_allow_html=True,
        )

with col_stat3:
    if health_data and health_data.get("ollama_connected"):
        st.markdown(
            f'<div class="metric-card"><span class="status-badge badge-success">● Ollama Online</span><br><b>Local LLM</b><br><small>{health_data.get("active_model")}</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="metric-card"><span class="status-badge badge-info">⚡ Standby / Heuristic</span><br><b>LLM Engine</b><br><small>Ollama / Offline Fallback</small></div>',
            unsafe_allow_html=True,
        )

with col_stat4:
    st.markdown(
        '<div class="metric-card"><span class="status-badge badge-success">🛡️ Active</span><br><b>QA Sandbox</b><br><small>DML Block & EXPLAIN Dry-Run</small></div>',
        unsafe_allow_html=True,
    )

st.write("")

# SIDEBAR: Database Schema Explorer & Settings
with st.sidebar:
    st.header("🗄️ Database Explorer")
    schema_data = fetch_schema()

    if schema_data and schema_data.get("tables"):
        tables = schema_data["tables"]
        st.caption(f"**{len(tables)} Active Analytical Tables:**")
        for tbl in tables:
            with st.expander(f"📁 {tbl['table_name']} ({tbl['row_count']} rows)"):
                col_defs = []
                for c in tbl["columns"]:
                    sample_txt = f" (eg: {', '.join(c['sample_values'][:2])})" if c["sample_values"] else ""
                    col_defs.append(f"- **`{c['name']}`** `{c['type']}`{sample_txt}")
                st.markdown("\n".join(col_defs))
    else:
        st.info("No tables discovered yet. Ensure backend is running.")

    st.divider()

    st.header("📤 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Parquet into DuckDB", type=["csv", "parquet"])
    if uploaded_file is not None:
        if st.button("Load Table into DuckDB", key="btn_upload"):
            with st.spinner("Ingesting into DuckDB..."):
                ok, res = upload_file_to_backend(uploaded_file)
                if ok:
                    st.success(f"Loaded '{res.get('table_name')}' with {res.get('row_count')} rows!")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {res.get('detail', 'Unknown error')}")

    st.divider()

    st.header("⚙️ Model Settings")
    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "offline_demo", "openai"],
        format_func=lambda x: {
            "ollama": "Ollama (duckdb-nsql:7b - Recommended)",
            "offline_demo": "Local Standby / Heuristic Engine",
            "openai": "OpenAI / Custom API",
        }.get(x, x),
    )

    model_name = "duckdb-nsql"
    api_key = None
    if provider == "ollama":
        model_name = st.text_input("Ollama Model Name", value="duckdb-nsql")
        st.caption("Ollama service active with `duckdb-nsql`")
    elif provider == "openai":
        api_key = st.text_input("API Key", type="password")
        model_name = st.text_input("Model", value="gpt-4o-mini")

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    with st.expander("🛠️ Direct SQL Console"):
        direct_sql = st.text_area("Write SQL to execute on DuckDB", value="SELECT * FROM e_commerce_sales LIMIT 5;")
        if st.button("Execute SQL"):
            res = execute_raw_sql(direct_sql)
            if res.get("success"):
                st.dataframe(pd.DataFrame(res.get("rows")), use_container_width=True)
            else:
                st.error(res.get("error"))

# Quick Query Suggestions
st.markdown("##### 💡 Suggested Analytical Questions:")
sample_queries = [
    "What are the top 5 customers by total revenue?",
    "Show total sales and items sold by product category",
    "List all failed tests with their suite and error type",
    "Compare test execution count and failure rate by browser",
    "Show total revenue and orders count by country",
]

chip_cols = st.columns(len(sample_queries))
for idx, (c_col, prompt_text) in enumerate(zip(chip_cols, sample_queries)):
    with c_col:
        if st.button(f"📌 {prompt_text[:28]}...", key=f"chip_{idx}", help=prompt_text):
            st.session_state.selected_sample_prompt = prompt_text

# Query Input
prompt_input_val = st.session_state.selected_sample_prompt or ""
user_question = st.chat_input("Ask a question about your data in plain English...")

# If chip was clicked or chat input submitted
query_to_run = user_question or st.session_state.selected_sample_prompt
if st.session_state.selected_sample_prompt and not user_question:
    # Clear the temporary chip state
    st.session_state.selected_sample_prompt = None

if query_to_run:
    with st.spinner("Analyzing question, generating SQL & validating execution plan..."):
        response = run_nl_query(
            question=query_to_run,
            provider=provider,
            model=model_name,
            temperature=temperature,
            api_key=api_key,
        )

        st.session_state.chat_history.insert(0, response)

# Display Query Results & History
if not st.session_state.chat_history:
    st.info(
        "👋 Welcome! Type an analytical question in the chat bar below or click one of the suggested prompts above to get started."
    )
else:
    for idx, item in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"### 💬 Question: *\"{item.get('question', '')}\"*")

            if not item.get("success"):
                st.error(f"❌ **Validation Error**: {item.get('error')}")
                if item.get("sql"):
                    st.code(item.get("sql"), language="sql")
                st.divider()
                continue

            # Success response metrics
            q_time = item.get("execution_time_ms", 0)
            rows_num = item.get("row_count", 0)
            attempts = item.get("validation_attempts", [])
            source = item.get("llm_source", "LLM")

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Rows Returned", f"{rows_num:,}")
            with m_col2:
                st.metric("Execution Time", f"{q_time} ms")
            with m_col3:
                attempts_count = len(attempts)
                if attempts_count == 1:
                    st.metric("QA Validation", "Passed (1 Attempt)", delta="Valid Plan")
                else:
                    st.metric("QA Validation", f"Corrected ({attempts_count} Attempts)", delta="Self-Corrected")
            with m_col4:
                st.metric("Engine / Source", source)

            # Generated SQL box
            sql_query = item.get("sql", "")
            with st.expander("🔍 View Generated SQL & QA Validation Audit Trail", expanded=False):
                st.markdown("**Executed SQL Query:**")
                st.code(sql_query, language="sql")

                st.markdown("**QA Validation Log:**")
                for att in attempts:
                    att_num = att.get("attempt_number")
                    is_val = att.get("valid")
                    if is_val:
                        st.success(f"✓ **Attempt {att_num} passed:** DuckDB execution plan verified.")
                        if att.get("execution_plan"):
                            st.caption(f"Plan: `{att.get('execution_plan')[:180]}...`")
                    else:
                        st.warning(f"⚠️ **Attempt {att_num} failed:** {att.get('error_message')}")
                        st.code(att.get("sql"), language="sql")

            # Data & Chart Tabs
            tab_data, tab_chart = st.tabs(["📊 Data Table", "📈 Visualization"])

            df = pd.DataFrame(item.get("rows", []))

            with tab_data:
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                    csv_data = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Result as CSV",
                        data=csv_data,
                        file_name="query_results.csv",
                        mime="text/csv",
                        key=f"dl_{idx}",
                    )
                else:
                    st.info("Query returned 0 rows.")

            with tab_chart:
                if not df.empty and len(df.columns) >= 2:
                    chart_sug = item.get("chart_suggestion") or {}
                    sug_type = chart_sug.get("chart_type", "bar")
                    x_col = chart_sug.get("x_axis", df.columns[0])
                    y_col = chart_sug.get("y_axis", df.columns[1] if len(df.columns) > 1 else df.columns[0])

                    # Chart customization toolbar
                    c_type_col, c_x_col, c_y_col = st.columns(3)
                    with c_type_col:
                        selected_type = st.selectbox(
                            "Chart Type",
                            options=["bar", "line", "pie", "scatter"],
                            index=["bar", "line", "pie", "scatter"].index(sug_type)
                            if sug_type in ["bar", "line", "pie", "scatter"]
                            else 0,
                            key=f"ctype_{idx}",
                        )
                    with c_x_col:
                        selected_x = st.selectbox(
                            "X Axis / Category",
                            options=df.columns.tolist(),
                            index=df.columns.tolist().index(x_col) if x_col in df.columns else 0,
                            key=f"cx_{idx}",
                        )
                    with c_y_col:
                        # Prioritize numeric columns for Y axis
                        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                        y_options = num_cols if num_cols else df.columns.tolist()
                        selected_y = st.selectbox(
                            "Y Axis / Values",
                            options=y_options,
                            index=y_options.index(y_col) if y_col in y_options else 0,
                            key=f"cy_{idx}",
                        )

                    # Render Plotly Chart
                    try:
                        title_str = f"{selected_y.replace('_', ' ').title()} by {selected_x.replace('_', ' ').title()}"
                        if selected_type == "bar":
                            fig = px.bar(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=title_str,
                                text_auto=True,
                                color_discrete_sequence=["#3B82F6"],
                            )
                        elif selected_type == "line":
                            fig = px.line(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=title_str,
                                markers=True,
                                color_discrete_sequence=["#10B981"],
                            )
                        elif selected_type == "pie":
                            fig = px.pie(
                                df,
                                names=selected_x,
                                values=selected_y,
                                title=title_str,
                                hole=0.3,
                            )
                        elif selected_type == "scatter":
                            fig = px.scatter(
                                df,
                                x=selected_x,
                                y=selected_y,
                                title=title_str,
                                size_max=12,
                                color_discrete_sequence=["#6366F1"],
                            )

                        fig.update_layout(
                            template="plotly_white",
                            margin=dict(l=20, r=20, t=40, b=20),
                            height=420,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not render chart automatically: {e}")
                else:
                    st.info("Single column or empty result; visualization not applicable.")

            st.divider()
