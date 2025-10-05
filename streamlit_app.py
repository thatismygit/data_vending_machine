# streamlit_app.py

import os
import sys
import re
import json
import subprocess
import asyncio
import logging
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

# Import local helper for fallback NL→SQL
import postgres_get_query as pgget

# Load environment variables
load_dotenv(override=True)

# -------------------------
# Logging Setup
# -------------------------
logger = logging.getLogger("data_vending")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -------------------------
# Utility functions
# -------------------------
def flatten_to_text(obj: Any) -> str:
    """Recursively flatten an object to plain text."""
    parts: List[str] = []

    def walk(x):
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
        elif isinstance(x, (int, float, bool)):
            parts.append(str(x))
        elif isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str):
                    parts.append(k)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for it in x:
                walk(it)
        else:
            try:
                if hasattr(x, "content"):
                    walk(getattr(x, "content"))
                elif hasattr(x, "text"):
                    walk(getattr(x, "text"))
                elif hasattr(x, "__dict__"):
                    walk(vars(x))
                else:
                    parts.append(str(x))
            except Exception:
                parts.append(str(x))

    walk(obj)
    return "\n".join([p for p in parts if p])

def extract_sql_from_text(text: str) -> str:
    """Extract SQL code from text fenced in ```sql``` or similar."""
    if not text:
        return ""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = re.search(r"\b(select|insert|update|delete|create|with)\b", text, re.IGNORECASE)
    if start:
        i = start.start()
        return text[i:].strip()
    return ""

def run_executor_raw(sql: str, timeout: int = 60) -> Dict[str, Any]:
    """Execute SQL by delegating to the helper script."""
    cmd = [sys.executable, "postgres_execute_query.py", "--run-sql"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=sql.encode("utf-8"), timeout=timeout)
        return {"stdout": stdout.decode(), "stderr": stderr.decode(), "rc": proc.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "rc": -1}

# -------------------------
# Streamlit UI Setup
# -------------------------
st.set_page_config(page_title="Data Vending Machine", layout="wide")

st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg, #0f172a 0%, #071029 100%); color: #e6eef8; }
    .app-bg { background: rgba(255,255,255,0.03); padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
    .user-bubble { background: linear-gradient(90deg,#4f46e5,#06b6d4); color: white; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:80%; }
    .bot-bubble { background: rgba(255,255,255,0.06); color: #e6eef8; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:80%; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2>Data Vending Machine</h2>", unsafe_allow_html=True)

# -------------------------
# Initialize Session State
# -------------------------
for key, default in {
    "chat_history": [],
    "last_sql": "",
    "last_result": None,
    "page_idx": 0,
    "total_rows": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -------------------------
# Layout
# -------------------------
left_col, right_col = st.columns([2.5, 1])

with left_col:
    st.markdown("<div class='app-bg'>", unsafe_allow_html=True)

    # Display chat messages
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        bubble = "user-bubble" if role == "user" else "bot-bubble"
        justify = "flex-end" if role == "user" else "flex-start"
        st.markdown(
            f"<div style='display:flex;justify-content:{justify};margin:6px 0'><div class='{bubble}'>{content}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_area("Ask a question or describe a query:", key="chat_input", height=100)
        send_col, regen_col, clear_col = st.columns(3)
        send = send_col.form_submit_button("Send")
        regen = regen_col.form_submit_button("Regenerate")
        clear = clear_col.form_submit_button("Clear Chat")

    if clear:
        st.session_state.chat_history = []
        st.session_state.last_sql = ""
        st.session_state.last_result = None
        st.rerun()

    # Determine user message
    user_prompt = None
    if send and prompt.strip():
        user_prompt = prompt.strip()
    elif regen:
        for m in reversed(st.session_state.chat_history):
            if m["role"] == "user":
                user_prompt = m["content"]
                break

    # Handle sending
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Replace this block with your real agent logic (for now it just simulates)
        def mock_agent_call(prompt_text):
            return {"content": f"Here’s a SQL for your request:\n```sql\nSELECT * FROM demo_table LIMIT 10;\n```"}

        with st.spinner("Thinking..."):
            response = mock_agent_call(user_prompt)

        flat = flatten_to_text(response)
        sql = extract_sql_from_text(flat)

        if not sql:
            try:
                sql = pgget.get_query(user_prompt)
            except Exception as e:
                logger.warning(f"Fallback failed: {e}")

        st.session_state.last_sql = sql or ""
        st.session_state.chat_history.append({"role": "assistant", "content": flat.replace('\n', '<br>')})
        st.rerun()

# -------------------------
# Right column: Controls & Results
# -------------------------
with right_col:
    st.markdown("<div class='app-bg'>", unsafe_allow_html=True)
    st.subheader("⚙️ Controls")

    page_size = st.number_input("Rows per page", 10, 500, 50, 10)
    search_term = st.text_input("Search term")
    st.markdown("---")

    st.subheader("🧠 SQL Candidate")
    if st.session_state.last_sql:
        st.code(st.session_state.last_sql, language="sql")
    else:
        st.info("No SQL generated yet.")

    st.markdown("---")
    execute = st.button("▶️ Execute SQL")

    if execute and st.session_state.last_sql:
        with st.spinner("Running query..."):
            result = run_executor_raw(st.session_state.last_sql)
        if result["stderr"]:
            st.warning(result["stderr"])
        try:
            data = json.loads(result["stdout"]) if result["stdout"].strip() else []
            st.session_state.last_result = data
        except Exception:
            st.session_state.last_result = result["stdout"]
        st.rerun()

    if st.session_state.last_result:
        st.markdown("---")
        st.subheader("📊 Result Preview")
        res = st.session_state.last_result
        if isinstance(res, list):
            st.dataframe(res[:page_size])
        else:
            st.text(res)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.caption("Made by Vishnu Pandey")
