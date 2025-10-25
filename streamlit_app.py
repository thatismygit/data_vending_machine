import os
import sys
import re
import json
import subprocess
import logging
from typing import Any, Dict, List, Optional
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import asyncio
from client import flatten_to_text, extract_sql_from_text
import postgres_get_query as pgget

load_dotenv(override=True)

# Compatibility fix for Streamlit rerun deprecation
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = st.rerun

# -------------------- Logging --------------------
logger = logging.getLogger("data_vending")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -------------------- Helpers --------------------
def run_executor_raw(sql: str, timeout: int = 120) -> Dict[str, Any]:
    cmd = [sys.executable, "postgres_execute_query.py", "--run-sql"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=sql.encode("utf-8"), timeout=timeout)
        return {"stdout": stdout.decode(), "stderr": stderr.decode(), "rc": proc.returncode}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "rc": -1}

# -------------------- Greeting detection --------------------
GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|hiya|who are you|introduce yourself|what are you)\b",
    re.IGNORECASE,
)

ASSISTANT_INTRO = """
## 🧠 Welcome to the **Data Vending Machine**  

#### Your bridge between **plain English** and **PostgreSQL**.  
#### Ask me questions about your data — I’ll translate them into precise SQL queries you can run instantly.  

For example:
```
show all tables
what are the columns in products
create table employees with id, name, email
insert into users (name, email) values ('Alice', 'a@b.com')
describe table101
"""

# -------------------- Streamlit Setup (minimal chat-like) --------------------
st.set_page_config(page_title="Data Vending Machine", layout="wide")
# Minimal header similar to ChatGPT
st.markdown(
    """
    <style>
    body { background: #0b1220; color: #e6eef8; }
    .chat-card { background: rgba(255,255,255,0.02); padding: 18px; border-radius: 12px; }
    .user-bubble { background: linear-gradient(90deg,#4f46e5,#06b6d4); color: white; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:88%; }
    .bot-bubble { background: rgba(255,255,255,0.03); color: #e6eef8; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:88%; }
    .chat-container { max-height: calc(100vh - 220px); overflow: auto; padding-right: 6px; }
    .input-area { position: sticky; bottom: 0; background: transparent; padding-top: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Data Vending Machine")
st.caption("Ask in plain English. (SQL candidate & results are available in an expandable advanced panel)")

# -------------------- State --------------------
for key, val in {
    "chat_history": [],
    "last_sql": "",
    "last_result": None,
    "edit_mode": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------- Layout: single-column chat --------------------
container = st.container()
with container:
    st.markdown("<div class='chat-card chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role, content = msg["role"], msg["content"]
        if role == "assistant":
            st.markdown(f"<div class='bot-bubble' style='width:88%'>", unsafe_allow_html=True)
            st.markdown(content, unsafe_allow_html=False)
            st.markdown("<hr style='border:0;border-top:1px solid #222;margin:8px 0;'>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='user-bubble'>{content}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("", height=120, placeholder="Ask a question or describe a query...", key="chat_input")
        col1, col2, col3 = st.columns([1,1,1])
        send = col1.form_submit_button("Send")
        regen = col2.form_submit_button("Regenerate")
        clear = col3.form_submit_button("Clear Chat")

    if clear:
        for k in ["chat_history", "last_sql", "last_result"]:
            st.session_state[k] = [] if k == "chat_history" else None
        st.rerun()

    prompt = None
    if send and user_input.strip():
        prompt = user_input.strip()
    elif regen:
        for m in reversed(st.session_state.chat_history):
            if m["role"] == "user":
                prompt = m["content"]
                break

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        if GREETING_RE.search(prompt):
            st.session_state.chat_history.append({"role": "assistant", "content": ASSISTANT_INTRO})
            st.session_state.last_sql = ""
            st.rerun()

        async def agent_call(text):
            model = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0,
            )
            try:
                response = await model.ainvoke(f"Convert this into SQL query for PostgreSQL:\n{text}")
                flat = flatten_to_text(response)
                sql = extract_sql_from_text(flat)
                if not sql:
                    sql = pgget.get_query(text)
                return {"content": flat, "sql": sql}
            except Exception as e:
                return {"content": f"Error generating SQL: {e}", "sql": pgget.get_query(text)}

        with st.spinner("Thinking..."):
            resp = asyncio.run(agent_call(prompt))
        flat = resp.get("content", "")
        sql = resp.get("sql", "")
        if not sql:
            try:
                sql = pgget.get_query(prompt)
            except Exception as e:
                logger.warning(f"fallback failed {e}")
        st.session_state.last_sql = sql or ""
        st.session_state.chat_history.append({"role": "assistant", "content": flat})
        st.experimental_rerun()

        # --- Inline Edit Box ---
        if st.session_state.edit_mode:
            st.markdown("### ✍️ Edit SQL")
            edited_sql = st.text_area(
                "Modify the SQL below and click Save",
                value=st.session_state.last_sql,
                height=160,
                key="edit_sql_box",
            )
            save_btn = st.button("💾 Save Query")
            if save_btn:
                st.session_state.last_sql = edited_sql.strip()
                st.session_state.edit_mode = False
                st.success("Query updated — now press Run Query.")
                st.rerun()

        # --- Display Query Results ---
        if st.session_state.last_result is not None:
            st.markdown("### 📊 Query Result")
            if isinstance(st.session_state.last_result, list):
                st.dataframe(st.session_state.last_result[:50])
            else:
                st.text(st.session_state.last_result)
        # --------------------------------------------------------------------
# 🧠 Display SQL candidate + buttons and results (runs every rerun)
# --------------------------------------------------------------------
if st.session_state.last_sql:
    st.markdown("### 💡 SQL Candidate (generated)")
    st.code(st.session_state.last_sql, language="sql")

    run_now = st.button("▶️ Run Query", key="run_now")
    edit_now = st.button("✏️ Edit Query", key="edit_now")

    # --- Run Query ---
    if run_now:
        with st.spinner("Running query..."):
            result = run_executor_raw(st.session_state.last_sql)

        if result["stderr"]:
            st.warning(result["stderr"])
        else:
            try:
                data = json.loads(result["stdout"]) if result["stdout"].strip() else []
                st.session_state.last_result = data
            except Exception:
                st.session_state.last_result = result["stdout"]
            st.success("Query executed successfully!")
        st.experimental_rerun()

    # --- Edit Query ---
    if edit_now:
        st.session_state.edit_mode = True
        st.experimental_rerun()

# --- Inline Edit Box ---
if st.session_state.edit_mode:
    st.markdown("### ✍️ Edit SQL")
    edited_sql = st.text_area(
        "Modify the SQL below and click Save",
        value=st.session_state.last_sql,
        height=160,
        key="edit_sql_box",
    )
    save_btn = st.button("💾 Save Query")
    if save_btn:
        st.session_state.last_sql = edited_sql.strip()
        st.session_state.edit_mode = False
        st.success("Query updated — now press Run Query.")
        st.experimental_rerun()

# --- Display Query Results ---
if st.session_state.last_result is not None:
    st.markdown("### 📊 Query Result")
    if isinstance(st.session_state.last_result, list):
        st.dataframe(st.session_state.last_result[:50])
    else:
        st.text(st.session_state.last_result)


st.caption("made by Vishnu Pandey.")