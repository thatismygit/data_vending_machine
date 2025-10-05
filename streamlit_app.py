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

# -------------------- Logging --------------------
logger = logging.getLogger("data_vending")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -------------------- Helpers --------------------
def flatten_to_text(obj: Any) -> str:
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
    if not text:
        return ""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    inline = re.findall(r"`([^`]{10,})`", text)
    for m in inline:
        if re.search(r"\bselect\b", m, re.IGNORECASE):
            return m.strip()
    start = re.search(r"\bselect\b", text, re.IGNORECASE)
    if start:
        return text[start.start():].strip()
    return ""


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
### 👋 Hi — I'm the **Data Vending Machine Assistant**

I help you explore your database using **plain English** and convert your requests into **PostgreSQL queries**.

You can try things like:

```text
show tables
show first 10 rows of customers
average revenue per region

"""


# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Data Vending Machine", layout="wide")
st.title("Data Vending Machine")
st.caption("Turn plain English into safe, paginated SQL results — with explanations")
st.markdown(
    """
    <style>
    body { background: #0b1220; color: #e6eef8; }
    .card { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }
    .user-bubble { background: linear-gradient(90deg,#4f46e5,#06b6d4); color: white; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:88%; }
    .bot-bubble { background: rgba(255,255,255,0.06); color: #e6eef8; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:88%; }
    .chat-container { max-height: calc(100vh - 200px); overflow: auto; padding-right: 6px; }
    #controls-panel { position: sticky; top: 20px; z-index: 999; }
    @media (max-width: 900px) {
        #controls-panel { position: static; margin-top: 12px; }
        .chat-container { max-height: none; overflow: visible; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- State --------------------
for key, val in {
    "chat_history": [],
    "last_sql": "",
    "last_result": None,
    "edit_mode": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------- Layout --------------------
left_col, right_col = st.columns([2.6, 1])

# -------- Left: Chat --------
with left_col:
    st.markdown("<div class='card chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role, content = msg["role"], msg["content"]
        justify = "flex-end" if role == "user" else "flex-start"
        bubble_class = "user-bubble" if role == "user" else "bot-bubble"
        if role == "assistant":
            st.markdown(
                f"<div class='{bubble_class}' style='width:88%'>",
                unsafe_allow_html=True,
            )
            st.markdown(content, unsafe_allow_html=False)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div class='{bubble_class}'>{content}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question or describe a query:", height=110, key="chat_input")
        col1, col2, col3 = st.columns(3)
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

        # placeholder agent
        async def agent_call(text):
            """Generate SQL using ChatGPT, fallback to rule-based generator."""
            model = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0)
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
        flat = flatten_to_text(resp)
        sql = resp.get("sql", "")
        if not sql:
            try:
                sql = pgget.get_query(prompt)
            except Exception as e:
                logger.warning(f"fallback failed {e}")
        st.session_state.last_sql = sql or ""
        st.session_state.chat_history.append({"role": "assistant", "content": flat.replace("\n", "<br>")})
        st.rerun()

# -------- Right: Controls --------
with right_col:
    st.markdown("<div id='controls-panel'><div class='card'>", unsafe_allow_html=True)
    st.header("🧠 SQL Candidate")

    # Edit mode toggle
    if st.session_state.edit_mode:
        new_sql = st.text_area("Edit SQL:", value=st.session_state.last_sql, height=150, key="sql_edit_box")
        save_btn = st.button("💾 Save Query")
        if save_btn:
            st.session_state.last_sql = new_sql.strip()
            st.session_state.edit_mode = False
            st.success("Query updated!")
            st.rerun()
    else:
        if st.session_state.last_sql:
            st.code(st.session_state.last_sql, language="sql")
        else:
            st.info("No SQL yet. Ask the assistant or describe a query.")

    # Buttons: Run Query & Edit Query
    run_col, edit_col = st.columns([1, 1])
    run_query = run_col.button("▶️ Run Query")
    edit_query = edit_col.button("✏️ Edit Query")

    if edit_query:
        st.session_state.edit_mode = True
        st.rerun()

    if run_query and st.session_state.last_sql:
        with st.spinner("Running query..."):
            result = run_executor_raw(st.session_state.last_sql)
        if result["stderr"]:
            st.warning(result["stderr"])
        try:
            data = json.loads(result["stdout"]) if result["stdout"].strip() else []
            st.session_state.last_result = data
        except Exception:
            st.session_state.last_result = result["stdout"]
        st.success("Query executed!")
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Query Result")
    if st.session_state.last_result is None:
        st.write("No results yet. Run a query to preview results.")
    else:
        res = st.session_state.last_result
        if isinstance(res, list):
            st.dataframe(res[:50])
        else:
            st.text(res)

    st.markdown("</div></div>", unsafe_allow_html=True)

# Footer
st.caption("made by Vishnu Pandey.")
