# streamlit_app.py

import os
import sys
import re
import json
import subprocess
import logging
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# local fallback: natural language -> SQL
import postgres_get_query as pgget

load_dotenv(override=True)

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("data_vending")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -------------------------
# Utilities
# -------------------------
def flatten_to_text(obj: Any) -> str:
    """Convert agent response objects to readable text (robust)."""
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

SQL_KEYWORDS_RE = re.compile(r"\b(select|insert|update|delete|create|alter|drop|with)\b", re.IGNORECASE)

def is_probable_sql(candidate: str) -> bool:
    if not candidate or len(candidate.strip()) < 8:
        return False
    if not SQL_KEYWORDS_RE.search(candidate):
        return False
    return True

def extract_sql_from_text(text: str) -> str:
    """Extract SQL fenced or inline or first plausible SQL snippet."""
    if not text:
        return ""
    # fenced block
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        cand = fence.group(1).strip()
        return cand if is_probable_sql(cand) else ""
    # inline code with backticks
    inline = re.findall(r"`([^`]{10,})`", text)
    for m in inline:
        if is_probable_sql(m):
            return m.strip()
    # find the first SQL keyword occurrence and return up to a semicolon or some chars
    start = re.search(r"\b(select|create|insert|update|delete|with)\b", text, re.IGNORECASE)
    if start:
        i = start.start()
        sem = text.find(";", i)
        if sem != -1:
            cand = text[i:sem+1].strip()
            return cand if is_probable_sql(cand) else ""
        # fallback: return remainder if looks like SQL
        cand = text[i:i+2000].strip()
        return cand if is_probable_sql(cand) else ""
    return ""

def run_executor_raw(sql: str, timeout: int = 120) -> Dict[str, Any]:
    """Run the SQL using the helper script (postgres_execute_query.py)."""
    cmd = [sys.executable, "postgres_execute_query.py", "--run-sql"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=sql.encode("utf-8"), timeout=timeout)
        rc = proc.returncode
        return {"rc": rc, "stdout": stdout.decode("utf-8", errors="replace"), "stderr": stderr.decode("utf-8", errors="replace")}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"rc": -1, "stdout": "", "stderr": "timeout"}
    except FileNotFoundError:
        return {"rc": -1, "stdout": "", "stderr": "postgres_execute_query.py not found"}
    except Exception as e:
        return {"rc": -1, "stdout": "", "stderr": str(e)}

# -------------------------
# Greeting detection & canned intro
# -------------------------
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|hiya)\b",
    r"\bwho\s+are\s+you\b",
    r"\bintroduce yourself\b",
    r"\bwhat\s+are\s+you\b",
    r"^\s*(good\s+morning|good\s+afternoon|good\s+evening)\b",
    r"^\s*hey,\s*who\s+are\s+you\b",
]
GREETING_RE = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)

ASSISTANT_INTRO = (
    "Hi — I'm the Data Vending Machine assistant. I can help you explore your database with natural language. "
    "<br><br>"
    "Examples you can try:<br>"
    "- \"list tables\"<br>"
    "- \"show me the first 10 rows of orders\"<br>"
    "- \"count customers per country\"<br><br>"
    "If you'd like to run a query, describe it and I'll propose SQL (and you can execute safely)."
)

def is_greeting_or_identity_query(text: str) -> bool:
    if not text:
        return False
    return bool(GREETING_RE.search(text))

# -------------------------
# Streamlit UI & state
# -------------------------
st.set_page_config(page_title="Data Vending Machine", layout="wide")

# CSS for layout & aesthetics
st.markdown(
    """
    <style>
    /* page and card visuals */
    body { background: #0b1220; color: #e6eef8; }
    .card { background: rgba(255,255,255,0.03); padding: 14px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); }

    /* chat bubbles */
    .user-bubble { background: linear-gradient(90deg,#4f46e5,#06b6d4); color: white; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:88%; }
    .bot-bubble { background: rgba(255,255,255,0.06); color: #e6eef8; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:88%; }

    /* left chat container scrollable so right panel remains visible */
    .chat-container {
        max-height: calc(100vh - 200px); 
        overflow: auto;
        padding-right: 6px;
    }

    /* sticky controls on the right */
    #controls-panel {
        position: sticky;
        top: 20px;
        z-index: 999;
    }

    /* responsive: on small screens stack controls below chat (normal flow) */
    @media (max-width: 900px) {
        #controls-panel { position: static; margin-top: 12px; }
        .chat-container { max-height: none; overflow: visible; }
    }

    /* make code blocks more compact */
    .stCodeBlock pre { background: rgba(0,0,0,0.35); padding:10px; border-radius:8px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle
st.markdown("<h1 style='margin-bottom:6px'>Data Vending Machine</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#9fb0d6;margin-bottom:14px'>Conversational SQL explorer — ask a question, or say hi</div>", unsafe_allow_html=True)

# Initialize session state safely (before widgets are created)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of dicts: {role, content}
if "last_sql" not in st.session_state:
    st.session_state["last_sql"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "page_idx" not in st.session_state:
    st.session_state["page_idx"] = 0
if "visible_columns" not in st.session_state:
    st.session_state["visible_columns"] = None
if "total_rows" not in st.session_state:
    st.session_state["total_rows"] = None

# Layout: left chat, right sticky controls
left_col, right_col = st.columns([2.6, 1])

# LEFT: chat area
with left_col:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Chat messages wrapped in a scrollable container so right panel stays visible while we scroll chat
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.chat_history:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(f"<div style='display:flex;justify-content:flex-end;margin:10px 0'><div class='user-bubble'>{content}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='display:flex;justify-content:flex-start;margin:10px 0'><div class='bot-bubble'>{content}</div></div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input form (clear_on_submit avoids modifying widget-backed session_state afterwards)
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask a question or describe a query:", height=110, key="chat_input")
        col1, col2, col3 = st.columns([1, 1, 1])
        send = col1.form_submit_button("Send")
        regenerate = col2.form_submit_button("Regenerate")
        clear_chat = col3.form_submit_button("Clear Chat")

    if clear_chat:
        # safe session-state mutations only
        st.session_state.chat_history = []
        st.session_state.last_sql = ""
        st.session_state.last_result = None
        st.session_state.visible_columns = None
        st.session_state.page_idx = 0
        st.session_state.total_rows = None
        st.rerun()

    # Determine prompt to send (send or regenerate last)
    prompt_to_send: Optional[str] = None
    if send and user_input and user_input.strip():
        prompt_to_send = user_input.strip()
    elif regenerate:
        for m in reversed(st.session_state.chat_history):
            if m.get("role") == "user":
                prompt_to_send = m.get("content")
                break

    # Handle sending prompt
    if prompt_to_send:
        # append user message
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_send})

        # if greeting/identity -> canned intro and do not attempt SQL
        if is_greeting_or_identity_query(prompt_to_send):
            st.session_state.chat_history.append({"role": "assistant", "content": ASSISTANT_INTRO})
            st.session_state.last_sql = ""
            st.rerun()

        # Otherwise call agent (placeholder). Replace this with your ask_agent_sync if you want.
        def agent_call(prompt_text: str) -> Dict[str, Any]:
            """
            Placeholder agent call. Replace this with your real agent invocation (ask_agent_sync).
            Should return an object that flatten_to_text() will convert to a string (or a dict with 'content').
            """
            lower = prompt_text.lower()
            if "tables" in lower or "list tables" in lower or "schema" in lower:
                # helpful non-sql response with hint
                return {"content": "Check the SQL Query box on the right."}
            # Generic SQL suggestion (if relevant). If nothing relevant, respond with a human message
            # Here we choose a safe non-SQL fallback message to avoid spamming SQL for small chit-chat.
            return {"content": "I can help build SQL for database questions — please describe what data you'd like."}

        with st.spinner("Thinking..."):
            try:
                raw_response = agent_call(prompt_to_send)
            except Exception as e:
                logger.exception("Agent invocation failed: %s", e)
                st.session_state.chat_history.append({"role": "assistant", "content": f"Agent failed: {e}"})
                st.rerun()

        flat_text = flatten_to_text(raw_response)
        sql_candidate = extract_sql_from_text(flat_text)

        # fallback NL->SQL helper if agent returned nothing SQL-like
        used_fallback = False
        if not sql_candidate:
            try:
                fb = pgget.get_query(prompt_to_send)
                if fb:
                    sql_candidate = fb
                    used_fallback = True
            except Exception as e:
                logger.debug("NL->SQL fallback failed: %s", e)

        st.session_state.last_sql = sql_candidate or ""
        assistant_display = flat_text.replace("\n", "<br>")
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_display})

        # rerender UI to show new messages
        st.rerun()

# RIGHT: sticky controls panel
with right_col:
    # wrap controls in a sticky container so it stays visible
    st.markdown("<div id='controls-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.header("⚙️ Controls")
    page_size = st.number_input("Rows per page", min_value=5, max_value=1000, value=50, step=5)
    search_term = st.text_input("Search term (applies ILIKE across visible columns)")
    st.markdown("---")

    st.subheader("🧠 SQL Candidate")
    if st.session_state.last_sql:
        st.code(st.session_state.last_sql, language="sql")
    else:
        st.info("No SQL proposed yet. Ask the assistant or describe a query.")

    st.markdown("---")
    exec_col1, exec_col2 = st.columns([1, 1])
    fetch_page = exec_col1.button("Fetch / Refresh Page")
    run_full = exec_col2.button("Run full (dangerous)")

    st.markdown("---")
    st.subheader("📊 Last Result (preview)")
    if st.session_state.last_result is None:
        st.write("No results yet. Execute a query to preview results here.")
    else:
        res = st.session_state.last_result
        if isinstance(res, list):
            st.dataframe(res[: page_size if page_size else 50])
        else:
            st.text(res)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Execution helpers & handlers (outside widget creation)
# -------------------------
def build_where_clause_for_search(term: str, columns: List[str]) -> str:
    if not term or not columns:
        return ""
    q = term.replace("%", "\\%").replace("_", "\\_")
    quoted = "%{}%".format(q.replace("'", "''"))
    parts = []
    for c in columns:
        if re.match(r"^[\w\.]+$", c):
            parts.append(f"{c} ILIKE '{quoted}'")
    return " OR ".join(parts)

# Handle fetch/run buttons
if st.session_state.get("last_sql"):
    if fetch_page:
        user_sql = st.session_state.last_sql
        if re.match(r"^\s*select\b", user_sql.strip(), re.IGNORECASE):
            where_clause = ""
            cols = st.session_state.get("visible_columns") or []
            if search_term:
                wc = build_where_clause_for_search(search_term, cols)
                where_clause = wc

            count_q = f"WITH user_query AS ({user_sql.rstrip(';')}) SELECT count(*) as cnt FROM user_query;"
            with st.spinner("Counting rows..."):
                raw_cnt = run_executor_raw(count_q)
            if raw_cnt.get("stderr"):
                st.warning(f"Count query stderr: {raw_cnt.get('stderr')}")

            try:
                parsed_cnt = json.loads(raw_cnt.get("stdout") or "null")
                if isinstance(parsed_cnt, list) and parsed_cnt:
                    k = list(parsed_cnt[0].keys())[0]
                    st.session_state.total_rows = int(parsed_cnt[0].get(k, 0))
            except Exception:
                st.session_state.total_rows = None

            wc_sql = f"WHERE {where_clause}" if where_clause else ""
            page_q = f"WITH user_query AS ({user_sql.rstrip(';')}) SELECT * FROM user_query {wc_sql} LIMIT {page_size} OFFSET {st.session_state.get('page_idx', 0) * page_size};"
            with st.spinner("Fetching page..."):
                raw_page = run_executor_raw(page_q)
            if raw_page.get("stderr"):
                st.warning(f"Page query stderr: {raw_page.get('stderr')}")
            try:
                parsed_page = json.loads(raw_page.get("stdout") or "null")
            except Exception as e:
                logger.exception("Failed to parse page JSON: %s", e)
                parsed_page = raw_page.get("stdout") or []

            st.session_state.last_result = parsed_page
            if isinstance(parsed_page, list) and parsed_page:
                st.session_state.visible_columns = list(parsed_page[0].keys())

            if st.session_state.total_rows is not None:
                max_page = max(0, (st.session_state.total_rows - 1) // page_size)
                st.info(f"Page {st.session_state.get('page_idx', 0) + 1} / {max_page + 1} — total rows: {st.session_state.total_rows}")
        else:
            # non-select: run directly
            with st.spinner("Executing query..."):
                raw = run_executor_raw(user_sql)
            if raw.get("stderr"):
                st.warning(f"Executor stderr: {raw.get('stderr')}")
            try:
                parsed = json.loads(raw.get("stdout") or "null")
                st.session_state.last_result = parsed
            except Exception:
                st.session_state.last_result = raw.get("stdout")

    if run_full:
        with st.spinner("Running full SQL (be careful)..."):
            raw = run_executor_raw(st.session_state.last_sql, timeout=300)
        if raw.get("stderr"):
            st.warning(f"Executor stderr: {raw.get('stderr')}")
        try:
            parsed_full = json.loads(raw.get("stdout") or "null")
        except Exception:
            parsed_full = raw.get("stdout")
        st.session_state.last_result = parsed_full
        if isinstance(parsed_full, list) and parsed_full:
            st.session_state.visible_columns = list(parsed_full[0].keys())
            st.session_state.total_rows = len(parsed_full)

# Explanation button
if st.button("Explain last result"):
    if st.session_state.get("last_result") is None:
        st.info("No result to explain.")
    else:
        res = st.session_state["last_result"]
        if isinstance(res, list):
            st.success(f"Result contains {len(res)} rows. Showing first few rows above.")
        else:
            st.info("Last result is a non-tabular response; inspect output above.")

# Footer
st.caption("Made by Vishnu Pandey")
