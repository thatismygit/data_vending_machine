# streamlit_app.py  -- rewritten to avoid modifying widget-backed session_state after widget creation
import os
import sys
import re
import json
import subprocess
import asyncio
import logging
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# local helper that maps short NL -> SQL (fallback)
import postgres_get_query as pgget

# lighten imports for agent part (keep your original imports if needed)
# from langchain_mcp_adapters.client import MultiServerMCPClient
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI

# load .env early
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
# Utility functions (kept compact)
# -------------------------
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

# SQL extraction utilities
SQL_KEYWORDS_RE = re.compile(r"\b(select|insert|update|delete|create|alter|drop|with)\b", re.IGNORECASE)

def is_probable_sql(candidate: str) -> bool:
    if not candidate or len(candidate.strip()) < 8:
        return False
    if not SQL_KEYWORDS_RE.search(candidate):
        return False
    if re.match(r"^\s*with\b", candidate, re.IGNORECASE) and not re.search(r"\b(select|insert|update|delete)\b", candidate, re.IGNORECASE):
        return False
    return True

def extract_sql_from_text(text: str) -> str:
    if not text:
        return ""
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        cand = fence.group(1).strip()
        return cand if is_probable_sql(cand) else ""
    inline = re.findall(r"`([^`]{10,})`", text)
    for m in inline:
        if is_probable_sql(m):
            return m.strip()
    start = re.search(r"\b(select|create|insert|update|delete|with)\b", text, re.IGNORECASE)
    if start:
        i = start.start()
        sem = text.find(";", i)
        if sem != -1:
            cand = text[i:sem+1].strip()
            return cand if is_probable_sql(cand) else ""
        return text[i:i+2000].strip() if is_probable_sql(text[i:i+2000]) else ""
    return ""

# Executor helper
def run_executor_raw(sql: str, timeout: int = 120) -> Dict[str, Any]:
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
# Streamlit UI (Aesthetic & Responsive)
# -------------------------
st.set_page_config(page_title="Data Vending Machine", layout="wide")

# Minimal CSS to make the app look like a modern chat app and responsive
st.markdown(
    """
    <style>
    /* page background and card */
    .app-bg { background: linear-gradient(180deg, #0f172a 0%, #071029 100%); padding: 18px; border-radius: 12px;}
    .card { background: rgba(255,255,255,0.03); padding: 18px; border-radius: 12px; box-shadow: 0 6px 18px rgba(2,6,23,0.6); color: #e6eef8}
    .muted { color: #9fb0d6 }

    /* chat bubbles */
    .user-bubble { background: linear-gradient(90deg,#4f46e5,#06b6d4); color: white; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:80%; }
    .bot-bubble { background: rgba(255,255,255,0.06); color: #e6eef8; padding: 10px 14px; border-radius: 18px; display:inline-block; max-width:80%; }

    /* responsive columns */
    @media (max-width: 900px) {
        .stColumns { flex-direction: column; }
        .sidebar .stButton { width: 100%; }
    }

    /* make code blocks more compact */
    .code-style { background: rgba(0,0,0,0.35); padding:10px; border-radius:8px }

    /* sticky bottom input on wide screens */
    .input-row { position: sticky; bottom: 8px; background: transparent; padding-top:8px }

    </style>
    """,
    unsafe_allow_html=True,
)

# Top bar
st.markdown("<div style='display:flex;align-items:center;gap:12px'><h2 style='margin:0;color:#e6eef8'>Data Vending Machine</h2><span class='muted'>— conversational SQL explorer</span></div>", unsafe_allow_html=True)

# Layout: left = chat, right = controls & results
left_col, right_col = st.columns([2.2, 1])

# initialize session state pieces we use (safe)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of {role, content}
if "last_sql" not in st.session_state:
    st.session_state["last_sql"] = ""
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "page_idx" not in st.session_state:
    st.session_state["page_idx"] = 0
if "total_rows" not in st.session_state:
    st.session_state["total_rows"] = None
if "visible_columns" not in st.session_state:
    st.session_state["visible_columns"] = None

with left_col:
    st.markdown("<div class='card app-bg'>", unsafe_allow_html=True)

    # Render chat messages
    for msg in st.session_state.chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            st.markdown(f"<div style='display:flex;justify-content:flex-end;margin:6px 0'><div class='user-bubble'>{content}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='display:flex;justify-content:flex-start;margin:6px 0'><div class='bot-bubble'>{content}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input area: use form with clear_on_submit so we do not need to mutate widget-backed session_state
    with st.form("chat_form", clear_on_submit=True):
        input_text = st.text_area("Say something (e.g. 'show me tables' or ask a question)", height=110, key="chat_input")
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:
            send_btn = st.form_submit_button("Send")
        with col_b:
            regenerate_btn = st.form_submit_button("Regenerate last")
        with col_c:
            clear_chat_btn = st.form_submit_button("Clear chat")

    # Process Clear Chat (safe: does not try to change the widget)
    if clear_chat_btn:
        st.session_state["chat_history"] = []
        st.session_state["last_sql"] = ""
        st.session_state["last_result"] = None
        st.session_state["page_idx"] = 0
        st.session_state["total_rows"] = None
        st.session_state["visible_columns"] = None
        st.experimental_rerun()

    # Determine what to send to agent:
    # - If send_btn: use input_text returned by the widget
    # - If regenerate_btn: find last user message and resend it (don't attempt to set the widget)
    send_prompt = None
    if send_btn and (input_text and input_text.strip()):
        send_prompt = input_text.strip()
    elif regenerate_btn:
        # find last user message
        last_user = None
        for m in reversed(st.session_state.chat_history):
            if m.get("role") == "user":
                last_user = m.get("content")
                break
        if last_user:
            send_prompt = last_user

    # If we have a prompt to send, call the agent (synchronously here using your helper)
    if send_prompt:
        # add user message to chat_history
        st.session_state.chat_history.append({"role": "user", "content": send_prompt})
        st.session_state["last_prompt"] = send_prompt

        # call agent - keep your original agent call; below we use a simplified placeholder call for clarity.
        # Replace `mock_agent_call` with your actual ask_agent_sync/agent invocation function.
        def mock_agent_call(prompt_text: str):
            # placeholder: echo with an imaginary SQL suggestion
            return {"content": f"Agent response for: {prompt_text}\n\nSuggested SQL:\n```sql\nSELECT * FROM my_table LIMIT 10;\n```"}

        with st.spinner("Thinking..."):
            try:
                response = mock_agent_call(send_prompt)
            except Exception as e:
                st.error(f"Agent call failed: {e}")
                logger.exception("Agent call failure: %s", e)
                st.stop()

        flat = flatten_to_text(response)
        st.session_state["agent_reply"] = flat

        # Try to extract SQL from the agent response
        sql_agent = extract_sql_from_text(flat)
        sql_candidate = sql_agent or ""
        used_fallback = False

        if not sql_candidate:
            # fallback: try your NL->SQL helper (this does not touch widget-backed keys)
            try:
                fb = pgget.get_query(send_prompt)
                if fb:
                    sql_candidate = fb
                    used_fallback = True
            except Exception as e:
                logger.warning("Fallback SQL generator failed: %s", e)

        st.session_state["last_sql"] = sql_candidate
        st.session_state["page_idx"] = 0
        st.session_state["total_rows"] = None
        st.session_state["last_result"] = None
        st.session_state["visible_columns"] = None

        # show assistant reply in chat (render newline as <br>)
        assistant_display = flat.replace("\n", "<br>")
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_display})

        # re-run to render updated chat and controls
        st.experimental_rerun()

with right_col:
    st.markdown("<div class='card' style='padding:12px'>", unsafe_allow_html=True)
    st.subheader("Controls & Results")

    page_size = st.number_input("Rows per page", min_value=10, max_value=1000, value=100, step=10)
    paginated_fetch = st.checkbox("Use paginated fetch (recommended)", value=True)
    auto_exec_fallback = st.checkbox("Auto-execute fallback SQL when agent emits none", value=False)
    if st.button("Reset agent"):
        # If you have agent objects in session_state, remove them here
        st.session_state.pop("mcp_agent", None)
        st.session_state.pop("mcp_model", None)
        st.success("Agent reset — will be recreated on next request.")

    st.markdown("---")
    st.subheader("SQL Candidate")
    if st.session_state.get("last_sql"):
        st.code(st.session_state["last_sql"], language="sql")
    else:
        st.info("No SQL candidate yet — ask the agent above.")

    st.markdown("---")
    st.subheader("Execute & Preview")
    exec_col1, exec_col2 = st.columns([1,1])
    do_execute = exec_col1.button("Fetch / Refresh Page")
    run_full = exec_col2.button("Run full (dangerous)")

    # simple search term box (not widget-key mutated after creation)
    search_term = st.text_input("Search term (applies ILIKE across chosen columns)")

    st.markdown("---")
    st.subheader("Last Result (preview)")
    if st.session_state.get("last_result") is None:
        st.write("No result yet. Fetch a page to preview results.")
    else:
        res = st.session_state["last_result"]
        if isinstance(res, list) and res and all(isinstance(r, dict) for r in res):
            preview = res[:10]
            st.dataframe(preview)
        else:
            st.write(res)

    st.markdown("</div>", unsafe_allow_html=True)

# Execute pagination / full query handling outside of right_col markup to maintain flow
def fetch_page_for_sql(user_sql: str, page_idx: int, page_size: int, search_term: str) -> None:
    # This re-implements the pagination logic safely
    if not user_sql:
        st.error("No SQL available to execute.")
        return

    if paginated_fetch and re.match(r"^\s*select\b", user_sql.strip(), re.IGNORECASE):
        # compute where clause if any
        where_clause = ""
        if search_term:
            cols = st.session_state.get("visible_columns") or []
            if not cols and isinstance(st.session_state.get("last_result"), list) and st.session_state["last_result"]:
                cols = list(st.session_state["last_result"][0].keys())
            # build basic where clause (simple)
            q = search_term.replace("%", "\\%").replace("_", "\\_")
            quoted = "%{}%".format(q.replace("'", "''"))
            parts = []
            for c in cols:
                if re.match(r"^[\w\.]+$", c):
                    parts.append(f"{c} ILIKE '{quoted}'")
            where_clause = " OR ".join(parts)

        # count
        count_q = f"WITH user_query AS ({user_sql.rstrip(';')}) SELECT count(*) as cnt FROM user_query;"
        with st.spinner("Getting total count..."):
            raw_cnt = run_executor_raw(count_q)
        if raw_cnt.get("stderr"):
            st.warning(f"Count query stderr: {raw_cnt.get('stderr')}")

        total = None
        try:
            if raw_cnt.get("stdout"):
                parsed_cnt = json.loads(raw_cnt["stdout"]) if raw_cnt["stdout"] else None
                if isinstance(parsed_cnt, list) and parsed_cnt and isinstance(parsed_cnt[0], dict):
                    k = list(parsed_cnt[0].keys())[0]
                    total = int(parsed_cnt[0].get(k, 0))
        except Exception as e:
            logger.warning("Failed to parse count result: %s", e)
            st.warning(f"Could not parse count result: {e}")
        st.session_state["total_rows"] = total

        # fetch page
        wc = f"WHERE {where_clause}" if where_clause else ""
        page_q = f"WITH user_query AS ({user_sql.rstrip(';')}) SELECT * FROM user_query {wc} LIMIT {page_size} OFFSET {page_idx * page_size};"
        with st.spinner("Fetching page..."):
            raw_page = run_executor_raw(page_q)
        if raw_page.get("stderr"):
            st.warning(f"Page query stderr: {raw_page.get('stderr')}")
        try:
            parsed_page = json.loads(raw_page.get("stdout", "null") or "null")
        except Exception as e:
            logger.exception("Failed to parse page JSON")
            st.error(f"Could not parse page JSON: {e}")
            parsed_page = None

        st.session_state["last_result"] = parsed_page if parsed_page is not None else []
        if isinstance(parsed_page, list) and parsed_page:
            if not st.session_state.get("visible_columns"):
                st.session_state["visible_columns"] = list(parsed_page[0].keys())

        if st.session_state.get("total_rows") is not None:
            max_page = max(0, (st.session_state["total_rows"] - 1) // page_size)
            st.info(f"Page {st.session_state.get('page_idx', 0) + 1} / {max_page + 1} — total rows: {st.session_state['total_rows']}")
    else:
        # non-paginated
        with st.spinner("Executing (non-paginated) query..."):
            raw = run_executor_raw(user_sql)
        if raw.get("stderr"):
            st.warning(f"Executor stderr: {raw.get('stderr')}")
        try:
            parsed = json.loads(raw.get("stdout", "null") or "null")
        except Exception as e:
            logger.exception("Failed to parse full result JSON")
            st.error(f"Could not parse full result JSON: {e}")
            parsed = None
        st.session_state["last_result"] = parsed
        if isinstance(parsed, list) and parsed:
            st.session_state["total_rows"] = len(parsed)
            if not st.session_state.get("visible_columns"):
                st.session_state["visible_columns"] = list(parsed[0].keys())

# Buttons in right column that trigger execution:
if st.session_state.get("last_sql") and (do_execute or st.button("Fetch page")):
    fetch_page_for_sql(st.session_state["last_sql"], st.session_state.get("page_idx", 0), page_size, search_term)

if run_full and st.session_state.get("last_sql"):
    with st.spinner("Running full SQL..."):
        raw = run_executor_raw(st.session_state["last_sql"])
    if raw.get("stderr"):
        st.warning(f"Executor stderr: {raw.get('stderr')}")
    try:
        parsed_full = json.loads(raw.get("stdout", "null") or "null")
    except Exception:
        parsed_full = raw.get("stdout")
    st.session_state["last_result"] = parsed_full
    st.session_state["total_rows"] = len(parsed_full) if isinstance(parsed_full, list) else None
    if isinstance(parsed_full, list) and parsed_full:
        st.session_state["visible_columns"] = list(parsed_full[0].keys())

# Explanation button (keeps model/agent parts as previously implemented)
st.markdown("---")
if st.button("Explain last result"):
    if st.session_state.get("last_result") is None:
        st.info("No result to explain.")
    else:
        # Place your model/agent explain call here
        st.info("Explanation flow would execute here (add your model explain call).")

st.caption("Made by Vishnu Pandey")
