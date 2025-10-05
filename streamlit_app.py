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
import markdown
from bs4 import BeautifulSoup

# local helper that maps short NL -> SQL (fallback)
import postgres_get_query as pgget

# MCP & agent imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

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

# MCP client wrapper
class MCPClientWrapper:
    def __init__(self, services: dict):
        self.services = services
        self._client: Optional[MultiServerMCPClient] = None

    async def _init_client(self):
        self._client = MultiServerMCPClient(self.services)
        return self._client

    async def get_tools(self):
        if not self._client:
            await self._init_client()
        return await self._client.get_tools()

# Agent & Model creation
async def _create_agent_and_model():
    services = {
        "postgres_get_query": {
            "command": sys.executable,
            "args": ["postgres_get_query.py"],
            "transport": "stdio",
        }
    }
    wrapper = MCPClientWrapper(services)

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found — set it in environment or .env")

    tools = await wrapper.get_tools()

    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key, temperature=0)
    agent = create_react_agent(model, tools)
    return agent, model

# synchronous helper

def get_persistent_agent():
    if "mcp_agent" in st.session_state and "mcp_model" in st.session_state:
        return st.session_state["mcp_agent"], st.session_state["mcp_model"]
    try:
        agent, model = asyncio.run(_create_agent_and_model())
    except Exception as e:
        logger.exception("Failed to initialize agent/model")
        st.error("Failed to initialize language agent — check logs and your OPENAI_API_KEY. Error: {}".format(e))
        raise
    st.session_state["mcp_agent"] = agent
    st.session_state["mcp_model"] = model
    return agent, model


def ask_agent_sync(prompt: str):
    agent, model = get_persistent_agent()
    try:
        response = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}))
    except Exception as e:
        logger.exception("Agent call failed")
        raise RuntimeError(f"Agent call failed: {e}")
    return response, model

# Explanation helper
async def explain_result_async(model, result_json) -> str:
    instructions = (
        "Explain the following PostgreSQL query result in short natural language. "
        "If the table/result is empty, say so and provide the schema where possible. Keep the explanation concise and factual."
    )
    explanation = await model.ainvoke(f"{instructions}\n\n{json.dumps(result_json, indent=2)}")
    if hasattr(explanation, "content"):
        return explanation.content
    return str(explanation)


def explain_result_sync(model, result_json) -> str:
    return asyncio.run(explain_result_async(model, result_json))

# Pagination / SQL helpers

def sanitize_sql_for_cte(sql: str) -> str:
    return sql.strip().rstrip(";")


def build_count_query(user_sql: str) -> str:
    cleaned = sanitize_sql_for_cte(user_sql)
    return f"WITH user_query AS ({cleaned}) SELECT count(*) as cnt FROM user_query;"


def build_page_query(user_sql: str, page_size: int, page_idx: int, where_clause: Optional[str] = None) -> str:
    cleaned = sanitize_sql_for_cte(user_sql)
    where = f"WHERE {where_clause}" if where_clause else ""
    return f"WITH user_query AS ({cleaned}) SELECT * FROM user_query {where} LIMIT {page_size} OFFSET {page_idx * page_size};"


def build_where_clause_for_search(term: str, columns: List[str]) -> str:
    if not term or not columns:
        return ""
    q = term.replace("%", "\\%").replace("_", "\\_")
    quoted = "%{}%".format(q.replace("'", "''"))
    parts = []
    for c in columns:
        if not re.match(r"^[\w\.]+$", c):
            continue
        parts.append(f"{c} ILIKE '{quoted}'")
    return " OR ".join(parts)

# -------------------------
# Streamlit UI (Aesthetic & Responsive)
# -------------------------

st.set_page_config(page_title="Data Vending Machine — Chat UI", layout="wide")

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

with left_col:
    st.markdown("<div class='card app-bg'>", unsafe_allow_html=True)

    # Chat area (uses st.chat_message if available)
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []  # list of {role: 'user'/'assistant'/'system', content}

    # Render chat messages in a scrollable container
    for msg in st.session_state.chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            st.markdown(f"<div style='display:flex;justify-content:flex-end;margin:6px 0'><div class='user-bubble'>{content}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='display:flex;justify-content:flex-start;margin:6px 0'><div class='bot-bubble'>{content}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Input area
    with st.form("chat_form", clear_on_submit=False):
        prompt = st.text_area("Say something (e.g. 'show me tables' or ask a question)", height=110, key="chat_input")
        col_a, col_b, col_c = st.columns([1,1,1])
        with col_a:
            send_btn = st.form_submit_button("Send")
        with col_b:
            regenerate = st.form_submit_button("Regenerate last")
        with col_c:
            clear_chat = st.form_submit_button("Clear chat")

    if clear_chat:
        st.session_state.chat_history = []
        st.session_state.last_sql = ""
        st.session_state.last_result = None
        st.experimental_rerun()

    if regenerate and st.session_state.chat_history:
        # resend the last user prompt
        last_user = None
        for m in reversed(st.session_state.chat_history):
            if m.get("role") == "user":
                last_user = m.get("content")
                break
        if last_user:
            prompt = last_user

    if send_btn and prompt.strip():
        # append user message to chat
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.chat_input})
        st.session_state.last_prompt = st.session_state.chat_input

        # call agent
        with st.spinner("Thinking..."):
            try:
                response, model = ask_agent_sync(st.session_state.last_prompt)
            except Exception as e:
                st.error(f"Agent call failed: {e}")
                logger.error("Agent call failure: %s", e)
                st.stop()

        flat = flatten_to_text(response)
        st.session_state.agent_reply = flat

        sql_agent = extract_sql_from_text(flat)
        sql = sql_agent or ""
        used_fallback = False
        if not sql:
            try:
                fb = pgget.get_query(st.session_state.last_prompt)
                if fb:
                    sql = fb
                    used_fallback = True
            except Exception as e:
                logger.warning("Fallback SQL generator failed: %s", e)

        st.session_state.last_sql = sql
        st.session_state.page_idx = 0
        st.session_state.total_rows = None
        st.session_state.last_result = None
        st.session_state.visible_columns = None

        # push assistant reply into chat history (rendered as bubble)
        assistant_display = flat.replace('\n', '<br>')
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_display})
        st.session_state.chat_input = ""
        st.experimental_rerun()

with right_col:
    st.markdown("<div class='card' style='padding:12px'>", unsafe_allow_html=True)
    st.subheader("Controls & Results")

    # Sidebar-like controls
    page_size = st.number_input("Rows per page", min_value=10, max_value=1000, value=100, step=10)
    paginated_fetch = st.checkbox("Use paginated fetch (recommended)", value=True)
    auto_exec_fallback = st.checkbox("Auto-execute fallback SQL when agent emits none", value=False)
    if st.button("Reset agent"):
        st.session_state.pop("mcp_agent", None)
        st.session_state.pop("mcp_model", None)
        st.success("Agent reset — will be recreated on next request.")

    st.markdown("---")
    st.subheader("SQL Candidate")
    if "last_sql" in st.session_state and st.session_state.last_sql:
        st.code(st.session_state.last_sql, language="sql")
    else:
        st.info("No SQL candidate yet — ask the agent above.")

    st.markdown("---")
    st.subheader("Execute & Preview")
    exec_col1, exec_col2 = st.columns([1,1])
    with exec_col1:
        do_execute = st.button("Fetch / Refresh Page")
    with exec_col2:
        run_full = st.button("Run full (dangerous)")

    if run_full and st.session_state.get("last_sql"):
        with st.spinner("Running full SQL..."):
            raw = run_executor_raw(st.session_state.last_sql)
        if raw.get("stderr"):
            st.warning(f"Executor stderr: {raw.get('stderr')}")
        try:
            parsed_full = json.loads(raw.get("stdout", "null") or "null")
        except Exception:
            parsed_full = raw.get("stdout")
        st.session_state.last_result = parsed_full
        st.session_state.total_rows = len(parsed_full) if isinstance(parsed_full, list) else None
        if isinstance(parsed_full, list) and parsed_full:
            st.session_state.visible_columns = list(parsed_full[0].keys())

    st.markdown("---")
    st.text_input("Search term (applies ILIKE across chosen columns)", key="search_term")

    # Show last result preview
    st.subheader("Last Result (preview)")
    if st.session_state.get("last_result") is None:
        st.write("No result yet. Fetch a page to preview results.")
    else:
        res = st.session_state.last_result
        if isinstance(res, list) and res and all(isinstance(r, dict) for r in res):
            # show a compact preview (first 10 rows)
            preview = res[:10]
            st.dataframe(preview)
        else:
            st.write(res)

    st.markdown("</div>", unsafe_allow_html=True)

# Execute pagination when asked
if st.session_state.get("last_sql") and (st.button("Fetch page") or do_execute):
    user_sql = st.session_state.last_sql
    if not user_sql:
        st.error("No SQL available to execute.")
    else:
        if paginated_fetch and re.match(r"^\s*select\b", user_sql.strip(), re.IGNORECASE):
            where_clause = ""
            if st.session_state.get("search_term"):
                # default: search all visible columns
                cols = st.session_state.get("visible_columns") or []
                if not cols and isinstance(st.session_state.get("last_result"), list) and st.session_state["last_result"]:
                    cols = list(st.session_state["last_result"][0].keys())
                wc = build_where_clause_for_search(st.session_state.get("search_term"), cols)
                where_clause = wc

            count_q = build_count_query(user_sql)
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
            st.session_state.total_rows = total

            page_q = build_page_query(user_sql, page_size, st.session_state.get("page_idx", 0), where_clause=where_clause)
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

            st.session_state.last_result = parsed_page if parsed_page is not None else []
            if isinstance(parsed_page, list) and parsed_page:
                if not st.session_state.get("visible_columns"):
                    st.session_state.visible_columns = list(parsed_page[0].keys())

            if st.session_state.total_rows is not None:
                max_page = max(0, (st.session_state.total_rows - 1) // page_size)
                st.info(f"Page {st.session_state.get('page_idx', 0) + 1} / {max_page + 1} — total rows: {st.session_state.total_rows}")
        else:
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
            st.session_state.last_result = parsed
            if isinstance(parsed, list) and parsed:
                st.session_state.total_rows = len(parsed)
                if not st.session_state.get("visible_columns"):
                    st.session_state.visible_columns = list(parsed[0].keys())

# Explanation tab at bottom (simple)
st.markdown("---")
if st.button("Explain last result"):
    if st.session_state.get("last_result") is None:
        st.info("No result to explain.")
    else:
        try:
            _, model = get_persistent_agent()
        except Exception as e:
            st.error(f"Could not get model to explain: {e}")
            model = None
        if model:
            with st.spinner("Generating explanation..."):
                try:
                    explanation = explain_result_sync(model, st.session_state.last_result)
                    st.markdown(f"**Explanation:**<br>{explanation}", unsafe_allow_html=True)
                except Exception as e:
                    logger.exception("Explanation failed")
                    st.error(f"Explanation generation failed: {e}")
        else:
            st.info("No model available to generate explanation.")

# Footer
st.caption("Made by Vishnu Pandey")
