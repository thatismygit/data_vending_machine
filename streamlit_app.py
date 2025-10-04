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
# Logging & streamlit detection
# -------------------------
logger = logging.getLogger("data_vending")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

IS_STREAMLIT = "streamlit" in sys.modules or os.environ.get("STREAMLIT_RUN", "") != ""
if IS_STREAMLIT:
    logger.info("Running inside Streamlit runtime. Enabling Streamlit-friendly logging.")
else:
    logger.info("Not running inside Streamlit")

# -------------------------
# Utility functions
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

# SQL extraction utilities (kept intentionally conservative)
SQL_KEYWORDS_RE = re.compile(r"\b(select|insert|update|delete|create|alter|drop|with)\b", re.IGNORECASE)


def is_probable_sql(candidate: str) -> bool:
    if not candidate or len(candidate.strip()) < 8:
        return False
    if not SQL_KEYWORDS_RE.search(candidate):
        return False
    # basic heuristics
    if re.match(r"^\s*with\b", candidate, re.IGNORECASE) and not re.search(r"\b(select|insert|update|delete)\b", candidate, re.IGNORECASE):
        return False
    return True


def extract_sql_from_text(text: str) -> str:
    if not text:
        return ""
    # code fences
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        cand = fence.group(1).strip()
        return cand if is_probable_sql(cand) else ""
    # inline ticks
    inline = re.findall(r"`([^`]{10,})`", text)
    for m in inline:
        if is_probable_sql(m):
            return m.strip()
    # find keyword start
    start = re.search(r"\b(select|create|insert|update|delete|with)\b", text, re.IGNORECASE)
    if start:
        i = start.start()
        sem = text.find(";", i)
        if sem != -1:
            cand = text[i:sem+1].strip()
            return cand if is_probable_sql(cand) else ""
        return text[i:i+2000].strip() if is_probable_sql(text[i:i+2000]) else ""
    return ""

# -------------------------
# Executor helper
# -------------------------

def run_executor_raw(sql: str, timeout: int = 120) -> Dict[str, Any]:
    """
    Executes postgres_execute_query.py in CLI mode (reads SQL from stdin, prints JSON to stdout).
    Returns dict with keys: rc, stdout, stderr
    """
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
# MCP client wrapper with Streamlit detection
# -------------------------
class MCPClientWrapper:
    """Create MCP client and tools with extra logging when running inside Streamlit.

    This wrapper centralizes creation so we can control retries, timeouts, and provide
    clearer error messages in the UI.
    """

    def __init__(self, services: dict):
        self.services = services
        self._client: Optional[MultiServerMCPClient] = None

    async def _init_client(self):
        # If running in streamlit, reduce noisy prints from MCP by setting env var or similar.
        if IS_STREAMLIT:
            logger.info("Initializing MultiServerMCPClient in Streamlit mode")
        self._client = MultiServerMCPClient(self.services)
        return self._client

    async def get_tools(self):
        if not self._client:
            await self._init_client()
        return await self._client.get_tools()

# -------------------------
# Agent & Model creation (uses gpt-4o-mini)
# -------------------------

async def _create_agent_and_model():
    # configure MCP wrapper to launch the local postgres_get_query helper
    services = {
        "postgres_get_query": {
            "command": sys.executable,
            "args": ["postgres_get_query.py"],
            "transport": "stdio",
        }
    }
    wrapper = MCPClientWrapper(services)

    # read OpenAI key from env
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not found — set it in environment or .env")

    tools = await wrapper.get_tools()

    # Use standard LLM format per request: replace deepseek/ollama with 'gpt-4o-mini'
    model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_key, temperature=0)

    # create react agent (langgraph) that can call tools
    agent = create_react_agent(model, tools)
    return agent, model

# synchronous helper to initialize once and persist in session state

def get_persistent_agent():
    """Create agent and model once and store in st.session_state. Returns (agent, model).

    Any initialization errors are surfaced to the user via Streamlit error box.
    """
    if "mcp_agent" in st.session_state and "mcp_model" in st.session_state:
        return st.session_state["mcp_agent"], st.session_state["mcp_model"]

    # create and persist
    try:
        agent, model = asyncio.run(_create_agent_and_model())
    except Exception as e:
        logger.exception("Failed to initialize agent/model")
        # show user-friendly error and raise to stop current flow
        st.error("Failed to initialize language agent — check logs and your OPENAI_API_KEY. Error: {}".format(e))
        raise
    st.session_state["mcp_agent"] = agent
    st.session_state["mcp_model"] = model
    return agent, model


def ask_agent_sync(prompt: str):
    """Ask the persisted agent synchronously (wraps async call). Returns response object and model."""
    agent, model = get_persistent_agent()
    try:
        response = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}))
    except Exception as e:
        logger.exception("Agent call failed")
        # rethrow so caller UI can surface message
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

# -------------------------
# Pagination / SQL helpers (same as before, robust)
# -------------------------

def sanitize_sql_for_cte(sql: str) -> str:
    return sql.strip().rstrip(";")


def build_count_query(user_sql: str) -> str:
    cleaned = sanitize_sql_for_cte(user_sql)
    return f"WITH user_query AS ({cleaned}) SELECT count(*) as cnt FROM user_query;"


def build_page_query(user_sql: str, page_size: int, page_idx: int, where_clause: Optional[str] = None) -> str:
    cleaned = sanitize_sql_for_cte(user_sql)
    where = f"WHERE {where_clause}" if where_clause else ""
    return f"WITH user_query AS ({cleaned}) SELECT * FROM user_query {where} LIMIT {page_size} OFFSET {page_idx * page_size};"


def quote_like(s: str) -> str:
    if s is None:
        return "''"
    return "'" + s.replace("'", "''") + "'"


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
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Data Vending Machine (gpt-4o-mini)", layout="wide")
st.title("Data Vending Machine")
st.caption("Turn plain English into safe, paginated SQL results — with explanations")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    page_size = st.number_input("Rows per page", min_value=10, max_value=1000, value=100, step=10)
    paginated_fetch = st.checkbox("Use paginated fetch (recommended)", value=True)
    auto_exec_fallback = st.checkbox("Auto-execute fallback SQL when agent emits none", value=False)
    if st.button("Reset agent"):
        st.session_state.pop("mcp_agent", None)
        st.session_state.pop("mcp_model", None)
        st.success("Agent reset — will be recreated on next request.")

# initialize session state defaults
st.session_state.setdefault("last_prompt", "")
st.session_state.setdefault("agent_reply", "")
st.session_state.setdefault("last_sql", "")
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("total_rows", None)
st.session_state.setdefault("page_idx", 0)
st.session_state.setdefault("visible_columns", None)
st.session_state.setdefault("search_term", "")
st.session_state.setdefault("search_columns", None)

# 1) Ask the agent
st.markdown("## 1) Ask the agent")
with st.form("agent_form"):
    prompt = st.text_area("Enter natural-language request", value=st.session_state.last_prompt, height=120, placeholder="e.g. show me tables")
    send = st.form_submit_button("Send to agent")

if send and prompt.strip():
    st.session_state.last_prompt = prompt
    with st.spinner("Calling language agent..."):
        try:
            response, model = ask_agent_sync(prompt)
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
            fb = pgget.get_query(prompt)
            if fb:
                sql = fb
                used_fallback = True
                if auto_exec_fallback:
                    st.info("Auto-executing fallback SQL (from NL->SQL generator)")
        except Exception as e:
            logger.warning("Fallback SQL generator failed: %s", e)
            st.warning("Fallback generator failed: {}".format(e))

    st.session_state.last_sql = sql
    st.session_state.page_idx = 0
    st.session_state.total_rows = None
    st.session_state.last_result = None
    st.session_state.visible_columns = None
    if used_fallback:
        st.success("SQL candidate produced by fallback generator.")
    else:
        st.success("Agent processed — SQL candidate updated.")

# 2) Show agent reply and SQL
st.markdown("## 2) Agent reply & SQL")
if st.session_state.agent_reply:
    st.markdown(st.session_state.agent_reply, unsafe_allow_html=True)
else:
    st.info("No agent reply yet. Submit a prompt above.")

st.subheader("SQL candidate")
if st.session_state.last_sql:
    st.code(st.session_state.last_sql, language="sql")
else:
    st.info("No SQL candidate produced. Try rephrasing.")

# 3) Execution controls
st.markdown("## 3) Execute")
col1, col2 = st.columns([1, 1])
with col1:
    do_execute = st.button("Fetch / Refresh Page")
with col2:
    reset_cols = st.button("Reset column chooser")

if reset_cols:
    st.session_state.visible_columns = None

with st.expander("Advanced / non-paginated (not recommended)"):
    run_non_paged = st.button("Execute full SQL (fetch everything)")
    if run_non_paged:
        if not st.session_state.last_sql:
            st.error("No SQL to run.")
        else:
            with st.spinner("Executing full SQL..."):
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

# Pagination UI
if st.session_state.last_sql:
    st.markdown("### Pagination & Filters")
    st.session_state.search_term = st.text_input("Search term (applies ILIKE across chosen columns)", value=st.session_state.search_term)

    candidate_columns = []
    if isinstance(st.session_state.last_result, list) and st.session_state.last_result and isinstance(st.session_state.last_result[0], dict):
        candidate_columns = list(st.session_state.last_result[0].keys())

    if candidate_columns:
        search_cols = st.multiselect("Columns to search (empty = all shown columns)", options=candidate_columns, default=st.session_state.search_columns or [])
    else:
        search_cols_raw = st.text_input("Columns to search (comma-separated)", value=",".join(st.session_state.search_columns) if st.session_state.search_columns else "")
        search_cols = [c.strip() for c in search_cols_raw.split(",") if c.strip()] if search_cols_raw else []
    st.session_state.search_columns = search_cols or None

    if candidate_columns:
        chosen_cols = st.multiselect("Columns to display", options=candidate_columns, default=st.session_state.visible_columns or candidate_columns)
        st.session_state.visible_columns = chosen_cols
    else:
        manual_cols = st.text_input("Columns to display (comma-separated)", value=",".join(st.session_state.visible_columns) if st.session_state.visible_columns else "")
        if manual_cols:
            st.session_state.visible_columns = [c.strip() for c in manual_cols.split(",") if c.strip()]

    nav_col1, nav_col2 = st.columns([1, 3])
    with nav_col1:
        prev_page = st.button("◀ Prev page")
        next_page = st.button("Next page ▶")
    with nav_col2:
        goto = st.number_input("Go to page (1-based)", min_value=1, value=st.session_state.page_idx + 1, step=1)

    if prev_page and st.session_state.page_idx > 0:
        st.session_state.page_idx -= 1
    if next_page:
        st.session_state.page_idx += 1
    if goto and (goto - 1) != st.session_state.page_idx:
        st.session_state.page_idx = max(0, int(goto) - 1)

    if do_execute or prev_page or next_page or goto:
        user_sql = st.session_state.last_sql
        if not user_sql:
            st.error("No SQL available to execute.")
        else:
            # only allow paginated fetch for SELECT queries
            if paginated_fetch and re.match(r"^\s*select\b", user_sql.strip(), re.IGNORECASE):
                where_clause = ""
                if st.session_state.search_term and st.session_state.search_columns:
                    wc = build_where_clause_for_search(st.session_state.search_term, st.session_state.search_columns)
                    where_clause = wc

                count_q = build_count_query(user_sql)
                with st.spinner("Getting total count (may be slow)..."):
                    raw_cnt = run_executor_raw(count_q)
                if raw_cnt.get("stderr"):
                    st.warning(f"Count query stderr: {raw_cnt.get('stderr')}")

                total = None
                try:
                    if raw_cnt.get("stdout"):
                        parsed_cnt = json.loads(raw_cnt["stdout"])
                        if isinstance(parsed_cnt, list) and parsed_cnt and isinstance(parsed_cnt[0], dict):
                            k = list(parsed_cnt[0].keys())[0]
                            total = int(parsed_cnt[0].get(k, 0))
                except Exception as e:
                    logger.warning("Failed to parse count result: %s", e)
                    st.warning(f"Could not parse count result: {e}")
                st.session_state.total_rows = total

                page_q = build_page_query(user_sql, page_size, st.session_state.page_idx, where_clause=where_clause)
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
                    if not st.session_state.visible_columns:
                        st.session_state.visible_columns = list(parsed_page[0].keys())

                if st.session_state.total_rows is not None:
                    max_page = max(0, (st.session_state.total_rows - 1) // page_size)
                    st.info(f"Page {st.session_state.page_idx + 1} / {max_page + 1} — total rows: {st.session_state.total_rows}")
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
                    if not st.session_state.visible_columns:
                        st.session_state.visible_columns = list(parsed[0].keys())

# 4) Results & Explanation
st.markdown("## 4) Results")
tab_raw, tab_table, tab_explain = st.tabs(["Raw JSON", "Pretty Table", "Explanation"]) 

with tab_raw:
    if st.session_state.last_result is None:
        st.info("No result yet.")
    else:
        if isinstance(st.session_state.last_result, (list, dict)):
            st.code(json.dumps(st.session_state.last_result, indent=2)[:200000])
        else:
            st.code(str(st.session_state.last_result)[:200000])

with tab_table:
    res = st.session_state.last_result
    if isinstance(res, list) and len(res) > 0 and all(isinstance(r, dict) for r in res):
        # show only selected columns
        if st.session_state.visible_columns:
            def _select_row(r):
                return {k: r.get(k) for k in st.session_state.visible_columns if k in r}
            display_rows = [_select_row(r) for r in res]
        else:
            display_rows = res
        st.dataframe(display_rows)
    elif res is None:
        st.info("No result to display.")
    else:
        st.write(res)

with tab_explain:
    if st.session_state.last_result is None:
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
                    st.write(explanation)
                except Exception as e:
                    logger.exception("Explanation failed")
                    st.error(f"Explanation generation failed: {e}")
        else:
            st.info("No model available to generate explanation.")

st.markdown("---")
st.caption("Made by Vishnu Pandey")
