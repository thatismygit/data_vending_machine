# streamlit_app.py
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

# local helpers
import postgres_get_query as pgget

# MCP & agent imports
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# Logging
logger = logging.getLogger("data_vending_chat")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# --------------------
# Utilities (lifted & simplified from your original app)
# --------------------
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
        cand = text[i:i+2000].strip()
        return cand if is_probable_sql(cand) else ""
    return ""

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

# Executor: run postgres_execute_query.py in CLI mode (stdin -> JSON stdout)
def run_executor_raw(sql: str, timeout: int = 120) -> Dict[str, Any]:
    cmd = [sys.executable, "postgres_execute_query.py", "--run-sql"]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=sql.encode("utf-8"), timeout=timeout)
        return {"rc": proc.returncode, "stdout": stdout.decode("utf-8", errors="replace"), "stderr": stderr.decode("utf-8", errors="replace")}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {"rc": -1, "stdout": "", "stderr": "timeout"}
    except FileNotFoundError:
        return {"rc": -1, "stdout": "", "stderr": "postgres_execute_query.py not found"}
    except Exception as e:
        return {"rc": -1, "stdout": "", "stderr": str(e)}

# --------------------
# MCP / Agent creation (persist in session_state)
# --------------------
class MCPClientWrapper:
    def __init__(self, services: dict):
        self.services = services
        self._client = None

    async def _init_client(self):
        self._client = MultiServerMCPClient(self.services)
        return self._client

    async def get_tools(self):
        if not self._client:
            await self._init_client()
        return await self._client.get_tools()

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

def get_persistent_agent():
    if "mcp_agent" in st.session_state and "mcp_model" in st.session_state:
        return st.session_state["mcp_agent"], st.session_state["mcp_model"]
    try:
        agent, model = asyncio.run(_create_agent_and_model())
    except Exception as e:
        st.error(f"Failed to initialize language agent — check OPENAI_API_KEY. Error: {e}")
        raise
    st.session_state["mcp_agent"] = agent
    st.session_state["mcp_model"] = model
    return agent, model

def ask_agent_sync(prompt: str):
    agent, model = get_persistent_agent()
    try:
        response = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": prompt}]}))
    except Exception as e:
        raise RuntimeError(f"Agent call failed: {e}")
    return response, model

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

# --------------------
# Streamlit Chat UI
# --------------------
st.set_page_config(page_title="Data Vending — Chat", layout="wide")
st.title("Data Vending Machine — Chat")
st.caption("Chat with the agent. When SQL is returned you can inspect and execute it safely.")

# Sidebar: settings
with st.sidebar:
    st.header("Settings")
    page_size = st.number_input("Rows per page", min_value=10, max_value=1000, value=100, step=10)
    paginated_fetch = st.checkbox("Use paginated fetch (recommended)", value=True)
    auto_exec_fallback = st.checkbox("Auto-execute fallback SQL when agent emits none", value=False)
    if st.button("Reset agent"):
        st.session_state.pop("mcp_agent", None)
        st.session_state.pop("mcp_model", None)
        st.success("Agent reset — will be recreated on next request.")

# Chat history stored in session_state as list of {"role": "user"|"assistant", "text": "...", "sql": optional, "last_result": optional}
st.session_state.setdefault("chat_messages", [])
st.session_state.setdefault("page_idx", 0)
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("total_rows", None)
st.session_state.setdefault("visible_columns", None)
st.session_state.setdefault("search_term", "")
st.session_state.setdefault("search_columns", None)

# Show chat messages
for i, msg in enumerate(st.session_state.chat_messages):
    # use streamlit chat primitives if available
    role = msg.get("role", "assistant")
    content = msg.get("text", "")
    if role == "user":
        st.chat_message("user").write(content)
    else:
        st.chat_message("assistant").markdown(content, unsafe_allow_html=True)
        # if assistant contained SQL, show it and an Execute button
        sql = msg.get("sql")
        if sql:
            st.code(sql, language="sql")
            # Buttons need unique keys
            exec_key = f"exec_sql_{i}"
            explain_key = f"explain_sql_{i}"
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Execute SQL", key=exec_key):
                    # run execution (paginated if select)
                    st.session_state.page_idx = 0
                    user_sql = sql
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
                        if isinstance(parsed_page, list) and parsed_page and not st.session_state.visible_columns:
                            st.session_state.visible_columns = list(parsed_page[0].keys())
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
                        if isinstance(parsed, list) and parsed and not st.session_state.visible_columns:
                            st.session_state.visible_columns = list(parsed[0].keys())
                    # attach result to this message for convenience
                    st.session_state.chat_messages[i]["last_result"] = st.session_state.last_result
            with col2:
                if st.button("Explain result", key=explain_key):
                    # generate explanation for last_result of this message (if none, will attempt on st.session_state.last_result)
                    to_explain = msg.get("last_result", st.session_state.last_result)
                    if to_explain is None:
                        st.info("No result to explain. Execute the SQL first.")
                    else:
                        try:
                            _, model = get_persistent_agent()
                        except Exception as e:
                            st.error(f"Could not get model to explain: {e}")
                            model = None
                        if model:
                            with st.spinner("Generating explanation..."):
                                try:
                                    explanation = explain_result_sync(model, to_explain)
                                    st.write(explanation)
                                except Exception as e:
                                    logger.exception("Explanation failed")
                                    st.error(f"Explanation generation failed: {e}")
                        else:
                            st.info("No model available to generate explanation.")

# Input area (chat form)
with st.form("chat_input", clear_on_submit=True):
    user_text = st.text_area("You", height=120, placeholder="Ask in plain English")
    submit = st.form_submit_button("Send")

if submit and user_text and user_text.strip():
    # append user message
    st.session_state.chat_messages.append({"role": "user", "text": user_text.strip()})
    # call agent synchronously (blocking) and append assistant reply
    with st.spinner("Calling language agent..."):
        try:
            response, model = ask_agent_sync(user_text)
        except Exception as e:
            st.error(f"Agent call failed: {e}")
            st.stop()
    # flatten response: try to extract some text from response object
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
    flat = flatten_to_text(response)
    # detect SQL from assistant reply
    sql_candidate = extract_sql_from_text(flat)
    used_fallback = False
    if not sql_candidate:
        # try to auto-generate using fallback NL->SQL helper
        try:
            fb = pgget.get_query(user_text)
            if fb:
                sql_candidate = fb
                used_fallback = True
        except Exception as e:
            logger.warning("Fallback generator failed: %s", e)
    # store assistant message
    assistant_record = {"role": "assistant", "text": flat, "sql": sql_candidate}
    if used_fallback:
        assistant_record["text"] = assistant_record["text"] + "\n\n*SQL candidate generated by fallback.*"
    st.session_state.chat_messages.append(assistant_record)
    # persist last_sql if we want quick access
    if sql_candidate:
        st.session_state.last_sql = sql_candidate

# After interactions: show last result / paginator controls (if any)
st.markdown("---")
st.header("Last query result")
if st.session_state.last_result is None:
    st.info("No query executed yet. Execute SQL from an assistant message.")
else:
    res = st.session_state.last_result
    tab_raw, tab_table = st.tabs(["Raw JSON", "Pretty Table"])
    with tab_raw:
        st.code(json.dumps(res, indent=2)[:200000])
    with tab_table:
        if isinstance(res, list) and res and all(isinstance(r, dict) for r in res):
            options = list(res[0].keys())
            prev_cols = st.session_state.get("visible_columns")
            default_cols = [c for c in (prev_cols or options) if c in options]
            if not default_cols:
                default_cols = options
            chosen = st.multiselect("Columns to display", options=options, default=default_cols)
            st.session_state.visible_columns = chosen or default_cols
            if chosen:
                def _select_row(r):
                    return {k: r.get(k) for k in chosen if k in r}
                st.dataframe([_select_row(r) for r in res])
            else:
                st.dataframe(res)
        else:
            st.write(res)

st.markdown("---")
st.caption("Made by Vishnu Pandey")
