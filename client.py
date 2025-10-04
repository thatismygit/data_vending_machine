import os
import re
import json
import asyncio
import shlex
import subprocess
from dotenv import load_dotenv
from typing import Any, Iterable, List, Dict, Union
import markdown
from bs4 import BeautifulSoup

# NEW: import your SQL-generator module so we can use its get_query() fallback
import postgres_get_query as pgget

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

def flatten_to_text(obj: Any) -> str:
    """
    Recursively walk the response object and collect all string content into one big string.
    This handles dicts, lists, objects with .__dict__, etc.
    """
    parts = []

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
            for item in x:
                walk(item)
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
    return "\n".join([p for p in parts if p is not None and p != ""])

def is_probable_sql(candidate: str) -> bool:
    """
    Heuristic checks to avoid misclassifying plain English as SQL.
    Returns True only if candidate looks like real SQL.
    """
    if not candidate or len(candidate.strip()) < 10:
        return False

    s = candidate.strip()
    # Quick sanity: must contain at least one SQL keyword (case-insensitive)
    sql_kw = re.search(r"\b(select|insert|update|delete|create|alter|drop|with|replace|truncate|grant|revoke|copy)\b", s, re.IGNORECASE)
    if not sql_kw:
        return False

    # If it starts with 'with', ensure it's a real WITH ... SELECT/INSERT/UPDATE/DELETE
    if re.match(r"^\s*with\b", s, re.IGNORECASE):
        # look for a SELECT/INSERT/UPDATE/DELETE somewhere after
        if not re.search(r"\b(select|insert|update|delete)\b", s, re.IGNORECASE):
            return False
        # also require some punctuation structure (comma, parentheses, or AS)
        if not re.search(r"\bAS\b|\(|,|;", s, re.IGNORECASE):
            # guard against "with ..." english clauses
            return False

    # If it contains SELECT, require FROM (common in real queries)
    if re.search(r"\bselect\b", s, re.IGNORECASE) and not re.search(r"\bfrom\b", s, re.IGNORECASE):
        # allow "select 1;" or "select count(*)" as valid
        if not re.search(r"\bselect\s+[0-9\(\)astrcinc_]+\b", s, re.IGNORECASE):
            return False

    # If it is basically a short English sentence (contains many spaces and ends with '.'), reject
    if s.count(" ") > 6 and s.strip().endswith(".") and not ";" in s:
        # long sentence ending with period — likely not SQL
        return False

    # presence of semicolon is a strong signal
    if ";" in s:
        return True

    # contain table-like identifiers: dot notation or quoted identifiers or common SQL punctuation
    if re.search(r"\w+\.\w+|\"[^\"]+\"|\[[^\]]+\]|\bfrom\b|\bwhere\b|\bjoin\b", s, re.IGNORECASE):
        return True

    # Fallback: if we reached here but matched a SQL keyword earlier, consider it probable
    return bool(sql_kw)

def extract_sql_from_text(text: str) -> str:
    """Find SQL in text: code fences, inline backticks, or first SQL statement by keyword.
       Then validate the candidate with is_probable_sql to avoid false positives.
    """
    if not text:
        return ""

    # 1) code fence with optional sql tag
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip()
        return candidate if is_probable_sql(candidate) else ""

    # 2) inline backticks with SQL-like content
    inline_matches = re.findall(r"`([^`]{10,})`", text)
    for m in inline_matches:
        if is_probable_sql(m):
            return m.strip()

    # 3) find SQL starting keyword and take until a semicolon (or reasonable boundary)
    start = re.search(r"\b(select|create|insert|update|delete|with|alter|drop)\b", text, re.IGNORECASE)
    if start:
        i = start.start()
        # prefer first semicolon after start
        sem = text.find(";", i)
        if sem != -1:
            candidate = text[i:sem+1].strip()
            return candidate if is_probable_sql(candidate) else ""
        # if no semicolon, take up to a blank line or 1000 chars
        m = re.search(r"\n\s*\n", text[i:])
        if m:
            candidate = text[i:i+m.start()].strip()
            return candidate if is_probable_sql(candidate) else ""
        candidate = text[i:i+2000].strip()
        return candidate if is_probable_sql(candidate) else ""

    return ""

def _truncate_cell(s: str, max_len: int = 50) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) > max_len:
        return s[:max_len-3] + "..."
    return s

def _compute_column_widths(rows: List[Dict[str, Any]], columns: List[str], max_col_width: int = 40) -> Dict[str, int]:
    widths = {}
    for col in columns:
        maxw = len(col)
        for r in rows:
            val = r.get(col, "")
            l = len(str(val)) if val is not None else 0
            if l > maxw:
                maxw = l
            if maxw >= max_col_width:
                maxw = max_col_width
                break
        widths[col] = min(maxw, max_col_width)
    return widths

def paginate_and_display(rows: List[Dict[str, Any]], page_size: int = 100) -> None:
    """
    Display list-of-dict rows in tabular form, `page_size` rows per page.
    Interactively ask user to navigate: n / p / q.
    """
    if not rows:
        print("(no rows)")
        return

    # Determine column order: keys of first row plus any other keys in order
    columns = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                columns.append(k)
    if not columns:
        print("(rows present but no columns)")
        return

    total = len(rows)
    total_pages = (total + page_size - 1) // page_size
    current = 0  # zero-based page index

    # Precompute column widths limited to a sensible max
    # We'll compute widths per page to better fit content (but cap widths)
    while True:
        start = current * page_size
        end = min(start + page_size, total)
        page_rows = rows[start:end]

        widths = _compute_column_widths(page_rows, columns, max_col_width=40)

        # Print header
        header_elems = []
        for col in columns:
            w = widths.get(col, len(col))
            header_elems.append(col.ljust(w))
        header_line = " | ".join(header_elems)
        sep_line = "-+-".join(["-" * widths.get(col, len(col)) for col in columns])

        print(f"\nShowing rows {start+1} - {end} of {total} (page {current+1}/{total_pages})")
        print(header_line)
        print(sep_line)

        # Print rows
        for r in page_rows:
            row_elems = []
            for col in columns:
                val = r.get(col, "")
                cell = _truncate_cell("" if val is None else str(val), max_len=widths.get(col, 40))
                row_elems.append(cell.ljust(widths.get(col, len(col))))
            print(" | ".join(row_elems))

        # Navigation prompt if multi-page
        if total_pages == 1:
            # single page only; break
            break

        # Options: n (next), p (prev), q (quit), goto page number
        prompt = "Navigate: [n]ext, [p]revious, [q]uit (or enter page number): "
        ans = input(prompt).strip().lower()
        if ans in ("n", "next", ""):
            if current + 1 < total_pages:
                current += 1
            else:
                print("Already at last page.")
        elif ans in ("p", "prev", "previous"):
            if current > 0:
                current -= 1
            else:
                print("Already at first page.")
        elif ans in ("q", "quit", "exit"):
            break
        else:
            # try parse page number
            try:
                pg = int(ans)
                if 1 <= pg <= total_pages:
                    current = pg - 1
                else:
                    print(f"Page number must be between 1 and {total_pages}.")
            except Exception:
                print("Unrecognized command.")

async def main():
    # Client used only to get the "postgres_get_query" tool (agent) as before
    client = MultiServerMCPClient(
        {
            "postgres_get_query": {
                "command": "python",
                "args": ["postgres_get_query.py"],
                "transport": "stdio",
            }
        }
    )

    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_key:
        raise RuntimeError("DEEPSEEK_API_KEY not found. Put it in environment or .env")

    tools = await client.get_tools()
    # create the react agent which can call the postgres_get_query tool
    model = ChatOpenAI(model="deepseek-chat", openai_api_key=deepseek_key, openai_api_base="https://api.deepseek.com/v1", temperature=0)
    agent = create_react_agent(model, tools)

    print("Enter requests. Type 'exit' to quit.")
    while True:
        raw = input(">>> ").strip()
        if not raw:
            continue
        if raw.lower() in ("exit", "quit"):
            break

        try:
            response = await agent.ainvoke({"messages": [{"role": "user", "content": raw}]})
        except Exception as e:
            print("Error calling agent:", e)
            continue

        # flatten the entire response object into text
        flat = flatten_to_text(response)

        # try to extract SQL from the agent reply
        sql_from_agent = BeautifulSoup(markdown.markdown(extract_sql_from_text(flat)), "html.parser").get_text()

        sql = sql_from_agent.strip() if sql_from_agent else ""

        # NEW: If agent reply didn't contain SQL, try using postgres_get_query.get_query(raw)
        # This makes the CLI treat (by default) user input as a DB query request unless
        # the SQL generator cannot produce anything confident.
        used_fallback = False
        if not sql:
            try:
                fallback_sql = pgget.get_query(raw)
                if fallback_sql:
                    sql = fallback_sql
                    used_fallback = True
                    print("\n[Note] No SQL found in agent reply — using postgres_get_query fallback to generate SQL from your input.\n")
            except Exception as e:
                print(f"[Warning] postgres_get_query fallback failed: {e}")

        if not sql:
            # No SQL found; show brief debug so you can inspect actual reply
            print("No SQL found — please rephrase.")
            snippet = flat.strip().replace("\n\n", "\n")
            if snippet:
                print("\n--- debug (response snippet) ---\n")
                print(snippet[:1000])
                if len(snippet) > 1000:
                    print("\n... (truncated)\n")
                print("\n--- end debug ---\n")
            continue

        # Print which path produced SQL
        if used_fallback:
            print("[SQL source] postgres_get_query fallback\n")
        else:
            print("[SQL source] agent reply\n")

        # Show the SQL and ask for confirmation before executing
        print("\n=== SQL to be executed ===\n")
        print(sql)
        print("\n==========================\n")
        confirm = input("Execute this SQL? (y/N): ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Execution cancelled by user.\n")
            continue

        # Execute using postgres_execute_query.py in CLI mode (we added --run-sql)
        try:
            # call the executor script and pass SQL on stdin; it will print JSON on stdout
            proc = subprocess.run(
                ["python", "postgres_execute_query.py", "--run-sql"],
                input=sql.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                print("Error executing SQL. stderr:")
                print(proc.stderr.decode("utf-8"))
                continue

            out = proc.stdout.decode("utf-8").strip()
            if not out:
                print("No output from executor.")
                continue

            # executor prints JSON; pretty-print / paginate if large
            try:
                result = json.loads(out)

                # If result is a list of dict rows and contains > 100 rows, paginate
                if isinstance(result, list) and len(result) > 100 and all(isinstance(r, dict) for r in result):
                    paginate_and_display(result, page_size=100)

                # After optionally showing pages, produce a natural-language explanation as before
                instructions= """Explain the following PostgreSQL query result in natural language \n
                                and to the point not taking any guesses if user ask for data inside the table \n
                                and its empty just say table is empty and provide them with the schema so they can enter data in it"""
                explanation = await model.ainvoke(f"{instructions}\n\n{json.dumps(result, indent=2)}")
                print("\n=== Natural Language Result ===")
                print(explanation.content if hasattr(explanation, "content") else str(explanation))
                print("================================\n")
            except Exception:
                # if not JSON, just print raw
                print("\n=== Execution raw output ===")
                print(out)
                print("===========================\n")
        except FileNotFoundError:
            print("Could not find postgres_execute_query.py. Make sure it's in the same directory.")
        except Exception as e:
            print("Error running executor:", e)

if __name__ == "__main__":
    asyncio.run(main())
