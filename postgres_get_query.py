from __future__ import annotations

import re
from typing import List, Optional, Tuple

SHOW_TABLES_RE = re.compile(r"\b(?:show|list|display)\b.*\b(?:table|tables)\b", re.IGNORECASE)

DESCRIBE_RE = re.compile(r"\b(?:describe|desc|show columns from|columns of)\s+([A-Za-z0-9_]+)\b", re.IGNORECASE)

COLUMNS_IN_OF_RE = re.compile(r"\bcolumns?\s+(?:in|of)\s+([A-Za-z0-9_]+)\b", re.IGNORECASE)

CREATE_TABLE_RE = re.compile(
    r"\bcreate\s+(?:a\s+)?table\s+(?:named\s+)?([A-Za-z0-9_]+)\s*(?:with|that has|having)?\s*(?:the\s*)?(?:attributes|columns|fields)?\s*[:\\-]?\s*(.*)",
    re.IGNORECASE,
)

INSERT_RE = re.compile(
    r"\binsert\s+into\s+([A-Za-z0-9_]+)\s*(?:\\(([^)]*)\\))?\s*(?:values|value)\s*(?:\\(([^)]*)\\))?",
    re.IGNORECASE,
)

def is_show_tables(message: str) -> bool:
    return bool(SHOW_TABLES_RE.search(message))

def is_describe_table(message: str) -> Optional[str]:
    """Return table name if message asks to describe/show columns for a table."""
    m = DESCRIBE_RE.search(message)
    if m:
        return m.group(1)
    m2 = COLUMNS_IN_OF_RE.search(message)
    if m2:
        return m2.group(1)
    return None

def is_create_table(message: str) -> Optional[Tuple[str, List[str]]]:
    """Parse simple "create table" natural language and return (table, [attrs])."""
    m = CREATE_TABLE_RE.search(message)
    if not m:
        return None
    table = m.group(1)
    tail = (m.group(2) or "").strip()
    if not tail:
        return (table, [])

    tail = re.sub(r"\b(and|with)\b", ",", tail, flags=re.IGNORECASE)
    parts = [p.strip() for p in re.split(r"[,,\\n\\|;]+", tail) if p.strip()]

    attrs: List[str] = []
    for p in parts:
        tokens = p.split()
        if len(tokens) == 1:
            attrs.append(tokens[0].strip('`"'))
        else:
            attrs.append(" ".join(tokens))
    return (table, attrs)

def is_insert(message: str) -> Optional[Tuple[str, List[str]]]:
    """Parse simple insert instructions. Returns (table, columns) where columns
    may be empty to indicate positional VALUES form.
    """
    m = INSERT_RE.search(message)
    if not m:
        return None
    table = m.group(1)
    cols = [c.strip() for c in (m.group(2) or "").split(",") if c.strip()] if m.group(2) else []
    vals = [v.strip() for v in (m.group(3) or "").split(",") if v.strip()] if m.group(3) else []
    if cols:
        return (table, cols)
    if vals:
        return (table, [])
    return (table, [])

def infer_sql_type(column_name: str) -> str:
    n = column_name.lower()
    if re.search(r"\\b(id|_id|empid|emp_id|user_id)\\b", n):
        return "SERIAL" if n in ("id", "emp_id") else "INT"
    if re.search(r"\\b(phone|phone_number|mobile|tel)\\b", n):
        return "VARCHAR(20)"
    if re.search(r"\\b(email|e-mail)\\b", n):
        return "VARCHAR(254)"
    if re.search(r"\\b(gender)\\b", n):
        return "VARCHAR(10)"
    if re.search(r"\\b(salary|price|amount|cost)\\b", n):
        return "DECIMAL(12,2)"
    if re.search(r"\\b(date|time|created|updated)\\b", n):
        return "TIMESTAMP"
    return "TEXT"

def build_create_table_sql(table: str, attrs: List[str]) -> str:
    cols_sql_parts: List[str] = []
    if not attrs:
        cols_sql_parts.append("id SERIAL PRIMARY KEY")
    else:
        for a in attrs:
            if " " in a:
                cols_sql_parts.append(a)
            else:
                colname = a
                coltype = infer_sql_type(colname)
                if coltype == "SERIAL" and colname.lower() in ("id", "emp_id"):
                    cols_sql_parts.append(f"{colname} SERIAL PRIMARY KEY")
                else:
                    cols_sql_parts.append(f"{colname} {coltype}")
    cols_sql = ", ".join(cols_sql_parts)
    return f"CREATE TABLE {table} ({cols_sql});"
def build_insert_sql(table: str, cols: List[str]) -> str:
    if cols:
        col_list = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        return f"INSERT INTO {table} ({col_list}) VALUES ({placeholders});"
    return f"INSERT INTO {table} VALUES (...);"

def get_query(message: str) -> str:
    """Convert a short natural-language request into a PostgreSQL query string.

    Returns an empty string when the intent cannot be confidently determined.
    """
    if not message:
        return ""

    msg = message.strip()

    if is_show_tables(msg):
        return "SELECT schemaname, tablename FROM pg_catalog.pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema') ORDER BY schemaname, tablename;"

    tab = is_describe_table(msg)
    if tab:
        return f"SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name = '{tab}' ORDER BY ordinal_position;"

    ct = is_create_table(msg)
    if ct:
        table_name, attrs = ct
        sql = build_create_table_sql(table_name, attrs)
        return " ".join(sql.split())

    ins = is_insert(msg)
    if ins:
        tbl, cols = ins
        return build_insert_sql(tbl, cols)

    if re.search(r"\\b(show|list)\\b.*\\b(current|all)\\b.*\\btable", msg, re.IGNORECASE) or re.search(r"\\bwhat\\b.*\\btable(s)?\\b", msg, re.IGNORECASE):
        return "SELECT schemaname, tablename FROM pg_catalog.pg_tables WHERE schemaname NOT IN ('pg_catalog', 'information_schema') ORDER BY schemaname, tablename;"

    return ""

if __name__ == "__main__":
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as e:
        raise RuntimeError("Attempted to run as MCP server but `mcp` package is not installed.") from e

    mcp = FastMCP("postgres_get_query")

    # Register `get_query` as an MCP tool at runtime (decorator-free registration)
    mcp.tool()(get_query)

    # Run the MCP server on stdio transport (compatible with local testing harnesses)
    mcp.run(transport="stdio")

