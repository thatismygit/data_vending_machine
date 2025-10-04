import os
import sys
import json
import argparse
import psycopg
from psycopg.rows import dict_row
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Create MCP server named "Postgres"
mcp = FastMCP("Postgres")

# Get connection info strictly from environment variables
PG_HOST = os.environ["PGHOST"]
PG_PORT = int(os.environ["PGPORT"])
PG_USER = os.environ["PGUSER"]
PG_PASSWORD = os.environ["PGPASSWORD"]
PG_DATABASE = os.environ["PGDATABASE"]

def get_connection():
    return psycopg.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DATABASE,
        row_factory=dict_row,
    )

@mcp.tool()
def run_query(sql: str) -> list[dict]:
    """
    Run a SQL query against the configured PostgreSQL database.
    Returns rows as a list of dictionaries.
    """
    print(f"[postgres_server:run_query] Executing SQL: {sql}")
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description:  # query returned rows
                    rows = cur.fetchall()
                    print(f"[postgres_server:run_query] Returned {len(rows)} rows")
                    return rows
                else:  # INSERT/UPDATE/DELETE
                    affected = cur.rowcount
                    print(f"[postgres_server:run_query] {affected} rows affected")
                    return [{"rows_affected": affected}]
    except Exception as e:
        print(f"[postgres_server:run_query] Error: {e}")
        return [{"error": str(e)}]

def cli_run_sql_from_stdin():
    """
    Read SQL from stdin and execute it, printing JSON to stdout.
    This mode is for simple CLI usage (called from client.py).
    """
    sql = sys.stdin.read()
    if not sql:
        print(json.dumps({"error": "No SQL provided on stdin"}))
        return 1

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description:  # SELECT-like
                    rows = cur.fetchall()
                    print(json.dumps(rows, default=str))
                else:
                    affected = cur.rowcount
                    print(json.dumps([{"rows_affected": affected}]))
        return 0
    except Exception as e:
        print(json.dumps([{"error": str(e)}]))
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-sql",
        action="store_true",
        help="Run in CLI mode: read SQL from stdin, execute it and print JSON result to stdout.",
    )
    args = parser.parse_args()

    if args.run_sql:
        # CLI execution mode
        exit_code = cli_run_sql_from_stdin()
        sys.exit(exit_code)
    else:
        # Run as MCP server as before
        mcp.run(transport="stdio")
