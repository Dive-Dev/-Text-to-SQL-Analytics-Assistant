import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import duckdb
import pandas as pd

from backend.models import ColumnInfo, TableMetadata


class DuckDBService:
    def __init__(self, db_path: Optional[str] = None, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
        self.db_path = db_path or ":memory:"
        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        self._init_data()

    def _init_data(self) -> None:
        """Scan data_dir and load CSV and Parquet files into tables."""
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in self.data_dir.glob("*"):
            if file_path.suffix.lower() in [".csv", ".parquet"]:
                table_name = re.sub(r"[^a-zA-Z0-9_]", "_", file_path.stem)
                self.load_file(str(file_path), table_name)

    def load_file(self, file_path: str, table_name: str) -> bool:
        """Load a CSV or Parquet file into DuckDB as a table."""
        path = Path(file_path)
        safe_table = re.sub(r"[^a-zA-Z0-9_]", "_", table_name)
        try:
            if path.suffix.lower() == ".csv":
                self.conn.execute(
                    f"CREATE OR REPLACE TABLE {safe_table} AS SELECT * FROM read_csv_auto(?)",
                    [str(path)],
                )
            elif path.suffix.lower() == ".parquet":
                self.conn.execute(
                    f"CREATE OR REPLACE TABLE {safe_table} AS SELECT * FROM read_parquet(?)",
                    [str(path)],
                )
            return True
        except Exception as e:
            print(f"Error loading file {file_path} into table {safe_table}: {e}")
            return False

    def list_tables(self) -> List[str]:
        """List all user tables in the database."""
        result = self.conn.execute(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main' AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()
        return [row[0] for row in result]

    def get_table_row_count(self, table_name: str) -> int:
        """Get the row count for a table."""
        try:
            res = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            return res[0] if res else 0
        except Exception:
            return 0

    def get_schema_metadata(self, table_filter: Optional[List[str]] = None) -> List[TableMetadata]:
        """Get structured metadata including column types and sample distinct values."""
        all_tables = self.list_tables()
        if table_filter:
            tables = [t for t in all_tables if t in table_filter]
        else:
            tables = all_tables

        table_metadata_list: List[TableMetadata] = []

        for table in tables:
            row_count = self.get_table_row_count(table)
            cols_res = self.conn.execute(
                """
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_schema = 'main' AND table_name = ?
                ORDER BY ordinal_position
                """,
                [table],
            ).fetchall()

            columns: List[ColumnInfo] = []
            for col_name, data_type in cols_res:
                # Fetch up to 3 distinct sample values for context (especially helpful for categorical text)
                sample_vals: List[str] = []
                try:
                    sample_query = f"""
                        SELECT DISTINCT CAST("{col_name}" AS VARCHAR) 
                        FROM "{table}" 
                        WHERE "{col_name}" IS NOT NULL 
                        LIMIT 3
                    """
                    sample_rows = self.conn.execute(sample_query).fetchall()
                    sample_vals = [str(r[0]) for r in sample_rows if r[0] is not None]
                except Exception:
                    pass

                columns.append(ColumnInfo(name=col_name, type=data_type, sample_values=sample_vals))

            table_metadata_list.append(
                TableMetadata(table_name=table, row_count=row_count, columns=columns)
            )

        return table_metadata_list

    def get_schema_prompt(self, table_filter: Optional[List[str]] = None) -> str:
        """Format the database schema as DDL-style prompt context with sample values."""
        metadata = self.get_schema_metadata(table_filter)
        prompt_lines = ["-- DuckDB SQL Database Schema:\n"]

        for table in metadata:
            prompt_lines.append(f"CREATE TABLE {table.table_name} (")
            col_definitions = []
            for col in table.columns:
                samples_str = ""
                if col.sample_values:
                    # e.g., -- examples: 'Electronics', 'Home & Kitchen'
                    quoted_samples = [f"'{s}'" if not s.isnumeric() else s for s in col.sample_values[:3]]
                    samples_str = f"  -- e.g., {', '.join(quoted_samples)}"
                col_definitions.append(f"    {col.name} {col.type}{samples_str}")
            prompt_lines.append(",\n".join(col_definitions))
            prompt_lines.append(");\n")

        return "\n".join(prompt_lines)

    def explain_query(self, sql: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validate query plan using DuckDB EXPLAIN.
        Returns: (is_valid, error_message, execution_plan)
        """
        clean_sql = sql.strip().rstrip(";")
        try:
            # We explain the query plan to check table and column existence and syntax
            explain_result = self.conn.execute(f"EXPLAIN {clean_sql}").fetchall()
            plan_str = "\n".join([f"{r[0]}: {r[1]}" if len(r) > 1 else str(r[0]) for r in explain_result])
            return True, None, plan_str
        except duckdb.Error as e:
            return False, str(e), None
        except Exception as e:
            return False, f"Unexpected error during EXPLAIN: {str(e)}", None

    def execute_query(self, sql: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str], float]:
        """Execute validated query against DuckDB and return DataFrame, error, execution time in ms."""
        clean_sql = sql.strip().rstrip(";")
        start_time = time.perf_counter()
        try:
            df = self.conn.execute(clean_sql).df()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return True, df, None, elapsed_ms
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return False, None, str(e), elapsed_ms


# Global singleton instance
_db_instance: Optional[DuckDBService] = None


def get_db_service() -> DuckDBService:
    global _db_instance
    if _db_instance is None:
        _db_instance = DuckDBService()
    return _db_instance
