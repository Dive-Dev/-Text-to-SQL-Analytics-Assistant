"""Automated test suite verifying DuckDB engine, schema extraction,

QA validation sandbox, DML blocking, and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.db_service import DuckDBService
from backend.main import app
from backend.validator import check_sql_safety, run_qa_validation_loop


def test_duckdb_service_init_and_tables():
    db = DuckDBService()
    tables = db.list_tables()
    assert "e_commerce_sales" in tables
    assert "qa_test_runs" in tables
    assert db.get_table_row_count("e_commerce_sales") > 0
    assert db.get_table_row_count("qa_test_runs") > 0


def test_schema_prompt_format():
    db = DuckDBService()
    prompt = db.get_schema_prompt()
    assert "CREATE TABLE e_commerce_sales" in prompt
    assert "CREATE TABLE qa_test_runs" in prompt
    assert "total_amount" in prompt
    assert "suite_name" in prompt


def test_sql_safety_check_dml_blocking():
    # Dangerous statements must be blocked
    unsafe_queries = [
        "DROP TABLE e_commerce_sales;",
        "DELETE FROM qa_test_runs WHERE status = 'FAILED';",
        "INSERT INTO e_commerce_sales (order_id) VALUES ('ORD-9999');",
        "UPDATE e_commerce_sales SET total_amount = 0;",
        "ALTER TABLE qa_test_runs ADD COLUMN dummy INT;",
        "TRUNCATE e_commerce_sales;",
    ]
    for q in unsafe_queries:
        is_safe, err = check_sql_safety(q)
        assert not is_safe, f"Query should have been blocked: {q}"
        assert err is not None

    # Safe SELECT statement must pass
    safe_query = "SELECT customer_name, SUM(total_amount) FROM e_commerce_sales GROUP BY customer_name;"
    is_safe, err = check_sql_safety(safe_query)
    assert is_safe
    assert err is None


def test_duckdb_explain_query():
    db = DuckDBService()

    # Valid query
    ok, err, plan = db.explain_query("SELECT category, COUNT(*) FROM e_commerce_sales GROUP BY category")
    assert ok
    assert err is None
    assert plan is not None

    # Invalid query with nonexistent column
    ok, err, plan = db.explain_query("SELECT non_existent_col FROM e_commerce_sales")
    assert not ok
    assert err is not None
    assert "not found" in err.lower() or "binder error" in err.lower()


def test_api_health_and_tables_endpoints():
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["duckdb_connected"] is True
    assert data["tables_count"] >= 2

    res_tables = client.get("/tables")
    assert res_tables.status_code == 200
    tables = [t["name"] for t in res_tables.json()["tables"]]
    assert "e_commerce_sales" in tables


def test_api_query_endpoint():
    client = TestClient(app)

    payload = {
        "question": "What are the top 3 customers by revenue?",
        "provider": "offline_demo",
    }
    res = client.post("/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert len(data["rows"]) > 0
    assert data["row_count"] > 0
    assert data["sql"] is not None
    assert len(data["validation_attempts"]) >= 1
    assert data["validation_attempts"][0]["valid"] is True
    assert data["analysis"] is not None
    assert len(data["analysis"]) > 0


def test_api_execute_direct_sql_endpoint():
    client = TestClient(app)

    res = client.post(
        "/execute",
        json={"sql": "SELECT suite_name, COUNT(*) AS cnt FROM qa_test_runs GROUP BY suite_name;"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["row_count"] > 0
    assert data["analysis"] is not None


def test_api_execute_blocks_dml():
    client = TestClient(app)

    res = client.post("/execute", json={"sql": "DROP TABLE e_commerce_sales;"})
    assert res.status_code == 400
    assert "Prohibited" in res.json()["detail"]
