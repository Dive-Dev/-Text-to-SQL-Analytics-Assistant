import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.db_service import DuckDBService, get_db_service
from backend.llm_service import LLMService
from backend.models import (
    ExecuteRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SchemaResponse,
)
from backend.validator import check_sql_safety, generate_data_analysis, run_qa_validation_loop, suggest_chart

app = FastAPI(
    title="Text-to-SQL Analytics Assistant API",
    description="Localized, privacy-first Text-to-SQL engine with DuckDB and Ollama QA validation loop",
    version="1.0.0",
)

# Enable CORS for local Streamlit interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db_service: DuckDBService = get_db_service()
llm_service: LLMService = LLMService()


@app.get("/")
def read_root():
    return {
        "service": "Text-to-SQL Analytics Assistant",
        "status": "online",
        "docs": "/docs",
        "tables": db_service.list_tables(),
    }


@app.get("/health", response_model=HealthResponse)
def get_health():
    tables = db_service.list_tables()
    ollama_ok, models = llm_service.check_ollama_status()
    active_model = llm_service.default_model
    if ollama_ok and models:
        active_model = models[0] if active_model not in models else active_model

    return HealthResponse(
        status="healthy",
        duckdb_connected=True,
        tables_count=len(tables),
        ollama_connected=ollama_ok,
        active_model=active_model,
    )


@app.get("/schema", response_model=SchemaResponse)
def get_schema(tables: Optional[str] = Query(None, description="Comma-separated table names to filter")):
    table_filter = [t.strip() for t in tables.split(",")] if tables else None
    metadata = db_service.get_schema_metadata(table_filter)
    schema_prompt = db_service.get_schema_prompt(table_filter)
    return SchemaResponse(tables=metadata, schema_prompt=schema_prompt)


@app.get("/tables")
def get_tables():
    all_tables = db_service.list_tables()
    tables_info = []
    for t in all_tables:
        count = db_service.get_table_row_count(t)
        tables_info.append({"name": t, "row_count": count})
    return {"tables": tables_info}


@app.post("/query", response_model=QueryResponse)
def query_natural_language(req: QueryRequest):
    """Main Text-to-SQL endpoint with self-correcting QA validation loop."""
    response = run_qa_validation_loop(
        question=req.question,
        db_service=db_service,
        llm_service=llm_service,
        max_retries=3,
        provider=req.provider or "ollama",
        model=req.model,
        temperature=req.temperature or 0.1,
        api_key=req.api_key,
        table_filter=req.table_filter,
    )
    return response


@app.post("/execute", response_model=QueryResponse)
def execute_direct_sql(req: ExecuteRequest):
    """Directly execute a SQL statement with safety verification."""
    is_safe, err = check_sql_safety(req.sql)
    if not is_safe:
        raise HTTPException(status_code=400, detail=err)

    is_valid, plan_err, plan = db_service.explain_query(req.sql)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"DuckDB Plan Error: {plan_err}")

    ok, df, exec_err, elapsed_ms = db_service.execute_query(req.sql)
    if not ok or df is None:
        raise HTTPException(status_code=400, detail=f"Execution Error: {exec_err}")

    chart_sug = suggest_chart(df)
    analysis_text = generate_data_analysis(df, "Direct SQL Execution")
    return QueryResponse(
        success=True,
        question="Direct SQL Execution",
        sql=req.sql,
        columns=df.columns.tolist(),
        rows=df.to_dict(orient="records"),
        row_count=len(df),
        execution_time_ms=round(elapsed_ms, 2),
        validation_attempts=[],
        chart_suggestion=chart_sug,
        llm_source="Direct SQL",
        analysis=analysis_text,
    )


@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Parquet file to create a new analytical table in DuckDB."""
    filename = file.filename
    if not filename.endswith((".csv", ".parquet")):
        raise HTTPException(status_code=400, detail="Only .csv and .parquet files are supported.")

    save_path = db_service.data_dir / filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        table_name = Path(filename).stem
        loaded = db_service.load_file(str(save_path), table_name)
        if not loaded:
            raise HTTPException(status_code=500, detail="Failed to load file into DuckDB.")

        row_count = db_service.get_table_row_count(table_name)
        return {
            "success": True,
            "table_name": table_name,
            "row_count": row_count,
            "file_path": str(save_path),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
