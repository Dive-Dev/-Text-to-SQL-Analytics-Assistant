from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question from the user")
    table_filter: Optional[List[str]] = Field(None, description="Optional specific tables to query")
    provider: Optional[str] = Field("ollama", description="LLM provider: ollama, openai, gemini, or offline_demo")
    model: Optional[str] = Field("duckdb-nsql:7b", description="Model name in provider")
    api_key: Optional[str] = Field(None, description="Optional API key for cloud providers")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")


class ValidationAttempt(BaseModel):
    attempt_number: int
    sql: str
    valid: bool
    error_message: Optional[str] = None
    execution_plan: Optional[str] = None


class ChartSuggestion(BaseModel):
    chart_type: str = "table"  # bar, line, scatter, pie, histogram, table
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    title: Optional[str] = None
    color: Optional[str] = None


class QueryResponse(BaseModel):
    success: bool
    question: str
    sql: Optional[str] = None
    columns: List[str] = []
    rows: List[Dict[str, Any]] = []
    row_count: int = 0
    execution_time_ms: float = 0.0
    validation_attempts: List[ValidationAttempt] = []
    error: Optional[str] = None
    chart_suggestion: Optional[ChartSuggestion] = None
    llm_source: Optional[str] = None
    analysis: Optional[str] = None


class ExecuteRequest(BaseModel):
    sql: str = Field(..., description="Direct SQL query to execute")


class ColumnInfo(BaseModel):
    name: str
    type: str
    sample_values: List[str] = []


class TableMetadata(BaseModel):
    table_name: str
    row_count: int
    columns: List[ColumnInfo]


class SchemaResponse(BaseModel):
    tables: List[TableMetadata]
    schema_prompt: str


class HealthResponse(BaseModel):
    status: str
    duckdb_connected: bool
    tables_count: int
    ollama_connected: bool
    active_model: str
