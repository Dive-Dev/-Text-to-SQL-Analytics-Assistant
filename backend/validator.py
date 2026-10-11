import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from backend.db_service import DuckDBService
from backend.llm_service import LLMService
from backend.models import ChartSuggestion, QueryResponse, ValidationAttempt

FORBIDDEN_KEYWORDS = [
    r"\bDROP\b",
    r"\bDELETE\b",
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bCREATE\b",
    r"\bREPLACE\b",
    r"\bATTACH\b",
    r"\bDETACH\b",
    r"\bCOPY\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bPRAGMA\b",
]


def check_sql_safety(sql: str) -> Tuple[bool, Optional[str]]:
    """Verify that the SQL query is strictly read-only and does not contain DML or DDL commands."""
    clean = sql.strip().rstrip(";")
    if not clean:
        return False, "Query is empty."

    # Verify query begins with a read-only keyword (SELECT or WITH)
    first_word = clean.split()[0].upper()
    if first_word not in ["SELECT", "WITH", "EXPLAIN", "DESCRIBE"]:
        return (
            False,
            f"Prohibited query command: '{first_word}'. Only SELECT queries are permitted in this analytics assistant.",
        )

    # Check for forbidden destructive keywords
    for pattern in FORBIDDEN_KEYWORDS:
        match = re.search(pattern, clean, re.IGNORECASE)
        if match:
            forbidden_term = match.group(0).upper()
            return (
                False,
                f"Security Sandbox Violation: Prohibited DML/DDL command '{forbidden_term}' detected. Only read-only analytical queries are permitted.",
            )

    return True, None


def suggest_chart(df: pd.DataFrame) -> ChartSuggestion:
    """Intelligently infer the best chart type and axis mappings from the query result."""
    if df.empty or len(df.columns) < 2:
        return ChartSuggestion(chart_type="table")

    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [
        c
        for c in cols
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or any(k in c.lower() for k in ["date", "time", "day", "month", "year"])
    ]
    categorical_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]

    # Time series line chart
    if datetime_cols and numeric_cols:
        return ChartSuggestion(
            chart_type="line",
            x_axis=datetime_cols[0],
            y_axis=numeric_cols[0],
            title=f"{numeric_cols[0].replace('_', ' ').title()} Over Time",
        )

    # Categorical breakdown -> Bar chart or Pie chart
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        if len(df) <= 5 and any(k in cat_col.lower() for k in ["status", "type", "method", "category"]):
            return ChartSuggestion(
                chart_type="pie",
                x_axis=cat_col,
                y_axis=num_col,
                title=f"Distribution by {cat_col.replace('_', ' ').title()}",
            )
        return ChartSuggestion(
            chart_type="bar",
            x_axis=cat_col,
            y_axis=num_col,
            title=f"{num_col.replace('_', ' ').title()} by {cat_col.replace('_', ' ').title()}",
        )

    # Two numeric columns -> Scatter plot
    if len(numeric_cols) >= 2:
        return ChartSuggestion(
            chart_type="scatter",
            x_axis=numeric_cols[0],
            y_axis=numeric_cols[1],
            title=f"{numeric_cols[1].replace('_', ' ').title()} vs {numeric_cols[0].replace('_', ' ').title()}",
        )

    return ChartSuggestion(chart_type="table")


def generate_data_analysis(df: pd.DataFrame, question: str = "") -> str:
    """Generate meaningful analytical insights and summary from query results."""
    if df.empty:
        return "The query completed successfully, but returned no matching records."

    insights = []
    num_rows = len(df)
    cols = df.columns.tolist()

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    datetime_cols = [
        c
        for c in cols
        if pd.api.types.is_datetime64_any_dtype(df[c])
        or any(k in c.lower() for k in ["date", "time", "day", "month", "year"])
    ]
    categorical_cols = [c for c in cols if c not in numeric_cols and c not in datetime_cols]

    # 1. High-level Scope Summary
    if num_rows == 1:
        insights.append("• **Result Overview**: Retrieved 1 aggregated metric record.")
    else:
        insights.append(f"• **Volume & Dimensions**: Analyzed **{num_rows:,}** rows across {len(cols)} columns.")

    # 2. Key Numeric Metrics / Aggregates
    for col in numeric_cols[:2]:
        col_clean = col.replace("_", " ").title()
        col_sum = df[col].sum()
        col_avg = df[col].mean()
        col_max = df[col].max()
        col_min = df[col].min()

        if isinstance(col_sum, float):
            sum_str = f"{col_sum:,.2f}"
            avg_str = f"{col_avg:,.2f}"
            max_str = f"{col_max:,.2f}"
            min_str = f"{col_min:,.2f}"
        else:
            sum_str = f"{col_sum:,}"
            avg_str = f"{col_avg:.1f}"
            max_str = f"{col_max:,}"
            min_str = f"{col_min:,}"

        is_currency = any(k in col.lower() for k in ["revenue", "sales", "amount", "price", "cost", "total"])
        prefix = "$" if is_currency else ""

        if num_rows > 1:
            insights.append(
                f"• **{col_clean}**: Cumulative total of **{prefix}{sum_str}** (average **{prefix}{avg_str}**; range: **{prefix}{min_str}** to **{prefix}{max_str}**)."
            )
        else:
            insights.append(f"• **{col_clean}**: Evaluated to **{prefix}{sum_str}**.")

    # 3. Categorical distribution and Top Performer
    if categorical_cols and numeric_cols and num_rows > 1:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        cat_clean = cat_col.replace("_", " ").title()
        num_clean = num_col.replace("_", " ").title()

        try:
            sorted_df = df.sort_values(by=num_col, ascending=False)
            top_row = sorted_df.iloc[0]
            top_val = top_row[num_col]
            top_name = top_row[cat_col]

            total_num = df[num_col].sum()
            is_curr = any(k in num_col.lower() for k in ["revenue", "sales", "amount", "price", "cost", "total"])
            pfix = "$" if is_curr else ""
            top_fmt = f"{pfix}{top_val:,.2f}" if isinstance(top_val, float) else f"{pfix}{top_val:,}"

            if total_num > 0:
                share = (top_val / total_num) * 100
                insights.append(
                    f"• **Dominant Segment**: **{top_name}** leads in **{cat_clean}** with **{top_fmt}** ({share:.1f}% of total {num_clean.lower()})."
                )
            else:
                insights.append(
                    f"• **Top Performer**: Highest recorded **{num_clean}** is **{top_name}** with **{top_fmt}**."
                )

            if num_rows > 2:
                bottom_row = sorted_df.iloc[-1]
                bot_val = bottom_row[num_col]
                bot_name = bottom_row[cat_col]
                bot_fmt = f"{pfix}{bot_val:,.2f}" if isinstance(bot_val, float) else f"{pfix}{bot_val:,}"
                insights.append(
                    f"• **Baseline**: Lowest recorded entry is **{bot_name}** at **{bot_fmt}**."
                )
        except Exception:
            pass

    # 4. QA Test Status Insights (if status column present)
    status_matches = [c for c in cols if c.lower() in ["status", "test_status", "result"]]
    if status_matches:
        stat_col = status_matches[0]
        counts = df[stat_col].value_counts()
        total_tests = len(df)
        parts = [f"**{st}**: {cnt} ({cnt/total_tests*100:.1f}%)" for st, cnt in counts.items()]
        insights.append(f"• **Execution Breakdown**: {', '.join(parts)}.")

    # 5. Timeline / Trend Insights (if datetime column present)
    if datetime_cols and num_rows > 1:
        date_col = datetime_cols[0]
        date_clean = date_col.replace("_", " ").title()
        try:
            min_date = str(df[date_col].min())[:10]
            max_date = str(df[date_col].max())[:10]
            insights.append(f"• **Timeline Span**: Results span **{date_clean}** from **{min_date}** to **{max_date}**.")
        except Exception:
            pass

    return "\n\n".join(insights)


def run_qa_validation_loop(
    question: str,
    db_service: DuckDBService,
    llm_service: LLMService,
    max_retries: int = 3,
    provider: str = "ollama",
    model: Optional[str] = None,
    temperature: float = 0.1,
    api_key: Optional[str] = None,
    table_filter: Optional[List[str]] = None,
) -> QueryResponse:
    """Execute the multi-step QA validation and self-correction loop as defined in architecture.

    1. Prompt LLM with schema and user question.
    2. Check SQL safety (block DML/DDL).
    3. Dry-run SQL with DuckDB EXPLAIN.
    4. If error occurs, reprompt LLM with exact DuckDB error (up to max_retries).
    5. Execute validated SQL and return results + audit log.
    """
    schema_prompt = db_service.get_schema_prompt(table_filter)
    validation_attempts: List[ValidationAttempt] = []
    current_sql = ""
    last_error: Optional[str] = None
    llm_source: Optional[str] = None

    effective_provider = provider
    for attempt in range(1, max_retries + 1):
        # Step 1: Prompt LLM
        if attempt == 1:
            prompt = llm_service.build_initial_prompt(question, schema_prompt)
        else:
            prompt = llm_service.build_correction_prompt(
                question=question,
                schema_prompt=schema_prompt,
                failed_sql=current_sql,
                error_message=last_error or "Unknown error",
                attempt=attempt,
            )

        current_sql, source = llm_service.generate_sql(
            prompt=prompt,
            provider=effective_provider,
            model=model,
            temperature=temperature,
            api_key=api_key,
            question_context=question,
            schema_context=schema_prompt,
        )
        llm_source = source
        # If primary provider fell back to heuristic engine, stick with heuristic for further retries
        if "Heuristic" in source or "Standby" in source:
            effective_provider = "offline_demo"

        # Step 2: DML/DDL Safety Validation
        is_safe, safety_err = check_sql_safety(current_sql)
        if not is_safe:
            last_error = safety_err
            validation_attempts.append(
                ValidationAttempt(
                    attempt_number=attempt,
                    sql=current_sql,
                    valid=False,
                    error_message=safety_err,
                )
            )
            continue

        # Step 3: DuckDB EXPLAIN validation sandbox
        is_valid, duckdb_err, plan = db_service.explain_query(current_sql)
        if not is_valid:
            last_error = duckdb_err
            validation_attempts.append(
                ValidationAttempt(
                    attempt_number=attempt,
                    sql=current_sql,
                    valid=False,
                    error_message=duckdb_err,
                )
            )
            continue

        # Success! Query is safe and execution plan passed
        validation_attempts.append(
            ValidationAttempt(
                attempt_number=attempt,
                sql=current_sql,
                valid=True,
                execution_plan=plan,
            )
        )

        # Step 4: Execute Validated SQL
        exec_ok, df, exec_err, elapsed_ms = db_service.execute_query(current_sql)
        if not exec_ok or df is None:
            last_error = exec_err
            continue

        # Step 5: Format response, chart suggestion, and analytical summary
        chart_sug = suggest_chart(df)
        analysis_text = generate_data_analysis(df, question)
        records = df.to_dict(orient="records")

        return QueryResponse(
            success=True,
            question=question,
            sql=current_sql,
            columns=df.columns.tolist(),
            rows=records,
            row_count=len(df),
            execution_time_ms=round(elapsed_ms, 2),
            validation_attempts=validation_attempts,
            chart_suggestion=chart_sug,
            llm_source=llm_source,
            analysis=analysis_text,
        )

    # If all retries exhausted
    return QueryResponse(
        success=False,
        question=question,
        sql=current_sql,
        validation_attempts=validation_attempts,
        error=f"QA loop failed after {max_retries} attempts. Last DuckDB error: {last_error}",
        llm_source=llm_source,
    )
