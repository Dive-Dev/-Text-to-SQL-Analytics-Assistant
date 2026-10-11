import os
import re
from typing import Dict, List, Optional, Tuple
import httpx


class LLMService:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        default_model: str = "duckdb-nsql:7b",
    ):
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", ollama_base_url).rstrip("/")
        self.default_model = os.getenv("OLLAMA_MODEL", default_model)

    def check_ollama_status(self) -> Tuple[bool, List[str]]:
        """Check if Ollama server is alive and return available models."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.ollama_base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    models = [m.get("name", "") for m in data.get("models", [])]
                    return True, models
        except Exception:
            pass
        return False, []

    def resolve_model_name(self, requested_model: Optional[str] = None) -> str:
        """Resolve model name against locally available Ollama models."""
        req = requested_model or self.default_model
        ok, models = self.check_ollama_status()
        if ok and models:
            if req in models:
                return req
            base_req = req.split(":")[0].lower()
            for m in models:
                if m.lower() == req.lower() or m.lower().startswith(base_req):
                    return m
            return models[0]
        return req


    def build_initial_prompt(self, question: str, schema_prompt: str) -> str:
        """Build initial Text-to-SQL prompt tailored for duckdb-nsql."""
        return f"""### Instruction:
Your task is to convert the user's question into a valid DuckDB SQL query.
Adhere strictly to these rules:
1. Return ONLY the raw SQL query. Do NOT include markdown formatting, backticks, or conversational explanations.
2. Only write read-only SELECT queries. Never use DML/DDL (INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE).
3. Only use tables and columns defined in the schema below.
4. For string matching, use ILIKE or LOWER() when appropriate.
5. Limit results to 100 rows unless the question specifies otherwise.

### Database Schema:
{schema_prompt}

### Question:
{question}

### Response:
"""

    def build_correction_prompt(
        self,
        question: str,
        schema_prompt: str,
        failed_sql: str,
        error_message: str,
        attempt: int,
    ) -> str:
        """Build reprompt for self-correcting QA validation loop."""
        return f"""### Instruction:
The previous SQL query failed DuckDB execution validation with an error (Attempt {attempt}).
Fix the SQL query so it runs successfully against the DuckDB database engine.
Adhere strictly to these rules:
1. Return ONLY the corrected raw SQL query. Do NOT include markdown backticks or explanations.
2. Fix the specific error reported below (e.g., nonexistent column name, syntax issue, or aggregation clause).
3. Use only tables and columns from the provided schema.

### Database Schema:
{schema_prompt}

### Question:
{question}

### Failed SQL:
{failed_sql}

### DuckDB Error:
{error_message}

### Corrected Response:
"""

    def clean_sql_output(self, raw_output: str) -> str:
        """Strip markdown fences, extra commentary, and clean SQL string."""
        text = raw_output.strip()

        # Extract content between ```sql ... ``` or ``` ... ```
        sql_block_match = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        if sql_block_match:
            text = sql_block_match.group(1).strip()

        # Remove leading 'Response:' or 'SQL:' labels
        text = re.sub(r"^(?:response|sql|query):\s*", "", text, flags=re.IGNORECASE).strip()

        # Remove any trailing semicolon or comments
        lines = [line for line in text.splitlines() if not line.strip().startswith("--")]
        cleaned = " ".join(lines).strip()

        # Normalize spaces
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def generate_sql(
        self,
        prompt: str,
        provider: str = "ollama",
        model: Optional[str] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        question_context: Optional[str] = None,
        schema_context: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Generate SQL query via selected provider or fallback.
        Returns: (cleaned_sql, source_info)
        """
        model_name = self.resolve_model_name(model)

        if provider.lower() == "ollama":
            # Attempt to call Ollama
            try:
                with httpx.Client(timeout=45.0) as client:
                    payload = {
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": temperature},
                    }
                    res = client.post(f"{self.ollama_base_url}/api/generate", json=payload)
                    if res.status_code == 200:
                        raw_response = res.json().get("response", "")
                        return self.clean_sql_output(raw_response), f"Ollama ({model_name})"
                    else:
                        print(f"Ollama returned HTTP {res.status_code}: {res.text}")
            except Exception as e:
                print(f"Ollama connection error: {e}")

        # Cloud OpenAI-compatible API if requested or if API key provided
        if provider.lower() in ["openai", "cloud"] and api_key:
            try:
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model_name if model_name != self.default_model else "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                }
                with httpx.Client(timeout=30.0) as client:
                    res = client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                    if res.status_code == 200:
                        raw = res.json()["choices"][0]["message"]["content"]
                        return self.clean_sql_output(raw), f"Cloud API ({payload['model']})"
            except Exception as e:
                print(f"Cloud API error: {e}")

        # Offline / Heuristic Generator fallback for immediate out-of-the-box operation
        heuristic_sql = self._generate_heuristic_sql(
            question=question_context or prompt,
            schema_context=schema_context or "",
        )
        return heuristic_sql, "Local Heuristic Engine (Offline / Ollama Standby)"

    def _generate_heuristic_sql(self, question: str, schema_context: str) -> str:
        """Intelligent SQL synthesis based on question intent and available schema tables."""
        q_lower = question.lower()

        # Detect relevant table
        is_qa = "qa" in q_lower or "test" in q_lower or "suite" in q_lower or "browser" in q_lower
        table_name = "qa_test_runs" if is_qa else "e_commerce_sales"

        if "e_commerce_sales" in schema_context and not is_qa:
            if "top" in q_lower and ("customer" in q_lower or "spending" in q_lower or "revenue" in q_lower):
                limit_match = re.search(r"top\s+(\d+)", q_lower)
                limit = limit_match.group(1) if limit_match else "5"
                return f"SELECT customer_name, country, ROUND(SUM(total_amount), 2) AS total_revenue, COUNT(*) AS orders_count FROM {table_name} GROUP BY customer_name, country ORDER BY total_revenue DESC LIMIT {limit};"

            if "category" in q_lower or "product" in q_lower:
                return f"SELECT category, ROUND(SUM(total_amount), 2) AS total_sales, COUNT(*) AS items_sold FROM {table_name} GROUP BY category ORDER BY total_sales DESC;"

            if "country" in q_lower:
                return f"SELECT country, COUNT(*) AS total_orders, ROUND(SUM(total_amount), 2) AS total_revenue FROM {table_name} GROUP BY country ORDER BY total_revenue DESC;"

            if "payment" in q_lower:
                return f"SELECT payment_method, COUNT(*) AS transaction_count, ROUND(SUM(total_amount), 2) AS total_volume FROM {table_name} GROUP BY payment_method ORDER BY transaction_count DESC;"

            if "discount" in q_lower:
                return f"SELECT product_name, category, unit_price, discount, total_amount FROM {table_name} WHERE discount > 0 ORDER BY discount DESC LIMIT 10;"

            # Default analytical query on e-commerce
            return f"SELECT order_id, customer_name, category, product_name, total_amount, order_date, order_status FROM {table_name} ORDER BY order_date DESC LIMIT 15;"

        elif "qa_test_runs" in schema_context or is_qa:
            if "failed" in q_lower or "failure" in q_lower or "error" in q_lower:
                return f"SELECT test_id, suite_name, test_name, browser, error_type, execution_time_ms FROM {table_name} WHERE status = 'FAILED' ORDER BY execution_time_ms DESC;"

            if "suite" in q_lower or "module" in q_lower:
                return f"SELECT suite_name, COUNT(*) AS total_tests, SUM(CASE WHEN status = 'PASSED' THEN 1 ELSE 0 END) AS passed_tests, SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failed_tests FROM {table_name} GROUP BY suite_name ORDER BY total_tests DESC;"

            if "slowest" in q_lower or "time" in q_lower or "duration" in q_lower or "execution" in q_lower:
                return f"SELECT test_name, suite_name, browser, execution_time_ms, status FROM {table_name} ORDER BY execution_time_ms DESC LIMIT 5;"

            if "browser" in q_lower:
                return f"SELECT browser, COUNT(*) AS total_runs, ROUND(AVG(execution_time_ms), 1) AS avg_duration_ms, SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS failures FROM {table_name} GROUP BY browser ORDER BY total_runs DESC;"

            # Default analytical query on QA
            return f"SELECT test_id, suite_name, test_name, execution_time_ms, status, browser, run_date FROM {table_name} ORDER BY run_date DESC LIMIT 15;"

        # Generic fallback
        return "SELECT * FROM (SELECT 'Please load a dataset or connect Ollama' AS status);"
