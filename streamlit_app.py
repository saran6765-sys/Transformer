import os
import sys
import pathlib
import json
import time
import hashlib
import io
import streamlit as st
from typing import Optional, Dict, List, Tuple
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    ROOT = pathlib.Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if SRC.exists() and str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
except Exception:
    pass

from infa_to_pyspark.secrets_loader import load_secrets_into_env
from infa_to_pyspark.agents import (
    extractor,
    extractor_streaming,
    normalizer,
    derive_logic,
    build_plan,
    load_few_shots,
    validator,
    summarize_ast,
    extract_code_blocks,
    extract_sql_sections,
)
from infa_to_pyspark.prompt_utils import assemble_context, compress_ast, shorten_text
from infa_to_pyspark.direct_llm import DirectChatLLM


def _load_extra_headers() -> Dict[str, str]:
    raw = os.getenv("LLM_EXTRA_HEADERS")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass
    return {}


def get_llm(api_key: Optional[str], model: Optional[str], base_url: Optional[str] = None):
    """Return an LLM client configured for the chosen provider."""
    llm_timeout = float(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    llm_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    provider = (os.getenv("LLM_PROVIDER") or "openai").lower()
    key_header = os.getenv("LLM_KEY_HEADER")
    direct_hint = os.getenv("LLM_DEPLOYMENT") or key_header or os.getenv("LLM_API_VERSION")
    if provider in {"direct", "http", "rest", "gateway"} or direct_hint:
        headers = _load_extra_headers()
        temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
        system_prompt = os.getenv("LLM_SYSTEM_PROMPT")
        return DirectChatLLM(
            base_url=base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "",
            api_key=api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            model=model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL"),
            deployment=os.getenv("LLM_DEPLOYMENT"),
            key_header=key_header,
            key_prefix=os.getenv("LLM_KEY_PREFIX"),
            api_version=os.getenv("LLM_API_VERSION"),
            temperature=temperature,
            timeout=llm_timeout,
            extra_headers=headers,
            system_prompt=system_prompt,
        )

    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("LLM_API_KEY", api_key)
    kwargs = {"model": model or "gpt-4o", "temperature": 0}
    if llm_timeout is not None:
        kwargs["timeout"] = llm_timeout
    if llm_retries is not None:
        kwargs["max_retries"] = llm_retries
    if base_url:
        try:
            return ChatOpenAI(base_url=base_url, **kwargs)
        except TypeError:
            try:
                return ChatOpenAI(openai_api_base=base_url, **kwargs)
            except TypeError:
                pass
    try:
        return ChatOpenAI(**kwargs)
    except TypeError:
        kwargs.pop("timeout", None)
        kwargs.pop("max_retries", None)
        return ChatOpenAI(**kwargs)


def _decode_xml(upload) -> str:
    data = upload.getvalue()
    if len(data) > 5_000_000:
        return extractor_streaming(io.BytesIO(data))
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="ignore")
    return extractor(text)


def _split_lines(block: str) -> List[str]:
    return [line.strip() for line in (block or "").splitlines() if line.strip()]


def build_structured_intent(
    filters: str,
    joins: str,
    lookups: str,
    aggregations: str,
    calculations: str,
) -> Tuple[Dict[str, List[str]], Optional[str]]:
    structured: Dict[str, List[str]] = {}
    if filters:
        structured["filters"] = _split_lines(filters)
    if joins:
        structured["joins"] = _split_lines(joins)
    if lookups:
        structured["lookups"] = _split_lines(lookups)
    if aggregations:
        structured["aggregations"] = _split_lines(aggregations)
    if calculations:
        structured["calculations"] = _split_lines(calculations)
    if not structured:
        return {}, None
    return structured, json.dumps(structured, indent=2)


def build_context_block(
    norm_ast: str,
    logic_json: str,
    plan_json: str,
    clarify_notes: Optional[str],
    intended_logic: Optional[str],
    structured_json: Optional[str],
    few_shots: Optional[str],
    parameters: Dict[str, str],
) -> str:
    compressed_ast = compress_ast(norm_ast)
    logic_block = shorten_text(logic_json, 4000)
    plan_block = shorten_text(plan_json, 2200)
    clarify_block = shorten_text(clarify_notes or "[none]", 1200)
    structured_block = shorten_text(structured_json or "[none]", 1600)
    few_shot_block = shorten_text(few_shots or "", 1800)
    intended_block = shorten_text(intended_logic or "[none]", 1600)
    parameters_text = "\n".join(f"{k}={v}" for k, v in parameters.items())

    return assemble_context(
        [
            ("AST_JSON", compressed_ast),
            ("DERIVED_LOGIC_JSON", logic_block),
            ("PLAN_JSON", plan_block),
            ("CLARIFICATIONS", clarify_block),
            ("INTENDED_LOGIC", intended_block),
            ("STRUCTURED_INTENT_JSON", structured_block),
            ("FEW_SHOT_EXEMPLARS", few_shot_block),
            ("PARAMETERS", parameters_text),
        ],
        max_chars=7000,
    )


def build_databricks_notebook(
    ast_summary: str,
    intended_logic: str,
    pyspark_code: str,
    sql_ddl: Optional[str],
    sql_dml: Optional[str],
    validator_notes: str,
) -> str:
    sections = [
        "# Databricks notebook source",
        "# COMMAND ----------",
        f"print(\"AST Summary:\\n{ast_summary}\")",
        "# COMMAND ----------",
        pyspark_code or "# PySpark code missing",
        "# COMMAND ----------",
    ]
    sql_parts = []
    if sql_ddl:
        sql_parts.append("-- DDL\n" + sql_ddl)
    if sql_dml:
        sql_parts.append("-- DML\n" + sql_dml)
    sections.append("# MAGIC %sql\n" + ("\n\n".join(sql_parts) if sql_parts else "-- SQL missing"))
    sections.append("# COMMAND ----------")
    sections.append(f"print(\"Validator notes:\\n{validator_notes}\")")
    return "\n\n".join(sections) + "\n"


def fingerprint_payload(payload: Dict[str, str]) -> str:
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


load_dotenv()
load_secrets_into_env()

st.set_page_config(page_title="Infa XML → PySpark & SQL", layout="wide")
st.title("Informatica XML → PySpark & SQL (with Reviewer)")

with st.sidebar:
    st.header("Settings")
    model = os.environ.get("LLM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
    st.caption("Model, API key, and base URL must be configured via environment variables or secrets.")
    databricks_mode = st.checkbox(
        "Databricks mode (generate notebook)",
        value=True,
        help="Adds a Databricks .py notebook output with config and validation cells.",
    )
    optimize_spark = st.checkbox(
        "Optimize Spark (broadcast, prune, repartition)",
        value=True,
        help="Adds optimization guidance to the generation prompt.",
    )
    force_regen = st.checkbox("Force fresh generation", value=False)
    auto_fix_passes = st.slider("Auto-fix iterations", min_value=0, max_value=3, value=1)
    st.subheader("Parameters")
    src_base = st.text_input("Source base path/DSN", value="/mnt/raw")
    tgt_db = st.text_input("Target database", value="analytics")
    tgt_table = st.text_input("Target table", value="emp_payroll")
    write_mode = st.selectbox("Write mode", ["overwrite", "append"], index=0)
    shuffle_parts = st.number_input("Shuffle partitions", min_value=1, max_value=1000, value=16, step=1)
    surrogate_key_col = st.text_input("Surrogate key column", value="emp_sk")
    merge_keys = st.text_input("Merge keys (comma-separated)", value="emp_id")
    partition_columns = st.text_input("Partition columns (comma-separated)", value="dept_id")
    enforce_schema = st.checkbox("Enforce target schema before write", value=True)
    merge_fallback = st.checkbox("Allow overwrite fallback if MERGE fails", value=False)
    key_columns = st.text_input("Not-null key columns (comma-separated)", value="emp_id")
    st.caption("Credentials are read only from environment variables or secrets.")

st.markdown(
    "Upload an Informatica mapping XML. The app extracts a structured AST,"
    " derives logic hints, generates PySpark/SQL, and reviews the output using"
    " validator heuristics plus an optional LLM reviewer."
)

col_left, _ = st.columns([1, 1])
with col_left:
    xml_file = st.file_uploader("Upload Informatica Mapping XML", type=["xml"])
    logic_text = st.text_area(
        "Intended Logic (optional)",
        placeholder="Describe filters, joins, aggregations, target expectations...",
        height=120,
    )
    with st.expander("Structured Intent (optional)"):
        filters_input = st.text_area("Filters", height=80)
        joins_input = st.text_area("Joins / lookups", height=80)
        lookups_input = st.text_area("Lookup defaults", height=80)
        aggregations_input = st.text_area("Aggregations", height=80)
        calculations_input = st.text_area("Calculations", height=80)

run_button = st.button("Generate & Review")


def generate_mapping():
    if not xml_file:
        st.error("Please upload an Informatica XML file.")
        return

    try:
        raw_ast = _decode_xml(xml_file)
    except Exception as exc:
        st.error(f"Failed to parse XML: {exc}")
        return

    _, structured_json = build_structured_intent(
        filters_input,
        joins_input,
        lookups_input,
        aggregations_input,
        calculations_input,
    )

    norm_ast = normalizer(raw_ast)
    logic_json = derive_logic(norm_ast)
    plan_json = build_plan(norm_ast)
    few_shots = load_few_shots(plan_json, folder="few_shots", max_n=2)

    parameters = {
        "SOURCE_BASE": src_base,
        "TARGET_DB": tgt_db,
        "TARGET_TABLE": tgt_table,
        "WRITE_MODE": write_mode,
        "SHUFFLE_PARTITIONS": shuffle_parts,
        "SURROGATE_KEY_COL": surrogate_key_col,
        "MERGE_KEYS": merge_keys,
        "PARTITION_COLUMNS": partition_columns,
        "ENFORCE_SCHEMA": enforce_schema,
        "MERGE_FALLBACK_OVERWRITE": merge_fallback,
        "KEY_COLUMNS_NOT_NULL": key_columns,
    }

    context_block = build_context_block(
        norm_ast,
        logic_json,
        plan_json,
        clarify_notes=None,
        intended_logic=logic_text,
        structured_json=structured_json,
        few_shots=few_shots,
        parameters={k: str(v) for k, v in parameters.items()},
    )

    llm_client = get_llm(api_key, model, base_url)

    pyspark_prompt = f"""{context_block}\n\nYou are an expert Databricks engineer. Generate a PySpark ETL job that faithfully implements the mapping above.\n\nRequirements:\n- Load all sources with spark.read and explicit schemas.\n- Apply filters, joins, lookups, routers, aggregations, and surrogate key logic derived from the AST and structured intent.\n- Respect parameters: SOURCE_BASE, TARGET_DB, TARGET_TABLE, WRITE_MODE, SHUFFLE_PARTITIONS, SURROGATE_KEY_COL, MERGE_KEYS, PARTITION_COLUMNS.\n- Ensure target schema matches the AST targets and enforce schema if requested.\n- Include validation (schema + counts) and optional audit logging hooks.\n- Return the full job in a single fenced `python code block.\n"""
    if optimize_spark:
        pyspark_prompt += "\nImplementation guidance: project only required columns, broadcast small lookups, and repartition to SHUFFLE_PARTITIONS before writes."

    pyspark_response = llm_client.predict(pyspark_prompt)
    pyspark_code, _ = extract_code_blocks(pyspark_response)
    if not pyspark_code:
        pyspark_code = pyspark_response.strip()

    sql_prompt = f"""{context_block}\n\nPySpark reference implementation:\n`python\n{pyspark_code}\n`\n\nGenerate SQL that mirrors the PySpark logic.\nOutput two fenced `sql` blocks: first DDL (CREATE TABLE ... USING DELTA), then DML (MERGE INTO or INSERT).\nInclude partitioning and surrogate key handling consistent with the PySpark code.\n"""

    sql_response = llm_client.predict(sql_prompt)
    sql_ddl, sql_dml = extract_sql_sections(sql_response)
    combined_sql = "\n\n".join([part for part in [sql_ddl, sql_dml] if part])

    validator_text = validator(
        pyspark_code,
        norm_ast,
        sql_code=combined_sql,
        intended_logic=f"{logic_text or ''}\nSTRUCTURED_INTENT={structured_json or '{}'}\nMERGE_KEYS={merge_keys}\nPARTITION_COLUMNS={partition_columns}",
        extra_target_names=[tgt_table.lower(), f"{tgt_db.lower()}.{tgt_table.lower()}"]
    )

    review_prompt = f"""{context_block}\n\nReview the following PySpark and SQL for correctness, readability, and fidelity to the mapping.\nHighlight high-risk gaps and recommend fixes.\n\nPySpark:\n`python\n{pyspark_code}\n`\n\nSQL:\n`sql\n{sql_ddl or '[ddl missing]'}\n`\n`sql\n{sql_dml or '[dml missing]'}\n`\n"""

    review_text = llm_client.predict(review_prompt)

    if auto_fix_passes:
        current_py = pyspark_code
        current_sql_ddl = sql_ddl or ""
        current_sql_dml = sql_dml or ""
        for _ in range(auto_fix_passes):
            fix_prompt = f"""{context_block}\n\nApply the reviewer feedback below to produce corrected PySpark and SQL.\nEnsure minimal, targeted edits.\n\nReviewer feedback:\n{review_text}\n\nCurrent PySpark:\n`python\n{current_py}\n`\n\nCurrent SQL DDL:\n`sql\n{current_sql_ddl or '[missing]'}\n`\n\nCurrent SQL DML:\n`sql\n{current_sql_dml or '[missing]'}\n`\n\nReturn updated PySpark and SQL in fenced code blocks.\n"""
            fix_response = llm_client.predict(fix_prompt)
            new_py, _ = extract_code_blocks(fix_response)
            new_sql_ddl, new_sql_dml = extract_sql_sections(fix_response)
            if new_py:
                current_py = new_py
            if new_sql_ddl:
                current_sql_ddl = new_sql_ddl
            if new_sql_dml:
                current_sql_dml = new_sql_dml
        pyspark_code = current_py
        sql_ddl = current_sql_ddl
        sql_dml = current_sql_dml
        combined_sql = "\n\n".join([part for part in [sql_ddl, sql_dml] if part])
        validator_text = validator(
            pyspark_code,
            norm_ast,
            sql_code=combined_sql,
            intended_logic=f"{logic_text or ''}\nSTRUCTURED_INTENT={structured_json or '{}'}\nMERGE_KEYS={merge_keys}\nPARTITION_COLUMNS={partition_columns}",
            extra_target_names=[tgt_table.lower(), f"{tgt_db.lower()}.{tgt_table.lower()}"]
        )

    tabs = st.tabs(["Generated Code", "AST & Logic", "Review"])
    with tabs[0]:
        st.subheader("PySpark")
        st.code(pyspark_code, language="python")
        st.subheader("SQL DDL")
        st.code(sql_ddl or "[missing]", language="sql")
        st.subheader("SQL DML")
        st.code(sql_dml or "[missing]", language="sql")
        st.info(validator_text)
        if databricks_mode:
            notebook_py = build_databricks_notebook(
                summarize_ast(norm_ast),
                logic_text or "[none]",
                pyspark_code,
                sql_ddl,
                sql_dml,
                validator_text,
            )
            st.download_button(
                "Download Databricks notebook (.py)",
                data=notebook_py,
                file_name="infa_mapping_notebook.py",
                mime="text/x-python",
            )

    with tabs[1]:
        st.subheader("Normalized AST")
        st.code(norm_ast, language="json")
        st.subheader("Derived Logic")
        st.code(logic_json, language="json")
        st.subheader("Plan")
        st.code(plan_json, language="json")
        if structured_json:
            st.subheader("Structured Intent")
            st.code(structured_json, language="json")
        if few_shots:
            st.subheader("Few-shot exemplars")
            st.markdown(f"`\n{few_shots}\n`")

    with tabs[2]:
        st.subheader("Reviewer Output")
        st.markdown(review_text)

    st.success("Generation complete.")


if run_button:
    generate_mapping()
