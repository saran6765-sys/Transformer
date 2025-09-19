# Informatica XML to PySpark/SQL Migration Toolkit

## Overview
This project converts Informatica mapping XML files into production-ready PySpark and SQL assets that can run on Databricks. A Streamlit application orchestrates XML extraction, logic derivation, LLM-based code generation, automated reviews, and rule-based validation so that migration teams can iterate quickly while keeping transformations auditable.

## Key Capabilities
- Parse Informatica mapping XML (streaming for large files) and build a normalized AST with sources, targets, transformations, and connectors.
- Derive filters, joins, lookups, routers, aggregations, and calculations from the AST to ground LLM prompts.
- Generate PySpark ETL jobs plus matching SQL DDL/DML, optimized for Databricks delta tables and parameter driven deployments.
- Run an LLM "reviewer" and rule-based validator to highlight gaps, enforce schemas, and guide optional auto-fix passes.
- Export a Databricks `.py` notebook that bundles validation cells, PySpark code, and SQL blocks for downstream execution.
- Connect to OpenAI-hosted models or any OpenAI-compatible gateway (custom headers, deployments, API versions, and retry settings supported).

## Workflow Architecture
- `streamlit_app.py` loads configuration, hosts the Streamlit UI, and coordinates XML ingestion, prompting, validation, review, and notebook export.
- `src/infa_to_pyspark/agents.py` implements XML extraction (buffered and streaming), normalization, logic derivation, review planning, validation, and code block parsing.
- `src/infa_to_pyspark/transform_framework.py` provides reusable PySpark helpers for lookups, surrogate keys, audit logging, Delta merge patterns, and validation hooks.
- `src/infa_to_pyspark/prompt_utils.py` trims large payloads and assembles structured prompt context to keep LLM calls stable.
- `src/infa_to_pyspark/direct_llm.py` implements a lightweight HTTP client for OpenAI-compatible endpoints when LangChain's client is not desired.
- `conf/secrets.example.json` outlines the fields that can be promoted into `conf/secrets.json` for local development without committing secrets.

## Prerequisites
- Python 3.10 or later.
- pip (bundled with Python) and the ability to create virtual environments.
- An OpenAI API key or access to an OpenAI-compatible gateway endpoint.
- (Optional) Databricks workspace for running the generated notebooks.

## Installation
1. Clone or download the repository.
2. Create and activate a virtual environment.
   - Windows PowerShell:
     - `py -m venv venv`
     - `./venv/Scripts/Activate.ps1`
     - If activation is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   - macOS/Linux:
     - `python3 -m venv venv`
     - `source venv/bin/activate`
3. Install dependencies:
   - Upgrade pip: `python -m pip install --upgrade pip`
   - Install requirements: `python -m pip install -r requirements.txt`

## Configuration
You can combine environment variables, a `.env` file, and `conf/secrets.json` to configure credentials and defaults. Environment variables always take precedence.

### Core Variables
- `OPENAI_API_KEY` or `LLM_API_KEY`: credential used by Streamlit when talking to the model provider.
- `LLM_MODEL` or `OPENAI_MODEL`: default model name (for example `gpt-4o-mini`).
- `LLM_BASE_URL` or `OPENAI_BASE_URL`: override endpoint for OpenAI-compatible gateways.
- `LLM_PROVIDER`: set to `direct`, `gateway`, or `openai`; `direct` activates the custom HTTP client.
- `LLM_DEPLOYMENT`: Azure/OpenAI deployment name when required.
- `LLM_KEY_HEADER` and `LLM_KEY_PREFIX`: customise HTTP header name and prefix (for example `X-Api-Key`).
- `LLM_API_VERSION`: append an `api-version` query parameter when targeting Azure OpenAI or similar services.
- `LLM_EXTRA_HEADERS`: JSON object of additional headers serialized as a string.
- `LLM_TEMPERATURE`, `LLM_TIMEOUT`, `LLM_MAX_RETRIES`: control model creativity, request timeout, and retry behaviour.

### Secrets File
1. Copy `conf/secrets.example.json` to `conf/secrets.json` (this file is git-ignored).
2. Populate the JSON fields with your API key, base URL, model, and optional deployment details.
3. Run the app; `secrets_loader.py` loads the values into environment variables if they are not already set.

### `.env` Support
The project uses `python-dotenv`, so placing key/value pairs in a `.env` file at the repository root will populate environment variables at startup. Do not commit real secrets.

## Running the Streamlit App
Run the application from the project root:

```
python -m streamlit run streamlit_app.py
```

Using the module form for `streamlit` prevents Windows launcher issues and respects the active virtual environment.

## Using the Application
1. Upload an Informatica mapping XML file. Large files automatically switch to streaming parsing to reduce memory usage.
2. (Optional) Provide "Intended Logic" notes or populate the structured intent fields (filters, joins, lookups, aggregations, calculations) to guide prompt grounding.
3. Adjust sidebar options:
   - Toggle Databricks notebook export.
   - Control Spark optimisations (projection, broadcast hints, repartition) and the number of auto-fix iterations.
   - Review which API key, model, and base URL are active.
4. Click **Generate & Review**. The pipeline will:
   - Extract and normalise the mapping AST.
   - Derive logic hints and assemble prompt context.
   - Generate PySpark and SQL code blocks.
   - Run the rule-based validator and LLM reviewer, then optionally loop through auto-fix passes.
5. Inspect the tabs:
   - **Generated Code**: PySpark, SQL DDL, SQL DML, validator notes, and Databricks notebook download.
   - **AST & Logic**: Normalised AST, derived logic JSON, execution plan, structured intent, and optional few-shot exemplars.
   - **Review**: LLM reviewer findings and recommendations.

## Few-Shot Exemplars
Place curated prompt-completion pairs in the `few_shots` directory and point the app to them when needed. These samples help stabilise the generated code for domain-specific transformations.

## Validation and Auto-Fix Loop
The rule-based validator checks schema coverage, target handling, merge behaviour, surrogate keys, and other best practices. Reviewer feedback is fed into an auto-fix loop (0-3 passes) that attempts targeted corrections while re-running validation between passes.

## Troubleshooting
- **"Fatal error in launcher" on Windows**: Always invoke tools with `python -m streamlit ...` and `python -m pip ...`. Recreate the virtual environment if the shim remains broken.
- **Cannot activate PowerShell virtual environment**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once per shell session.
- **Model capacity or 429 errors**: The UI falls back to `gpt-4o-mini` automatically when possible. Otherwise, retry later or choose a lighter model.
- **Validation keeps failing**: Review the structured intent and intended logic sections for required joins or filters that may be missing from the XML and re-run generation.

## Repository Layout
- `streamlit_app.py` – Streamlit entry point and orchestration logic.
- `src/infa_to_pyspark/` – Core modules for extraction, prompting, validation, and helpers.
- `conf/` – Example secrets file; copy to `conf/secrets.json` for local use.
- `docs/` – Supplemental documentation assets.
- `scripts/` – Utility scripts for experimentation or automation.
- `few_shots/` – Optional prompt exemplars that can be loaded during generation.
- `requirements.txt` – Python dependencies.

## Development Notes
- There are no automated tests yet; add unit tests around `src/infa_to_pyspark` modules before extending critical logic.
- When integrating with additional LLM providers, use `DirectChatLLM` as a template for custom authentication or request flows.
- Avoid committing `.env`, `conf/secrets.json`, or any files containing real credentials.
