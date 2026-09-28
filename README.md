# Document Quote Extraction Agent

An agent built with [Pydantic AI](https://ai.pydantic.dev/) that parses documents (e.g. project proposals, bills of materials, specifications, hardware requisitions), identifies all mentioned items, generates **context-aware descriptions** explaining why and how each item is needed, and looks up live pricing and listings on **Amazon**.

---

## Features

- **Context-Aware Extraction**: Rather than just extracting item names, the agent generates contextual descriptions explaining the item's role, requirements, and constraints as stated in the document.
- **Amazon Price Lookup Tool**: Integrates an `AmazonPriceService` tool (`@quote_extraction_agent.tool`) that queries Amazon search for current prices, product titles, ASINs, and direct URLs.
- **Robust Multi-Tier Service**: Handles live HTTP Amazon lookups, detects anti-bot challenges gracefully, and falls back to a curated catalog or dynamic market estimator.
- **Structured Pydantic Models**: Returns validated, typed outputs (`DocumentQuoteOutput`, `ExtractedItem`, `AmazonProductInfo`) with quantity calculation and budget totals.
- **Flexible Provider Support**: Works with OpenAI (`openai:gpt-4o`), Google Gemini (`google-gla:gemini-2.0-flash`), Anthropic Claude (`anthropic:claude-3-5-sonnet-latest`), or Pydantic AI's built-in `TestModel` for offline execution.

---

## Architecture

Following the design pattern of Pydantic AI:

```
+-------------------------------------------------------------------+
|                        Input Document                             |
| (Specifications, Proposals, Bills of Materials, Requisitions)     |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     quote_extraction_agent                        |
|  - System Prompt: Contextual item extraction guidelines           |
|  - Dynamic Instructions (@agent.instructions): Document context   |
+---------------------------------+---------------------------------+
                                  |
                                  | calls tool for each item
                                  v
+-------------------------------------------------------------------+
|               @agent.tool: lookup_amazon_price                    |
|  Calls ctx.deps.amazon_service.lookup_price(search_query)        |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   DocumentQuoteOutput (BaseModel)                 |
|  - document_summary: str                                          |
|  - items: list[ExtractedItem]                                     |
|      - item_name: str                                             |
|      - contextual_description: str                                |
|      - quantity: int                                              |
|      - amazon_search_query: str                                   |
|      - amazon_product_title: str                                  |
|      - unit_price: float                                          |
|      - estimated_total_price: float                               |
|      - product_url: str                                           |
|  - total_estimated_budget: float                                  |
+-------------------------------------------------------------------+
```

---

## Quick Start

### 1. Run the sample demo

To run with Pydantic AI's built-in test model (no API key required):

```bash
uv run python main.py
```

### 2. Run with a live LLM

#### Local with Ollama (Free, Offline)
Ensure Ollama is running (`ollama serve`), then run:

```bash
# Using your local Ollama model (e.g. llama3.1:8b or gemma4:latest):
export MODEL="ollama:llama3.1:8b"
uv run python main.py

# Or configure via a .env file (see .env.example):
# MODEL="ollama:llama3.1:8b"
# OLLAMA_BASE_URL="http://localhost:11434/v1"
```

#### Cloud LLMs (OpenAI, Gemini, Anthropic)
Set your API key for OpenAI, Gemini, or Anthropic:

```bash
# For OpenAI:
export OPENAI_API_KEY="your-api-key"
uv run python main.py

# Or specify any supported model:
export MODEL="openai:gpt-4o"
uv run python main.py
```

### 3. Run unit tests

```bash
uv run python -m unittest test_main.py
```
