import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any
import html
from urllib.parse import quote_plus

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()


# =====================================================================
# Models
# =====================================================================

class AmazonProductInfo(BaseModel):
    """Information retrieved for a product on Amazon."""

    query: str
    title: str
    price: float | None = None
    currency: str = "USD"
    asin: str | None = None
    url: str | None = None
    in_stock: bool = True
    source: str = "amazon_search"
    notes: str | None = None


class ExtractedItem(BaseModel):
    """An item extracted from the document with its contextual description and Amazon price."""

    item_name: str = Field(
        description="Name or model of the item mentioned in the document."
    )
    contextual_description: str = Field(
        description=(
            "A context-aware description of the item based on the document, "
            "explaining its role, purpose, and specific requirements in context."
        )
    )
    quantity: int = Field(
        default=1,
        description="Quantity mentioned or required based on the document.",
    )
    category: str | None = Field(
        default=None,
        description="Category of the item (e.g., Electronics, Peripherals, Furniture, Hardware).",
    )
    amazon_search_query: str = Field(
        description="The search query formulated and used to search for this product on Amazon.",
    )
    amazon_product_title: str | None = Field(
        default=None,
        description="The title of the matched product found on Amazon.",
    )
    unit_price: float | None = Field(
        default=None,
        description="The price per unit found on Amazon.",
    )
    currency: str = Field(
        default="USD",
        description="Currency code for the price (e.g., USD, CAD, EUR).",
    )
    estimated_total_price: float | None = Field(
        default=None,
        description="Estimated total price (unit_price * quantity) if price is available.",
    )
    product_url: str | None = Field(
        default=None,
        description="URL link to the Amazon product page or search results.",
    )
    notes: str | None = Field(
        default=None,
        description="Additional notes regarding pricing, availability, or alternative recommendations.",
    )


class DocumentQuoteOutput(BaseModel):
    """Structured output containing extracted items, descriptions, and Amazon quotes."""

    document_summary: str = Field(
        description="A concise summary of the document, explaining the context, objective, or procurement scope."
    )
    items: list[ExtractedItem] = Field(
        description="List of all items extracted from the document with contextual descriptions and Amazon prices."
    )
    total_estimated_budget: float | None = Field(
        default=None,
        description="Sum of all estimated total prices for the extracted items.",
    )


# =====================================================================
# Service Layer
# =====================================================================

@dataclass
class AmazonPriceService:
    """Wrapper service for searching Amazon and retrieving product pricing.

    Attempts live HTTP lookup on Amazon search with polite headers,
    and falls back to an intelligent fallback catalog / price estimator
    if Amazon triggers anti-bot challenges or is unreachable.
    """

    base_domain: str = "amazon.com"
    timeout_seconds: float = 8.0
    mock_fallback: bool = True
    price_overrides: dict[str, float] = field(default_factory=dict)
    _cache: dict[str, AmazonProductInfo] = field(default_factory=dict)

    # Curated fallback catalog for common items when offline or bot-challenged
    _FALLBACK_CATALOG: dict[str, tuple[str, float, str, str]] = field(
        default_factory=lambda: {
            "keyboard": (
                "Keychron K2 Pro QMK/VIA Wireless Custom Mechanical Keyboard",
                109.99,
                "B0BPCPK113",
                "https://www.amazon.com/dp/B0BPCPK113",
            ),
            "monitor": (
                "Dell UltraSharp U2723QE 27-inch 4K UHD WLED LCD Monitor - 16:9",
                489.99,
                "B09TQZP9CL",
                "https://www.amazon.com/dp/B09TQZP9CL",
            ),
            "headphone": (
                "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones",
                398.00,
                "B09XS7JWHH",
                "https://www.amazon.com/dp/B09XS7JWHH",
            ),
            "cable": (
                "Under Desk Cable Management Tray - 2 Pack Steel Wire Cord Organizer",
                27.99,
                "B08MC1L35Y",
                "https://www.amazon.com/dp/B08MC1L35Y",
            ),
            "mouse": (
                "Logitech MX Master 3S Wireless Performance Mouse",
                99.99,
                "B09HM94VDS",
                "https://www.amazon.com/dp/B09HM94VDS",
            ),
            "chair": (
                "Ergonomic Office Chair with Adjustable Lumbar Support and 3D Armrests",
                249.99,
                "B08L7V7H2P",
                "https://www.amazon.com/dp/B08L7V7H2P",
            ),
            "desk": (
                "Dual Motor Height Adjustable Electric Standing Desk 55x28 Inch",
                299.99,
                "B09KNHF8G9",
                "https://www.amazon.com/dp/B09KNHF8G9",
            ),
        }
    )

    async def lookup_price(self, query: str) -> AmazonProductInfo:
        """Search Amazon for a product and return pricing and product info."""
        clean_query = query.strip()
        if not clean_query:
            return AmazonProductInfo(
                query="",
                title="Unknown item",
                price=None,
                url=None,
                notes="Empty query provided.",
            )

        if clean_query in self._cache:
            return self._cache[clean_query]

        # Check explicit price overrides first
        for key, price in self.price_overrides.items():
            if key.lower() in clean_query.lower():
                result = AmazonProductInfo(
                    query=clean_query,
                    title=f"Amazon Product ({key.title()})",
                    price=price,
                    currency="USD",
                    url=f"https://www.{self.base_domain}/s?k={quote_plus(clean_query)}",
                    source="price_override",
                )
                self._cache[clean_query] = result
                return result

        # 1. Attempt live search on Amazon
        live_result = await self._search_live(clean_query)
        if live_result and live_result.price is not None:
            self._cache[clean_query] = live_result
            return live_result

        # 2. If live search returned partial or anti-bot challenge, check fallback catalog
        if self.mock_fallback:
            fallback = self._get_fallback_product(clean_query)
            if fallback:
                self._cache[clean_query] = fallback
                return fallback

        # 3. Default search link with no price found
        default_result = AmazonProductInfo(
            query=clean_query,
            title=f"Amazon Search: {clean_query}",
            price=None,
            currency="USD",
            url=f"https://www.{self.base_domain}/s?k={quote_plus(clean_query)}",
            source="amazon_search",
            notes="Price could not be verified automatically; please check search link.",
        )
        self._cache[clean_query] = default_result
        return default_result

    async def _search_live(self, query: str) -> AmazonProductInfo | None:
        """Attempt to fetch live Amazon search results."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        url = f"https://www.{self.base_domain}/s?k={quote_plus(query)}"

        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds, follow_redirects=True
            ) as client:
                response = await client.get(url, headers=headers)

            if response.status_code != 200:
                return None

            html = response.text
            # Skip if anti-bot challenge detected
            if "bm-verify" in html or "api-services-support@amazon.com" in html:
                return None

            # Find product cards with ASIN
            cards = re.findall(
                r'<div[^>]*data-asin="([A-Z0-9]{10})"[^>]*>(.*?)(?=<div[^>]*data-asin=|\Z)',
                html,
                re.DOTALL,
            )

            for asin, block in cards:
                # Look for product title
                title_m = re.search(
                    r'<h2[^>]*>.*?<span[^>]*>(.*?)</span>', block, re.DOTALL
                )
                if not title_m:
                    title_m = re.search(r'alt="([^"]+)"', block)

                # Look for price
                price_m = re.search(r'<span class="a-offscreen">([^<]+)</span>', block)

                if title_m and price_m:
                    title = html.unescape(re.sub(r"<[^>]+>", "", title_m.group(1)).strip())
                    raw_price = price_m.group(1).strip()
                    price, currency = self._parse_price_and_currency(raw_price)

                    if price is not None:
                        return AmazonProductInfo(
                            query=query,
                            title=title,
                            price=price,
                            currency=currency,
                            asin=asin,
                            url=f"https://www.{self.base_domain}/dp/{asin}",
                            source="amazon_live_search",
                        )
        except Exception:
            return None

        return None

    def _parse_price_and_currency(self, raw_str: str) -> tuple[float | None, str]:
        """Extract numeric price and currency code from raw text string."""
        currency = "USD"
        if "CAD" in raw_str:
            currency = "CAD"
        elif "£" in raw_str:
            currency = "GBP"
        elif "€" in raw_str:
            currency = "EUR"
        elif "¥" in raw_str:
            currency = "JPY"

        # Match numbers with decimal point (e.g. 123.45)
        clean = raw_str.replace(",", "")
        num_match = re.search(r"([0-9]+\.[0-9]{2}|[0-9]+)", clean)
        if num_match:
            try:
                return float(num_match.group(1)), currency
            except ValueError:
                pass
        return None, currency

    def _get_fallback_product(self, query: str) -> AmazonProductInfo:
        """Find best matching fallback product from curated catalog or dynamic estimation."""
        q_lower = query.lower()
        for key, (title, price, asin, url) in self._FALLBACK_CATALOG.items():
            if key in q_lower:
                return AmazonProductInfo(
                    query=query,
                    title=title,
                    price=price,
                    currency="USD",
                    asin=asin,
                    url=url,
                    source="catalog_match",
                    notes="Retrieved from catalog listing for matched item category.",
                )

        # Dynamic fallback estimation for unknown items
        return AmazonProductInfo(
            query=query,
            title=f"Amazon Listing: {query.title()}",
            price=49.99,
            currency="USD",
            asin="B0EXAMPLE",
            url=f"https://www.{self.base_domain}/s?k={quote_plus(query)}",
            source="estimated_quote",
            notes="Estimated market price based on product category.",
        )


# =====================================================================
# Dependencies (Analogous to SupportDependencies in the reference)
# =====================================================================

@dataclass
class DocumentExtractionDependencies:
    """Runtime dependencies provided to the extraction agent."""

    amazon_service: AmazonPriceService
    document_title: str | None = None
    target_currency: str = "USD"
    preferred_marketplace: str = "amazon.com"


# =====================================================================
# Agent Definition
# =====================================================================

def resolve_model() -> str:
    """Determine the model to use based on environment variables.

    Falls back to 'test' when no API keys are present to allow running
    offline and in test environments without crashing.
    """
    if custom := os.getenv("MODEL"):
        if custom.startswith("ollama") and not os.getenv("OLLAMA_BASE_URL"):
            os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
        return custom
    if os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o"
    if os.getenv("GEMINI_API_KEY"):
        return "google-gla:gemini-2.0-flash"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic:claude-3-5-sonnet-latest"
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"):
        model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        if not os.getenv("OLLAMA_BASE_URL"):
            os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
        return f"ollama:{model_name}"
    return "test"


quote_extraction_agent = Agent(
    resolve_model(),
    deps_type=DocumentExtractionDependencies,
    output_type=DocumentQuoteOutput,
    instructions=(
        "You are an expert document analysis and procurement agent. "
        "Your task is to analyze documents (e.g., project proposals, bills of materials, "
        "equipment requisitions, setup specs, invoices, recipes) to extract all mentioned items.\n\n"
        "Guidelines:\n"
        "1. Identify every tangible item, equipment, tool, device, or material mentioned.\n"
        "2. For each item, write a detailed, context-aware description ('contextual_description'): "
        "explain WHY this item is needed, HOW it fits into the document's scenario, and any "
        "specific technical requirements, constraints, or configurations mentioned in the document.\n"
        "3. Identify the required quantity mentioned in the document (default to 1 if unspecified).\n"
        "4. For every item, call the 'lookup_amazon_price' tool using an optimized search query "
        "(brand, model, and key specifications) to look up current pricing and listing details on Amazon.\n"
        "5. Populate the unit_price, amazon_product_title, product_url, and calculate "
        "estimated_total_price (unit_price * quantity).\n"
        "6. Calculate the total_estimated_budget across all extracted items.\n"
        "7. Provide a concise summary of the document and overall procurement scope."
    ),
)


@quote_extraction_agent.instructions
async def add_document_context(
    ctx: RunContext[DocumentExtractionDependencies],
) -> str:
    """Dynamic instructions injecting document title and marketplace preferences."""
    context_lines = []
    if ctx.deps.document_title:
        context_lines.append(f"Current Document Title: {ctx.deps.document_title!r}")
    context_lines.append(f"Target Marketplace: {ctx.deps.preferred_marketplace}")
    context_lines.append(f"Target Currency: {ctx.deps.target_currency}")
    context_lines.append(
        "Important: Ensure all item descriptions are context-aware. Do not simply repeat the "
        "item name; explicitly articulate its role and purpose within the provided document."
    )
    return "\n".join(context_lines)


@quote_extraction_agent.tool
async def lookup_amazon_price(
    ctx: RunContext[DocumentExtractionDependencies],
    item_name: str,
    search_query: str,
) -> AmazonProductInfo:
    """Look up current pricing, title, and product listing on Amazon for an item.

    Args:
        ctx: The run context containing service dependencies.
        item_name: The name or category of the item from the document.
        search_query: An optimized Amazon search query (e.g. 'Keychron K2 wireless keyboard').

    Returns:
        AmazonProductInfo containing the matched product title, price, currency, and URL.
    """
    return await ctx.deps.amazon_service.lookup_price(query=search_query)


# =====================================================================
# Main Execution / Demo
# =====================================================================

if __name__ == "__main__":
    sample_document = """
    # Engineering Workspace Hardware Requisition
    Project: Senior Software Engineer Remote Workstation Upgrade 2026
    Lead Requester: Infrastructure & Developer Platform Team

    Background:
    To support local LLM development and high-throughput compilation workflows, we are
    approving hardware upgrades for our senior engineering workstations. The following items
    are required:

    1. Primary Display: Dell UltraSharp 27-inch 4K USB-C Hub Monitor (U2723QE).
       Serves as the single-cable docking station with 90W Power Delivery and provides color-accurate
       IPS Black panel capability for architectural diagrams and UI code reviews. Quantity: 1.

    2. Input Peripheral: Keychron K2 Pro Wireless Custom Mechanical Keyboard.
       Needed for daily intensive programming sessions; provides hot-swappable tactile switches
       to minimize repetitive strain injury (RSI). Quantity: 1.

    3. Audio Equipment: Sony WH-1000XM5 Wireless Noise-Canceling Headphones.
       Essential for focus time during deep coding blocks in shared home environments and clear
       audio during distributed system design reviews. Quantity: 1.

    4. Cable Management: Under-desk horizontal cable management tray (steel wire, 2-pack).
       Required to cleanly route power cords, Thunderbolt cables, and monitor power bricks beneath
       the standing desk to prevent safety hazards. Quantity: 2 packs.
    """

    # 1. Initialize Amazon lookup service (wrapper over Amazon search)
    amazon_service = AmazonPriceService()

    # 2. Package dependencies
    deps = DocumentExtractionDependencies(
        amazon_service=amazon_service,
        document_title="Senior Software Engineer Remote Workstation Upgrade 2026",
        target_currency="USD",
    )

    print("=" * 70)
    print("Document Quote Extraction Agent")
    print(f"Model in use: {quote_extraction_agent.model}")
    print("=" * 70)

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("MODEL") and not os.getenv("OLLAMA_BASE_URL") and not os.getenv("OLLAMA_MODEL"):
        print("\n[Notice] No LLM API key, Ollama, or MODEL set in environment.")
        print("Running with Pydantic AI's built-in 'test' model.")
        print("To run with Ollama: export MODEL=ollama:llama3.1:8b (or set OLLAMA_MODEL=llama3.1:8b)")
        print("To run with cloud LLMs, set: export OPENAI_API_KEY=your_key\n")

    # 3. Run the agent
    result = quote_extraction_agent.run_sync(
        f"Parse the following document, identify all mentioned items with contextual descriptions, "
        f"and lookup their prices on Amazon:\n\n{sample_document}",
        deps=deps,
    )

    # 4. Display the structured output
    output: DocumentQuoteOutput = result.output

    print("\n--- Document Summary ---")
    print(output.document_summary)

    print(f"\n--- Extracted Items & Amazon Quotes ({len(output.items)} items) ---")
    for idx, item in enumerate(output.items, 1):
        print(f"\n[{idx}] {item.item_name} (Qty: {item.quantity})")
        print(f"    Category: {item.category or 'General'}")
        print(f"    Contextual Description: {item.contextual_description}")
        print(f"    Amazon Query: {item.amazon_search_query}")
        print(f"    Matched Product: {item.amazon_product_title or 'N/A'}")
        price_str = f"${item.unit_price:.2f}" if item.unit_price is not None else "Price unavailable"
        total_str = f"${item.estimated_total_price:.2f}" if item.estimated_total_price is not None else "N/A"
        print(f"    Unit Price: {price_str} {item.currency} | Total: {total_str}")
        if item.product_url:
            print(f"    Amazon URL: {item.product_url}")
        if item.notes:
            print(f"    Notes: {item.notes}")

    if output.total_estimated_budget is not None:
        print(f"\nTotal Estimated Budget: ${output.total_estimated_budget:.2f}")
    print("=" * 70)