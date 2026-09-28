import unittest
import asyncio
from main import (
    AmazonPriceService,
    AmazonProductInfo,
    DocumentExtractionDependencies,
    DocumentQuoteOutput,
    ExtractedItem,
    quote_extraction_agent,
)
from pydantic_ai.models.test import TestModel


class TestDocumentQuoteAgent(unittest.TestCase):
    def setUp(self):
        self.service = AmazonPriceService(
            price_overrides={
                "custom keyboard": 125.00,
                "dell monitor": 450.00,
            }
        )
        self.deps = DocumentExtractionDependencies(
            amazon_service=self.service,
            document_title="Test Specification Document",
            target_currency="USD",
        )

    def test_amazon_price_override(self):
        async def _test():
            info = await self.service.lookup_price("Custom Keyboard Wireless")
            self.assertEqual(info.price, 125.00)
            self.assertEqual(info.currency, "USD")
            self.assertEqual(info.source, "price_override")
        asyncio.run(_test())

    def test_amazon_fallback_catalog(self):
        async def _test():
            info = await self.service.lookup_price("Keychron K2 wireless keyboard")
            self.assertIsNotNone(info.price)
            self.assertIn("amazon.com", info.url)
        asyncio.run(_test())

    def test_extracted_item_validation(self):
        item = ExtractedItem(
            item_name="Dell UltraSharp 27 Monitor",
            contextual_description="Primary high-resolution monitor specified for UI design and programming reviews.",
            quantity=2,
            category="Peripherals",
            amazon_search_query="Dell UltraSharp 27 4K",
            amazon_product_title="Dell UltraSharp U2723QE 27-inch 4K",
            unit_price=489.99,
            currency="USD",
            estimated_total_price=979.98,
            product_url="https://www.amazon.com/dp/B09TQZP9CL",
        )
        self.assertEqual(item.quantity, 2)
        self.assertEqual(item.estimated_total_price, 979.98)
        self.assertIn("Primary high-resolution monitor", item.contextual_description)

    def test_agent_run_sync_with_test_model(self):
        prompt = "Parse document: We need 1 Dell UltraSharp 4K monitor and 1 Keychron keyboard."
        result = quote_extraction_agent.run_sync(
            prompt,
            deps=self.deps,
            model=TestModel(),
        )
        self.assertIsInstance(result.output, DocumentQuoteOutput)
        self.assertTrue(len(result.output.items) >= 1)


if __name__ == "__main__":
    unittest.main()
