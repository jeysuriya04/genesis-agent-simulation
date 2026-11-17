import json
import unittest

from genesis.llm import (
    LLMAuthError, LLMRateLimitError, LLMRequest, MockProvider, ProviderPool, Usage,
    UsageTracker, estimate_cost, extract_json_object, request_structured,
)
from genesis.llm.openai_compat import GroqProvider, XAIProvider


class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body


def ok_body(content='{"action":"pass"}', model="grok-4.7", usage=None):
    return {
        "model": model,
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or {
            "prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1600,
            "completion_tokens_details": {"reasoning_tokens": 500},
            "prompt_tokens_details": {"cached_tokens": 200},
        },
    }


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


def make_provider(responses, **kwargs):
    sleeps = []
    provider = XAIProvider(
        api_key="test-key", session=FakeSession(responses), sleep=sleeps.append,
        backoff_base=1.0, **kwargs,
    )
    return provider, sleeps


def request(**kwargs):
    base = dict(model="grok-4.7", messages=[{"role": "user", "content": "hi"}])
    base.update(kwargs)
    return LLMRequest(**base)


class XAIProviderTests(unittest.TestCase):
    def test_payload_uses_xai_endpoint_schema_and_reasoning_effort(self):
        provider, _ = make_provider([FakeResponse(200, ok_body())])
        schema = {"name": "action", "schema": {"type": "object", "properties": {}}}
        response = provider.complete(request(json_schema=schema, reasoning_effort="medium"))
        call = provider.session.calls[0]
        self.assertEqual(call["url"], "https://api.x.ai/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(call["json"]["response_format"]["type"], "json_schema")
        self.assertTrue(call["json"]["response_format"]["json_schema"]["strict"])
        self.assertEqual(call["json"]["reasoning_effort"], "medium")
        self.assertEqual(response.content, '{"action":"pass"}')

    def test_reasoning_effort_is_dropped_for_models_that_reject_it(self):
        provider, _ = make_provider([FakeResponse(200, ok_body(model="grok-4.20-0309-non-reasoning"))])
        provider.complete(request(model="grok-4.20-0309-non-reasoning", reasoning_effort="low"))
        self.assertNotIn("reasoning_effort", provider.session.calls[0]["json"])

    def test_groq_falls_back_to_json_object_mode(self):
        provider = GroqProvider(api_key="k", session=FakeSession([FakeResponse(200, ok_body())]))
        provider.complete(request(model="openai/gpt-oss-20b",
                                  json_schema={"name": "a", "schema": {}}))
        self.assertEqual(provider.session.calls[0]["json"]["response_format"],
                         {"type": "json_object"})

    def test_rate_limit_backoff_honours_retry_after_then_succeeds(self):
        provider, sleeps = make_provider([
            FakeResponse(429, {"error": {"message": "slow down"}}, {"retry-after": "3"}),
            FakeResponse(503, {"error": "overloaded"}),
            FakeResponse(200, ok_body()),
        ])
        response = provider.complete(request())
        self.assertEqual(response.attempts, 3)
        self.assertEqual(sleeps[0], 3.0)
        self.assertGreaterEqual(sleeps[1], 2.0)  # exponential: base * 2**1 (+ jitter)

    def test_retries_are_bounded_and_auth_errors_are_not_retried(self):
        provider, sleeps = make_provider([FakeResponse(429, {})] * 3, max_retries=2)
        with self.assertRaises(LLMRateLimitError):
            provider.complete(request())
        self.assertEqual(len(sleeps), 2)
        provider, sleeps = make_provider([FakeResponse(401, {"error": "bad key"})])
        with self.assertRaises(LLMAuthError):
            provider.complete(request())
        self.assertEqual(sleeps, [])

    def test_missing_api_key_message_names_the_env_var(self):
        import os
        old = os.environ.pop("XAI_API_KEY", None)
        try:
            with self.assertRaises(LLMAuthError) as ctx:
                XAIProvider(session=FakeSession([])).complete(request())
            self.assertIn("XAI_API_KEY", str(ctx.exception))
        finally:
            if old is not None:
                os.environ["XAI_API_KEY"] = old

    def test_usage_counts_reasoning_and_cost_uses_cached_rate(self):
        provider, _ = make_provider([FakeResponse(200, ok_body())])
        response = provider.complete(request())
        self.assertEqual(response.usage.reasoning_tokens, 500)
        self.assertEqual(response.usage.output_tokens, 600)  # 100 completion + 500 reasoning
        expected = (800 * 2.00 + 200 * 0.50 + 600 * 6.00) / 1_000_000
        self.assertAlmostEqual(response.cost_usd, expected)


class PricingTests(unittest.TestCase):
    def test_known_unknown_and_overridden_models(self):
        usage = Usage(prompt_tokens=100_000, completion_tokens=100_000, total_tokens=200_000)
        self.assertAlmostEqual(estimate_cost("grok-4.3", usage), (1.25 + 2.50) / 10)
        self.assertAlmostEqual(estimate_cost("grok-4.20-0309-non-reasoning", usage), 0.375)
        self.assertIsNone(estimate_cost("mystery-model", usage))
        self.assertAlmostEqual(estimate_cost("mystery-model", usage, {"mystery": [1, 1, 1]}), 0.2)
        long = Usage(prompt_tokens=250_000, completion_tokens=0, total_tokens=250_000)
        self.assertAlmostEqual(estimate_cost("grok-4.7", long), 250_000 * 4.0 / 1e6)  # 2x long-context

    def test_tracker_aggregates_and_enforces_budget(self):
        tracker = UsageTracker(max_cost_usd=0.01)
        tracker.record("A1", "grok-4.7", Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15), 0.004)
        self.assertFalse(tracker.over_budget())
        tracker.record("A2", "grok-4.7", Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15), 0.007, attempts=3)
        self.assertTrue(tracker.over_budget())
        summary = tracker.summary()
        self.assertEqual(summary["calls"], 2)
        self.assertEqual(summary["retries"], 2)
        self.assertAlmostEqual(summary["cost_usd"], 0.011)
        self.assertEqual(summary["by_model"]["grok-4.7"]["calls"], 2)


class StructuredTests(unittest.TestCase):
    def parse(self, raw):
        try:
            value = extract_json_object(raw)
        except json.JSONDecodeError:
            return None, ["not JSON"]
        return value, ([] if "action" in value else ["missing action"])

    def test_extract_json_tolerates_fences_and_prose(self):
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json_object('Sure! {"a": 2} done'), {"a": 2})

    def test_invalid_reply_is_repaired_with_error_feedback(self):
        mock = MockProvider(script=["oops", '{"x": 1}', '{"action": "pass"}'])
        result = request_structured(mock, request(), self.parse, max_repairs=2)
        self.assertTrue(result.ok)
        self.assertEqual(result.repairs, 2)
        self.assertIn("rejected", mock.requests[-1].messages[-1]["content"])

    def test_gives_up_after_max_repairs(self):
        mock = MockProvider(script=["a", "b"])
        result = request_structured(mock, request(), self.parse, max_repairs=1)
        self.assertFalse(result.ok)
        self.assertEqual(result.calls, 2)


class PoolTests(unittest.TestCase):
    def test_tiers_overrides_and_mock(self):
        pool = ProviderPool()
        main = pool.resolve({"id": "a", "tier": "main"})
        light = pool.resolve({"id": "b", "tier": "light"})
        self.assertEqual(main["model"], "grok-4.7")
        self.assertEqual(light["model"], "grok-4.20-0309-non-reasoning")
        custom = pool.resolve({"id": "c", "llm": {"model": "grok-4.3", "reasoning_effort": "low"}})
        self.assertEqual((custom["model"], custom["params"]["reasoning_effort"]), ("grok-4.3", "low"))
        mock_pool = ProviderPool({"provider": "mock"})
        self.assertTrue(mock_pool.resolve({"id": "d"})["model"].startswith("mock"))
        self.assertIsInstance(mock_pool.get(), MockProvider)
        with self.assertRaises(ValueError):
            pool.resolve({"id": "e", "tier": "nope"})
        with self.assertRaises(ValueError):
            pool.get("nonsense")


if __name__ == "__main__":
    unittest.main()
