# Grok (xAI) backend

Genesis talks to xAI through the **OpenAI-compatible Chat Completions API**
(`https://api.x.ai/v1/chat/completions`, key from `XAI_API_KEY`). The client is a small
`requests`-based implementation in `genesis/llm/openai_compat.py`; no vendor SDK is needed.

## Model choice (checked against docs.x.ai on 2026-11-14)

| Role | Model id | Why |
|---|---|---|
| Main agents (`tier: main`) | `grok-4.7`, `reasoning_effort: medium` | xAI's current flagship for agentic/knowledge work ("most capable model we've built"); 500k context; structured outputs and function calling; $2.00 in / $0.50 cached / $6.00 out per 1M tokens. Reasoning cannot be disabled on this family, so `medium` balances quality against latency/cost (default is `high`; use `low` for cheaper runs, `high`/`xhigh` for the hardest scenarios). |
| Light agents (`tier: light`) | `grok-4.20-0309-non-reasoning` | Fast non-reasoning model with strict instruction following and structured outputs; no reasoning tokens, so replies are quick and cheap: $1.25 in / $0.20 cached / $2.50 out per 1M tokens. Alias `grok-4.20-non-reasoning` also works. |

Alternatives: `grok-4.3` is the same price as the light model but does reason (supports
`reasoning_effort: none|low|medium|high|xhigh`, default `low`); it is a good light model if
you want a little thinking. `grok-4.6` / `grok-4.5` sit between `grok-4.3` and `grok-4.7`.

> Model ids, prices and parameter support change. The tiers live in `config.json` under
> `llm.tiers`, so updating them is a config change, not a code change. Pricing is in
> `genesis/llm/pricing.py` and can be overridden with a top-level `pricing` object
> (`{"model-prefix": [input, cached_input, output]}` per 1M tokens).

## Behaviour worth knowing

* **Structured output.** Scenario agents are called with `response_format: json_schema`
  (strict, every property required, nullable fields) built from the scenario's allowed
  actions. xAI guarantees schema conformance for supported keywords, but game constraints
  (amount limits, ownership, ids) are checked in Python, because they depend on state.
  Providers without schema support (Groq) fall back to `json_object`.
* **Repair loop.** An invalid reply is returned to the model with the specific errors
  (`llm.max_repairs`, default 2); after that the agent `pass`es and the step is flagged
  `FALLBACK_PASS`.
* **Retries.** 429, 5xx, timeouts and connection errors retry with exponential backoff plus
  jitter, honouring `Retry-After` (`llm.max_retries`, default 4). 401/403 and 400 errors are
  *not* retried and abort the run, so a bad key or model id cannot masquerade as 150 "pass"
  actions.
* **`reasoning_effort`** is only sent to families that accept it (4.7/4.6/4.5/4.3). On the
  4.5-4.7 family `none` is mapped to `low` because reasoning cannot be turned off there.
  It is dropped for `grok-4.20*`.
* **Tokens and cost.** Usage includes cached and reasoning tokens. xAI reports reasoning
  tokens separately from `completion_tokens`, so billable output is derived from
  `total_tokens - prompt_tokens`. Cost is an *estimate* from the pricing table (requests with
  >= 200k prompt tokens use the 2x long-context rate). `--max-cost` stops a run when the
  estimate reaches the cap.
* **Reasoning field.** The dashboard's "reasoning" is the model's own short, self-reported
  rationale from the JSON reply, not its hidden chain of thought. This matches the repo's
  existing caveat about model-generated explanations.
* **Reasoning models need headroom.** Reasoning tokens count against `max_tokens`
  (`4096` for the main tier). An empty reply with `finish_reason=length` raises a clear error.

## Not verified live

The test suite exercises the HTTP layer with a fake session and the mock provider. It was
written against the current xAI docs but **was not run against the live xAI API** (no key in
the authoring environment). Do a short real run first, e.g.
`python run.py --scenario ram-setu --turns 2 --max-cost 0.25`.
