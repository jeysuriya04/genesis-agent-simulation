# Genesis: Multi-model artificial society

A Python research prototype for a five-agent artificial society. Groq-hosted models propose decisions; the simulator validates and executes them, maintains shared world state, and records the complete event history in SQLite.

For the complete end-to-end execution flow, data lifecycle, showcase guide, and
the latest completed 10-day experiment summary, see
[docs/GENESIS-FLOW.md](./docs/GENESIS-FLOW.md).

## Important limitations

- This is a toy experiment, not a validated model of human morality or development.
- A flag such as `UNAUTHORIZED_TAKING` is based on explicit simulation rules. It is not a universal moral verdict.
- Model-generated `reason` text is a generated explanation, not verified access to internal reasoning.
- LLM output is stochastic. For reproducibility, record model IDs, prompts, parameters and configuration.
- This is an experimental software system, not a validated model of human behavior, morality, consciousness, or psychology.
- Behavior dimensions and rule flags are simulation-specific measurements, not universal moral judgments.
- API latency, rate limits, retries, model availability, and stochastic outputs affect the observed run. Preserve run metadata and repeat experiments before drawing conclusions.

## Requirements

- macOS, Linux or Windows
- Python 3.10+
- A Groq account and API key
- Internet access for the Groq API and Python packages

## 1. Set up the Python project

```bash
cd genesis_ai_society
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The app loads `GROQ_API_KEY` from `../RAG/.env` (relative to this project). Keep your key there, or set it in your shell:

```bash
export GROQ_API_KEY="your_api_key_here"
```

If you still need a key, create one at https://console.groq.com/keys. Do not put it in `config.json` or commit it to source control. The simulation checks the configured model IDs against the models available to your API key before it starts.

The current configuration uses three models across five agents: `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, and `qwen/qwen3.8-27b`. The first two are listed as production models; Qwen3.8-27B is listed as a preview model and may change or be discontinued. Check the [Groq supported models catalogue](https://console.groq.com/docs/models) before changing assignments. Availability may vary by account.

## 2. Run the simulation

The clock is virtual: each loop advances one simulated day, applies the scheduled scenario and daily resource accounting, gathers one decision from each of the five agents against the same start-of-day snapshot, then validates and resolves decisions in a seeded order. There are no waits for real days to pass. A fresh 30-day run makes up to 150 Groq completion requests in concurrent daily batches (default concurrency: five).

The five-minute target is not guaranteed: with five concurrent requests, 30 days require 30 daily batches, so each batch would need to finish in about ten seconds on average. Actual runtime depends on Groq latency, model choice, retries, and account rate limits; the CLI and showcase report measured runtime.

Change the concurrency (and reproducibility seed) as needed:

```bash
python run.py --reset --days 30 --concurrency 5 --seed 7
```

Scenarios are configured in `scenarios.json`; schedule, wording, verifiable facts, and supported supply effects can be edited without changing simulation code. `--reset` starts a new experiment run and preserves earlier run/event history in SQLite. A fixed seed controls the action-resolution and resource-allocation order.

Start a fresh 5-agent, 30-day experiment:

```bash
python run.py --reset --days 30
```

To continue from the latest saved snapshot:

```bash
python run.py --days 30
```

The program records each agent decision, daily environment update, and post-action state in `genesis.db`. Schema upgrades are additive; existing events and snapshots are retained.

## 3. Open the AgentVerse-style replay showcase

In another terminal, from the project directory with the virtual environment activated:

```bash
python showcase.py
```

Open http://localhost:7860. The Gradio interface replays Genesis runs stored in
SQLite; it does not initiate model requests. It provides run/day/resident
selection, a decision timeline, resource and behavioral plots, a directed
relationship graph, incident details, explicit propagation links, and
human-review notes.

The showcase uses Gradio, the UI toolkit used by OpenBMB AgentVerse's local
GUI, while preserving Genesis as the simulation and data source. It does not
run AgentVerse's separate simulation engine: AgentVerse's stock GUI starts
AgentVerse task environments rather than loading Genesis experiment records.
See the [OpenBMB AgentVerse repository](https://github.com/OpenBMB/AgentVerse)
for the framework itself.

## 4. Change agents/models

Edit `config.json` and use the exact model IDs from the Groq catalogue. Agents can share models; the current catalogue does not require five distinct model families.

For a fair model comparison, keep roles, starting resources, prompts and world conditions identical across repeated runs. Swap model assignments between roles in separate experiments.

## 5. Replay showcase

The Gradio showcase provides run/day/resident filters, progress and measured
runtime, daily action and resource charts, chronological event replay, agent
behavior traces, a directed relationship graph, explicit imitation/response
links, incident inspection, human-review notes, and the persistent event ledger.

The simulator records daily supplies, consumption, unmet needs, and health changes as world events. It includes configurable scarcity, rumor, peer-pressure, medicine, loyalty, accountability, governance, and reconciliation scenarios. Resource transfers are validated; trade requires an offer and recipient acceptance. Rule proposals require a majority vote before they become public rules.

## 6. What gets logged?

The `events` table records:
- real timestamp, virtual day/minute, and agent (or `WORLD`)
- configured model ID, returned model version, latency, parameters, and API error
- model context, raw response, validated decision, prompt/config hashes
- world state before action and objective consequences, including beneficiaries, harmed residents, and witnesses
- rule-based flags, scenario ID, behavior-dimension observations, related incidents, and human-review status/notes

The `snapshots` table records end-of-day world states. The `runs` table stores seed, configuration, scenario/world-rule versions, status, and runtime; `propagations` stores explicit links to remembered events.

## 7. World mechanics and research interpretation

- Agents have independent food, medicine, health, memories, and directed relationship measures.
- `help` and `share` transfer available resources; `take` is an unauthorized transfer and creates a restitution obligation.
- Trade uses proposal, recipient acceptance/rejection, and a two-sided transfer; balances are checked again at execution.
- Checkable claims are compared to an independent fact ledger. A `lie` action records declared deceptive intent, not an independently verified falsehood.
- Apologies do not compel forgiveness. Recipients choose whether to forgive; restitution is separately tracked.
- Governance proposals remain pending until majority support or rejection.
- Behavior measures are distinct, bounded, and rule-coded. The model's `reason` is a generated explanation, not evidence of private motives.

## 8. Reproducible experiments and limitations

Use separate fresh runs to compare model-to-role assignments while keeping scenario files, prompts, resource settings, and seeds controlled. Set `seed`, `model_parameters`, `world_rules_version`, and each agent's model independently in `config.json`. Model availability and account access can change; the application checks configured IDs against the live Groq catalogue before a real run.

The five-minute target is a performance goal, not a guarantee. A 30-day simulation makes up to 150 model requests; the automated tests use a local decision provider and do not incur Groq requests.

## Troubleshooting

- `GROQ_API_KEY is not set`: export your key in the terminal running the simulation.
- Configured model unavailable: check the current Groq catalogue and your account's model access, then update `config.json`.
- Invalid model response: invalid actions are safely converted to `observe` and flagged; API failures are explicitly recorded and the run completes with an error status. Inspect the event ledger.
- Start over: `python run.py --reset --days 30`.
