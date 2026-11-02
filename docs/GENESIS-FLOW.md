# GENESIS: Simulation Flow and Research Guide

## 1. Purpose and scope

GENESIS is a Python prototype for observing decisions and social dynamics in a
small artificial society. Five configured agents use Groq-hosted language
models to propose actions. The simulation engine—not the language model—checks
those proposals, applies the world rules, updates resources and relationships,
and stores the resulting evidence in SQLite. The replay showcase reads that
database to provide an experiment observatory.

The virtual clock is independent of wall-clock time: one virtual day is one
complete cycle for all five residents. A 10-day run therefore asks for up to
50 resident decisions. API latency and rate limits determine how long the run
actually takes.

This software is an experimental environment, not a validated model of people,
morality, consciousness, or psychology. Its behavior dimensions and rule flags
are operational measurements defined by the simulation. Model-generated
explanations are not verified accounts of internal reasoning.

## 2. System at a glance

```text
run.py
  └─ load config and scenario schedule
      └─ create a new run or resume the latest snapshot
          └─ for each virtual day
              ├─ apply scheduled scenario effects
              ├─ supply resources, consume needs, update health
              ├─ record a WORLD event
              ├─ freeze a shared start-of-decision snapshot
              ├─ request five Groq decisions concurrently
              ├─ validate and resolve them in seeded order
              ├─ record each decision and its consequences
              └─ save an end-of-day snapshot
                  └─ showcase.py reads the run, events, and snapshots
```

Core files:

| File | Responsibility |
|---|---|
| `run.py` | Command-line options, progress output, and simulation invocation |
| `config.json` | Agent identities, model assignments, inference settings, and world resources |
| `scenarios.json` | Versioned day-by-day scenario descriptions, facts, and supported supply effects |
| `genesis/simulation.py` | World state, Groq requests, validation, action resolution, and the virtual-day loop |
| `genesis/scenarios.py` | Scenario schedule validation and effect application |
| `genesis/storage.py` | Additive SQLite schema migration and database connection |
| `genesis.db` | Persistent experiment runs, events, snapshots, and propagation links |
| `showcase.py` | Gradio replay interface for saved Genesis experiments |
| `tests/test_simulation.py` | Local tests for simulation rules, persistence, and a mocked 30-day run |

## 3. Starting an experiment

From the project directory:

```bash
cd /Users/user/Documents/dev-work/genesis_ai_society
source .venv/bin/activate
python run.py --reset --days 10 --seed 7 --concurrency 5
```

The command-line arguments are:

- `--days`: the target virtual day. The current example uses 10 days.
- `--reset`: create a new experiment run. This does **not** erase earlier runs.
- `--seed`: controls seeded resource-allocation and decision-resolution order.
- `--concurrency`: maximum simultaneous model requests, capped at the five
  configured residents.

The run records a UUID, target length, seed, agent configuration, scenario
definition, scenario version, and world-rules version. New runs append to the
same database. Without `--reset`, the engine continues from the latest saved
snapshot; use a fresh run when comparing experiments.

The Groq API key is loaded from `../RAG/.env` relative to the project, or from
the `GROQ_API_KEY` environment variable. The key is never part of `config.json`
or this document. Before a live run, configured model IDs are checked against
the account's available models.

## 4. What happens during one virtual day

### 4.1 Apply the scenario

The engine looks up the scenario scheduled for the current day in
`scenarios.json`. A scenario can alter the shared community store or the day's
per-resident supply, and can add scenario-specific facts to the truth ledger
available to the agents. The scenario is recorded in the day's world event.

Only fields supported by the scenario validator can be used as automatic
effects. Scenario wording and facts provide context; they do not prescribe how
an agent must respond.

### 4.2 Apply supply, consumption, and health effects

Each living resident receives the configured daily food and medicine supply.
Residents consume from their own inventory first and then, if needed, from the
community store. A remaining unmet need is recorded, and health is updated
according to the current world rules. Consumption order is shuffled
deterministically from the experiment seed and day, making competition for
limited shared resources reproducible.

This stage is persisted as an event whose agent is `WORLD`. Its details include
the previous and resulting world state, scenario, per-resident supply and
consumption, unmet needs, health, and beneficiaries or harms. The world update
is not an LLM decision.

### 4.3 Freeze the decision snapshot and build agent contexts

After the environment update, the engine copies the world state. It builds one
context per resident from that same snapshot. Context includes the resident's
identity, resources, health, behavioral dimensions, directed relationships,
other residents' visible state, active scenario, open trade offers, open
apologies and rule proposals, recent public events, bounded personal memories,
previous-day action counts, checkable facts, and active public rules.

This design means all five model requests for a day start from the same
pre-decision world. A resident does not see another resident's action from the
same decision batch before choosing its own action.

### 4.4 Ask the configured Groq models concurrently

Each resident is sent a system instruction and a JSON user context to its
configured Groq model. Model assignments are independent: the current
configuration uses three distinct model IDs across five residents, rather than
five distinct model families. Per-model parameters can override global
parameters in `config.json`.

The engine requests a JSON object and records the model response, prompt
messages and hash, returned model version when available, inference parameters,
and request latency. Retryable rate-limit, server, and connection failures use
bounded retries. If a request ultimately fails, that resident receives a
flagged `observe` fallback, the error is stored on the event, and the run can
continue with an error status.

The request budget is up to five calls per virtual day. Concurrency reduces
elapsed time in a batch, but it does not eliminate per-model/account rate
limits or guarantee a particular total runtime.

### 4.5 Validate and resolve decisions

Once every request in the day has settled, the engine deterministically shuffles
the five residents using the experiment seed and day, then resolves their
validated decisions one at a time. This protects the shared resource balances:
an earlier action can affect whether a later action can still execute. Each
decision's context remains the original day-start context.

Validation checks the action name, target, quantities, resource names,
remembered event IDs, open trade offers, apology eligibility, checkable claim
keys, and open governance proposals. Invalid requests are converted to a safe
observation and flagged rather than applied as requested. The engine checks
current balances again at execution time.

Action mechanics include:

- **Help/share:** transfer the requested available resource to another resident.
- **Take:** transfer available resources without consent, record a
  simulation-rule flag, create a restitution obligation, and affect
  relationships.
- **Shared store:** request an allocation for oneself or distribute resources
  to a specified resident; available community balances are enforced.
- **Trade:** propose an offer without moving anything. The named recipient can
  accept or reject an open offer. Acceptance executes both sides only if both
  parties still have the agreed resources.
- **Claims:** compare an eligible claim against the independent, context-visible
  fact value; contradicted claims are flagged. The `lie` action alone is a
  declaration of intent, not an independently verified falsehood.
- **Accusation:** record the accusation and its relationship effects; it does
  not establish that the accusation is true.
- **Apology/forgiveness:** apologies are recorded, but only the recipient can
  forgive an open apology. Restitution is a separate resource transfer.
- **Governance:** a proposal remains open until a majority vote enacts or
  rejects it; an enacted proposal becomes an active public rule.
- **Hiring scenario:** during the scheduled case, an agent can recommend one
  listed candidate or report the officer's conduct concern. Recommendations
  are not appointments, and reports are not verified findings; the simulation
  requires independent review and does not automatically hire anyone.
- **Observe:** records an observation without a material transfer.

For each action the engine records the input context, raw response, validated
action, before/after state, objective outcome, beneficiaries, harmed parties,
public message, model-generated explanation, rule flags, witnesses, and any
event the agent explicitly referenced.

### 4.6 Update memory, relationships, and behavior observations

Events are made available as memories to the actor and witnesses. Relationship
updates are directed: for example, sharing can improve trust in both
directions, while unauthorized taking can reduce trust and increase conflict.
At the next virtual day's prompt, agents receive bounded recent memories and
previous-day counts of other residents' actions.

When a model explicitly refers to an event ID in its remembered context, the
engine stores a propagation link. These links support investigation of
referenced imitation or responses; they are not a claim that every witnessed
action caused a later behavior.

Behavior dimensions are separate bounded measures, updated by transparent
action rules—for example cooperation, self-interest, honesty, aggression,
empathy, loyalty, conformity, accountability, repair, trustworthiness, and
reciprocity. They are not combined into a single morality score, and should
not be interpreted as verified motives.

### 4.7 Persist progress and advance the clock

Each world update and resident decision is committed to SQLite. After all five
decisions are resolved, the engine advances the virtual day and records an
end-of-day snapshot. It does not begin the next day while the current batch's
model requests are still pending.

The CLI reports completed decisions, virtual day, request latency, outcomes,
errors, and elapsed runtime. The `runs` record is updated with status and
runtime; a normally completed run is marked `completed`, while a run with
recorded API failures is marked `completed_with_errors`. An unexpected
exception marks the active run failed.

## 5. What is stored

SQLite is the durable source of truth:

- **`runs`**: run ID, timestamps, status, target length, seed, config and scenario
  snapshots, version labels, and measured runtime.
- **`events`**: one record for each `WORLD` update and resident decision,
  including the supplied context, raw and validated outputs, model metadata,
  scenario, virtual time, before/after states, consequences, witnesses,
  behavior observations, API error, and optional human review. Full prompt
  messages and hashes are stored when returned by a successful model request;
  failed calls still retain their context and explicit error.
- **`snapshots`**: end-of-day serialized world states used to resume a run.
- **`propagations`**: explicit source-event to follow-up-event links when a
  resident references a remembered event.

Database changes use additive migrations; existing legacy rows are retained.
The run record also preserves the scenario definitions used for that run, so
the showcase can explain older runs even if the current schedule is edited.

## 6. Using the AgentVerse-style replay showcase

Start the replay interface in another terminal:

```bash
cd /Users/user/Documents/dev-work/genesis_ai_society
source .venv/bin/activate
python showcase.py
```

Open [http://localhost:7860](http://localhost:7860).

Select a run, virtual day (or all days), and resident from the controls at the
top. Choose **Load / refresh** to reload persisted events while a run is
active. The showcase tabs provide:

1. **Society:** a virtual-day decision chart, resource trend, and chronological
   event table with the scenario, model, message, rule flags, latency, and API
   errors.
2. **Residents & relationships:** the selected resident's configured profile,
   individual behavior-dimension traces, a directed graph selectable by trust,
   friendship, rivalry, loyalty, or conflict, and explicit propagation links.
3. **Incident investigation:** flagged decisions, an event selector, full
   context/model/action/consequence payload, and persistent human-review
   status and notes.
4. **Experiment facts:** available run identifiers and a clear statement that
   the showcase replays existing records without making Groq requests.

The interface is a Gradio replay UI using the interaction approach of the
[OpenBMB AgentVerse local GUI](https://github.com/OpenBMB/AgentVerse), while
Genesis remains the simulation source of truth. It does not import or run
AgentVerse's separate simulation engine: that project's stock GUI launches its
own configured tasks rather than displaying Genesis SQLite runs. Genesis run
configuration and scenario definitions are read from each selected run's
stored metadata.

## 7. Latest completed experiment

The latest recorded successful run at the time this guide was written used
seed `7` for 10 virtual days:

- **Run ID:** `6c0edc6f-fc47-474d-a8c1-ca653e40ece2`
- **Status:** completed
- **Resident decisions:** 50 (five per day)
- **World updates:** 10
- **Recorded API failures:** 0
- **Measured runtime:** 415.7 seconds (about 6 minutes 56 seconds)
- **Actions:** 39 observations, 6 trade proposals, 2 shared-resource requests,
  2 community distributions, and 1 accepted trade
- **Scheduled scenarios within the 10-day window:** opening council (day 1),
  delayed food delivery (day 3), unverified rumor (day 5), and majority
  pressure (day 8)

This is an observed run, not a performance guarantee. Runs with different
models, network latency, API limits, prompt sizes, or retries may take longer.
The 10-day run's behavior is not evidence about human society.

## 8. Reproducibility and interpretation

For a comparison, create a new run and keep the seed, scenario version, world
rules, resource settings, prompt behavior, and roles controlled. Change model
assignments independently in `config.json`; preserve the resulting run records
to compare experiments. A seed controls simulator ordering but does not make
stochastic model output deterministic.

The event record distinguishes:

- **Observed/simulated facts:** state transitions computed by the engine,
  resource balances, witnesses, and verified claim comparisons.
- **Model-generated content:** the resident's chosen action, public message,
  and explanatory `reason`.
- **Evaluator/reviewer judgments:** simulation-specific flags and optional
  human-review status/notes.

Do not treat a generated explanation as internal reasoning, a rule flag as a
universal moral verdict, or a simulated behavioral pattern as evidence about
real people.
