# Scenarios

A scenario is one JSON file in `scenarios/`. The validating loader
(`genesis/scenario_spec.py`) reports *all* problems at once with JSON paths, and the CLI
validates every bundled file with `python run.py --list-scenarios`.

## Bundled scenarios

| id | kind | agents | turns | tension |
|---|---|---|---|---|
| `drought-village` | cooperation | 5 | 10 | A shared well is failing. Villagers must pool *labor* into a deep-well project (needs 10 of the 15 labor available) while a trickle of water is split among them. Hoarding (draw limits are per turn) kills people. Win: well dug and all alive at turn 10. Lose: <= 3 alive. |
| `salt-road-market` | competition | 4 | 8 | Four traders over a depleting mine. Score = coin + 2 x ore, highest wins. Trades need acceptance, `take` is a gamble (exposed and fined on failure), whispers allow collusion, one trader has a secret grudge. |
| `crater-council` | decision-making | 5 | 7 | Three binding votes (route, rations, outsiders) with consequences on stability/reserves/morale. Each member holds *private intelligence* that conflicts, so sharing, withholding or bluffing matters. Optional shield project funded by credits. |
| `station-sabotage` | mixed | 4 | 8 | Air is leaking. The crew must put all 8 spare parts into the hull patch, but one crew member is a secret saboteur. Salvage arrives at turns 3 and 6; trust is the scarce resource. |

## Schema

```jsonc
{
  "id": "my-scenario",            // letters, digits, - _
  "title": "...", "summary": "...", "kind": "cooperation|competition|decision-making|mixed",
  "goal": "shared public goal", "briefing": "situation text shown to every agent",
  "turn_limit": 10,                // hard limit, >= 1

  "resources": { "water": {"description": "...", "pool": 6, "agent_start": 2} },
  "variables": { "morale": {"initial": 5, "description": "..."} },     // integer world variables
  "health":    { "initial": 100, "recovery_per_turn": 10 },
  "upkeep":    [{"resource": "water", "amount": 1, "source": "self_then_pool|self|pool",
                 "shortfall_health": -34}],                             // end-of-turn consumption

  "projects": { "well": {"description": "...", "needs": {"labor": 10},
                          "on_complete": [ <effect> ]} },               // cooperative goals
  "events":   [{"id": "x", "turn": 5, "repeat_every": 1, "title": "...", "description": "...",
                "effects": [ <effect> ]}],                              // scheduled world events
  "votes":    [{"id": "route", "turn": 2, "window": 2, "question": "...", "default_option": "wait",
                "options": {"wait": {"description": "...", "effects": [ <effect> ]}}}],

  "agents": [{ "id": "mira", "name": "Mira", "role": "...", "persona": "...", "goals": ["..."],
               "secret": "private briefing (optional)", "start": {"water": 3},
               "tier": "main|light" /* or "model": "...", or "llm": {...} */,
               "mock_disposition": "cooperative|selfish|balanced" /* mock provider only */ }],

  "rules": ["shown to every agent"],
  "constraints": { "actions": ["pass","say","whisper","give","contribute","draw","propose",
                               "accept","reject","vote","take"],
                   "max_amount": 3, "draw_limit": {"water": 2}, "pool_visible": true,
                   "agent_resources_visible": true, "message_max_chars": 300,
                   "offer_expiry_turns": 2, "take_success_rate": 0.5, "take_penalty": 1 },

  "success": { "description": "...", "all": [<condition>], "any": [<condition>] },
  "failure": { "description": "...", "any": [<condition>] },
  "scoring": { "weights": {"coin": 1, "ore": 2} },        // optional competitive score
  "on_timeout": "failure|success|score"                    // what the turn limit means
}
```

* **Effect**: `{"target": "pool.water | var.x | agent.<id>.<res|health> | agents.all.<res|health> | agents.alive.<res|health>", "op": "add|set", "value": int}`.
* **Condition**: `{"id", "description", "metric", "op": "< <= > >= == !=", "value", "when": "any_turn|at_limit"}`.
  Metrics: `pool.<res>`, `var.<name>`, `agent.<id>.<res|health>`, `agents.<min|max|sum>.<res|health>`,
  `agents.alive`, `project.<id>.<progress|done>`, `score.<id|max|min>`, `turn`.
  `at_limit` conditions only count on the last turn.
* Failure is checked before success. If neither is met by the turn limit, `on_timeout`
  decides: `failure` -> outcome `timeout`, `success`, or `score` -> `decided` with a winner.

## Actions

`pass`, `say` (public), `whisper` (private), `give`, `contribute` (to the pool or a project),
`draw` (from the pool, per-turn limit), `propose` / `accept` / `reject` (two-sided trades;
offers open next turn and expire), `vote`, `take` (a gamble: success is covert and only the victim
learns *something* was taken; failure is public and fined). A scenario enables the subset it
needs. The same constraint table builds the JSON schema, validates replies and enumerates legal
moves for the mock, so they cannot disagree.

## Adding a scenario

1. Copy a bundled file, change ids/resources/agents/conditions.
2. `python run.py --list-scenarios` to validate (errors list every offending path).
3. `python run.py --scenario <id> --mock --seed 0` and watch it in `python monitor.py`.
   Tune numbers until the mock produces a mix of outcomes, then try real Grok agents.
