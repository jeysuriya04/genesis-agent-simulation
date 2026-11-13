"""Genesis command line.

Scenario mode (tightly scoped scenarios, Grok agents, JSONL events for the monitor):
    python run.py --list-scenarios
    python run.py --scenario drought-village              # real Grok via XAI_API_KEY
    python run.py --scenario salt-road-market --mock      # no API key needed

Legacy society mode (original 5-resident, SQLite-backed day simulation):
    python run.py --reset --days 30
"""
import argparse
import sys

from genesis.llm import ProviderPool, UsageTracker, load_env, merge_llm_config
from genesis.scenario_spec import ScenarioError, get_scenario, list_scenarios

load_env()

parser = argparse.ArgumentParser(
    description="Run Genesis scenarios (or the legacy society) with Grok or mock agents.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__,
)
parser.add_argument("--scenario", help="Scenario id (see --list-scenarios) or path to a scenario .json.")
parser.add_argument("--list-scenarios", action="store_true", help="List and validate bundled scenarios.")
parser.add_argument("--turns", type=int, help="Cap the scenario's turn limit (never raises it).")
parser.add_argument("--days", type=int, default=30, help="Legacy society mode: virtual days.")
parser.add_argument("--reset", action="store_true",
                    help="Legacy society mode: start a fresh run, keeping earlier history.")
parser.add_argument("--seed", type=int, default=None, help="Seed for action ordering (default: config).")
parser.add_argument("--concurrency", type=int, default=None, help="Concurrent LLM calls per turn.")
parser.add_argument("--provider", choices=["xai", "groq", "openai", "mock"],
                    help="LLM provider (default from config.json: xai).")
parser.add_argument("--mock", action="store_true", help="Shortcut for --provider mock (no API key).")
parser.add_argument("--mock-delay", type=float, default=0.0,
                    help="Seconds of fake latency per mock call, to watch a run live in the monitor.")
parser.add_argument("--model-main", help="Override the main-tier model id (default grok-4.7).")
parser.add_argument("--model-light", help="Override the light-tier model id.")
parser.add_argument("--max-cost", type=float, help="Stop the run once estimated spend reaches this USD.")
parser.add_argument("--runs-dir", default="runs", help="Where JSONL event logs are written.")
parser.add_argument("--no-events", action="store_true", help="Legacy mode: skip the JSONL event log.")
args = parser.parse_args()


def llm_overrides(base_config):
    override = {}
    if args.mock:
        args.provider = "mock"
    if args.provider:
        override["provider"] = args.provider
    tiers = {}
    if args.model_main:
        tiers["main"] = {"model": args.model_main}
    if args.model_light:
        tiers["light"] = {"model": args.model_light}
    if tiers:
        override["tiers"] = tiers
    if args.max_cost is not None:
        override["max_cost_usd"] = args.max_cost
    return merge_llm_config(base_config.get("llm"), override)


def run_scenario():
    import json
    from pathlib import Path

    from genesis.engine import ScenarioRunner

    base = json.loads((Path(__file__).parent / "config.json").read_text())
    llm = llm_overrides(base)
    seed = args.seed if args.seed is not None else base.get("seed", 7)
    spec = get_scenario(args.scenario)
    pool = ProviderPool(llm, pricing=base.get("pricing"), mock_seed=seed, mock_delay=args.mock_delay)
    tracker = UsageTracker(max_cost_usd=llm.get("max_cost_usd"))

    def progress(step):
        agent = step["agent_id"]
        flags = f" [{','.join(step['validation']['flags'])}]" if step["validation"]["flags"] else ""
        cost = step["llm"].get("cost_usd")
        print(f"T{step['turn']:02d} {agent:<8} {step['action']['action']:<10} "
              f"{step['effect']['summary']}{flags}"
              f"{'' if cost is None else f'  (${cost:.4f})'}")

    runner = ScenarioRunner(spec, pool, seed=seed, runs_dir=args.runs_dir, tracker=tracker,
                            concurrency=args.concurrency or base.get("max_concurrency", 4),
                            progress=progress, turn_limit=args.turns)
    print(f"Scenario: {spec['title']} ({spec['kind']}) | provider={llm['provider']} | "
          f"run {runner.run_id} | up to {runner.spec['turn_limit']} turns")
    print(f"Events: {runner.log.path}")
    result = runner.run()
    usage = result["usage"]
    print(f"\nOutcome: {result['outcome'].upper()} after {result['turns_played']} turns - {result['reason']}")
    if result["scores"]:
        print("Scores:", ", ".join(f"{k}={v}" for k, v in result["scores"].items()))
    print(f"LLM usage: {usage['calls']} calls, {usage['total_tokens']} tokens, "
          f"{usage['retries']} retries, est. cost ${usage['cost_usd']:.4f}")
    print(f"Watch/replay it: python monitor.py --runs-dir {args.runs_dir}")


def run_society():
    from genesis.simulation import CONFIG, run_simulation

    config = json.loads(json.dumps(CONFIG))
    config["llm"] = llm_overrides(config)
    seed = args.seed if args.seed is not None else CONFIG.get("seed", 7)
    concurrency = args.concurrency or CONFIG.get("max_concurrency", 5)
    print(
        f"Virtual time: target day {args.days}; {len(config['agents'])} agent decisions "
        f"per simulated day via {config['llm']['provider']}. "
        f"At most {args.days * len(config['agents'])} requests for a fresh run; "
        "no real-time day waits."
    )

    def progress(day, agent_id, event):
        api_status = f" | API ERROR: {event['api_error']}" if event["api_error"] else ""
        cost = event.get("total_cost_usd")
        print(
            f"Progress {event['completed_decisions']}/{event['target_decisions']} | "
            f"Day {day:02d} | elapsed {event['elapsed_seconds']:.1f}s | "
            f"{agent_id} | {event['action']} | "
            f"{event['latency_ms']:.0f} ms | {event['outcome']}{api_status}"
            f"{'' if not cost else f' | spend ${cost:.4f}'}"
        )

    state = run_simulation(
        days=args.days, progress=progress, reset=args.reset, seed=seed,
        concurrency=concurrency, config=config,
        events_dir=None if args.no_events else args.runs_dir,
    )
    usage = state.get("usage", {})
    print(
        f"Simulation run {state['run_id']} completed through day {args.days} "
        f"in {state['elapsed_seconds']:.1f} seconds ({state['run_status']}). "
        f"LLM calls: {usage.get('calls', 0)}, est. cost ${usage.get('cost_usd', 0):.4f}."
    )
    print("Open the replay showcase with: python showcase.py"
          + ("" if args.no_events else f"  |  live monitor: python monitor.py --runs-dir {args.runs_dir}"))


import json  # noqa: E402  (used by both modes)

try:
    if args.list_scenarios:
        for sid, spec in list_scenarios().items():
            print(f"{sid:<18} {spec['kind']:<16} {spec['turn_limit']:>2} turns  "
                  f"{len(spec['agents'])} agents  - {spec['summary']}")
    elif args.scenario:
        run_scenario()
    else:
        run_society()
except (ScenarioError, KeyError) as exc:
    sys.exit(f"error: {exc.args[0] if isinstance(exc, KeyError) else exc}")
except Exception as exc:  # noqa: BLE001 - keep missing-key errors friendly
    from genesis.llm import LLMAuthError
    if isinstance(exc, LLMAuthError):
        sys.exit(f"error: {exc}\n(or run offline with --mock)")
    raise
