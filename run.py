import argparse
from genesis.simulation import CONFIG, run_simulation

parser = argparse.ArgumentParser(
    description="Run the Genesis society using accelerated, event-driven virtual days."
)
parser.add_argument("--days", type=int, default=30)
parser.add_argument(
    "--reset", action="store_true",
    help="Start a fresh run without deleting existing experiment history.",
)
parser.add_argument("--seed", type=int, default=CONFIG.get("seed", 7))
parser.add_argument("--concurrency", type=int, default=CONFIG.get("max_concurrency", 5))
args = parser.parse_args()

print(
    f"Virtual time: target day {args.days}; {len(CONFIG['agents'])} agent decisions "
    "per simulated day. "
    f"At most {args.days * len(CONFIG['agents'])} requests for a fresh run; "
    "no real-time day waits."
)

def progress(day, agent_id, event):
    api_status = f" | API ERROR: {event['api_error']}" if event["api_error"] else ""
    print(
        f"Progress {event['completed_decisions']}/{event['target_decisions']} | "
        f"Day {day:02d} | elapsed {event['elapsed_seconds']:.1f}s | "
        f"{agent_id} | {event['action']} | "
        f"{event['latency_ms']:.0f} ms | {event['outcome']}{api_status}"
    )

state = run_simulation(
    days=args.days,
    progress=progress,
    reset=args.reset,
    seed=args.seed,
    concurrency=args.concurrency,
)
print(
    f"Simulation run {state['run_id']} completed through day {args.days} "
    f"in {state['elapsed_seconds']:.1f} seconds ({state['run_status']})."
)
print("Simulation run complete. Open the replay showcase with: python showcase.py")
