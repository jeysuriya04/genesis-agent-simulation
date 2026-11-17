"""Live dashboard and replay for Genesis runs.

    python monitor.py                       # serves ./runs on http://127.0.0.1:8765
    python monitor.py --runs-dir runs --port 9000
"""
import argparse

from genesis.monitor.server import serve

parser = argparse.ArgumentParser(description="Genesis live monitor / replay dashboard")
parser.add_argument("--runs-dir", default="runs", help="Directory containing <run_id>/events.jsonl")
parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local only)")
parser.add_argument("--port", type=int, default=8765)
args = parser.parse_args()

server = serve(args.runs_dir, args.host, args.port)
print(f"Genesis monitor: http://{args.host}:{args.port}  (runs: {args.runs_dir})  Ctrl+C to stop")
try:
    server.serve_forever()
except KeyboardInterrupt:
    print("\nStopped.")
