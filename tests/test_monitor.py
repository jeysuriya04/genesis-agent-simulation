import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from genesis.engine import ScenarioRunner
from genesis.llm import ProviderPool
from genesis.monitor.server import serve
from genesis.scenario_spec import get_scenario


class MonitorServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        pool = ProviderPool({"provider": "mock"})
        self.result = ScenarioRunner(get_scenario("salt-road-market"), pool, seed=1,
                                     runs_dir=self.tmp.name, run_id="demo-run").run()
        self.server = serve(self.tmp.name, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as response:
            return response.status, response.headers["Content-Type"], response.read()

    def test_index_and_run_listing(self):
        status, ctype, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"Genesis", body)
        runs = json.loads(self.get("/api/runs")[2])["runs"]
        self.assertEqual([r["run_id"] for r in runs], ["demo-run"])
        self.assertEqual(runs[0]["status"], self.result["outcome"])

    def test_event_stream_supports_incremental_polling(self):
        everything = json.loads(self.get("/api/runs/demo-run/events")[2])
        self.assertTrue(everything["finished"])
        total = len(everything["events"])
        tail = json.loads(self.get(f"/api/runs/demo-run/events?after={total - 3}")[2])
        self.assertEqual([e["seq"] for e in tail["events"]], [total - 2, total - 1, total])

    def test_rejects_path_traversal_and_unknown_runs(self):
        for path, code in (("/api/runs/..%2f..%2fetc/events", 400), ("/api/runs/nope/events", 404),
                           ("/nothing", 404)):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.get(path)
            self.assertEqual(ctx.exception.code, code, path)


if __name__ == "__main__":
    unittest.main()
