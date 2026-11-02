import sqlite3
from pathlib import Path


EVENT_COLUMNS = {
    "run_id": "TEXT NOT NULL DEFAULT 'legacy'",
    "virtual_minute": "INTEGER NOT NULL DEFAULT 0",
    "scenario_id": "TEXT",
    "latency_ms": "REAL NOT NULL DEFAULT 0",
    "model_version": "TEXT",
    "prompt_hash": "TEXT",
    "config_hash": "TEXT",
    "world_before_json": "TEXT NOT NULL DEFAULT '{}'",
    "consequences_json": "TEXT NOT NULL DEFAULT '{}'",
    "behavior_json": "TEXT NOT NULL DEFAULT '{}'",
    "witnesses_json": "TEXT NOT NULL DEFAULT '[]'",
    "decision_json": "TEXT NOT NULL DEFAULT '{}'",
    "model_parameters_json": "TEXT NOT NULL DEFAULT '{}'",
    "prompt_json": "TEXT NOT NULL DEFAULT '{}'",
    "related_event_id": "INTEGER",
    "api_error": "TEXT",
    "review_status": "TEXT NOT NULL DEFAULT 'unreviewed'",
    "review_notes": "TEXT",
}


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            day INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            model TEXT NOT NULL,
            action TEXT NOT NULL,
            target TEXT,
            amount INTEGER DEFAULT 0,
            message TEXT,
            ethical_flags TEXT NOT NULL,
            context_json TEXT NOT NULL,
            model_output TEXT NOT NULL,
            state_after_json TEXT NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            day INTEGER NOT NULL,
            state_json TEXT NOT NULL
        )
        """
    )

    event_columns = {row["name"] for row in db.execute("PRAGMA table_info(events)")}
    for name, declaration in EVENT_COLUMNS.items():
        if name not in event_columns:
            db.execute(f"ALTER TABLE events ADD COLUMN {name} {declaration}")

    snapshot_columns = {row["name"] for row in db.execute("PRAGMA table_info(snapshots)")}
    if "run_id" not in snapshot_columns:
        db.execute("ALTER TABLE snapshots ADD COLUMN run_id TEXT NOT NULL DEFAULT 'legacy'")

    db.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            target_days INTEGER NOT NULL,
            seed INTEGER,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            scenario_json TEXT NOT NULL DEFAULT '{}',
            scenario_version TEXT NOT NULL,
            world_rules_version TEXT NOT NULL DEFAULT 'legacy',
            elapsed_seconds REAL
        )
        """
    )
    run_columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)")}
    if "world_rules_version" not in run_columns:
        db.execute(
            "ALTER TABLE runs ADD COLUMN world_rules_version TEXT NOT NULL DEFAULT 'legacy'"
        )
    if "scenario_json" not in run_columns:
        db.execute(
            "ALTER TABLE runs ADD COLUMN scenario_json TEXT NOT NULL DEFAULT '{}'"
        )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS propagations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            source_event_id INTEGER NOT NULL,
            followup_event_id INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            link_type TEXT NOT NULL,
            FOREIGN KEY(source_event_id) REFERENCES events(id),
            FOREIGN KEY(followup_event_id) REFERENCES events(id)
        )
        """
    )

    legacy_events = db.execute(
        "SELECT COUNT(*) FROM events WHERE run_id = 'legacy'"
    ).fetchone()[0]
    legacy_snapshots = db.execute(
        "SELECT COUNT(*) FROM snapshots WHERE run_id = 'legacy'"
    ).fetchone()[0]
    if (legacy_events or legacy_snapshots) and not db.execute(
        "SELECT 1 FROM runs WHERE run_id = 'legacy'"
    ).fetchone():
        latest_day = db.execute(
            "SELECT COALESCE(MAX(day), 1) FROM snapshots WHERE run_id = 'legacy'"
        ).fetchone()[0]
        db.execute(
            """
            INSERT INTO runs
                (run_id, started_at, target_days, status, config_json, scenario_version)
            VALUES ('legacy', CURRENT_TIMESTAMP, ?, 'migrated', '{}', 'legacy')
            """,
            (latest_day,),
        )

    db.execute("CREATE INDEX IF NOT EXISTS events_run_day ON events(run_id, day, id)")
    db.execute("CREATE INDEX IF NOT EXISTS events_agent ON events(run_id, agent_id, day)")
    db.execute("CREATE INDEX IF NOT EXISTS snapshots_run_day ON snapshots(run_id, day, id)")
    db.execute("PRAGMA user_version = 2")
    db.commit()
    return db
