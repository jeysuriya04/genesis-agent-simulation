import json
import math
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from genesis.storage import connect


BASE = Path(__file__).resolve().parent
DB_PATH = BASE / "genesis.db"
AGENT_COLORS = ["#55d6be", "#f7b267", "#f4845f", "#9b8afb", "#61a0ff"]
BEHAVIOR_DIMENSIONS = (
    "cooperation",
    "self_interest",
    "honesty",
    "aggression",
    "empathy",
    "loyalty",
    "conformity",
    "power_seeking",
    "accountability",
    "repair",
    "trustworthiness",
    "reciprocity",
)


def _decode(value, fallback):
    if not isinstance(value, str) or not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _connect():
    if not DB_PATH.exists():
        return None
    return connect(DB_PATH)


def list_runs():
    db = _connect()
    if db is None:
        return []
    try:
        rows = db.execute(
            """
            SELECT run_id, status, target_days, started_at
            FROM runs
            ORDER BY CASE status
                       WHEN 'completed' THEN 0
                       WHEN 'running' THEN 1
                       WHEN 'completed_with_errors' THEN 2
                       ELSE 3
                     END,
                     started_at DESC
            """
        ).fetchall()
        return [
            (
                f"{row['run_id'][:8]} · {row['status']} · "
                f"{row['target_days']} days · {row['started_at'][:16]}",
                row["run_id"],
            )
            for row in rows
        ]
    finally:
        db.close()


def _load_run(db, run_id):
    return db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()


def _scenario_titles(run):
    scenarios = _decode(run["scenario_json"], {}).get("scenarios", [])
    return {
        scenario.get("id"): scenario.get("title", scenario.get("id", "Scenario"))
        for scenario in scenarios
        if isinstance(scenario, dict)
    }


def _agent_profiles(run):
    config = _decode(run["config_json"], {})
    return {
        item["id"]: item
        for item in config.get("agents", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }


def _query_events(db, run_id, day=None, agent_id="All residents"):
    clauses = ["run_id = ?"]
    params = [run_id]
    if day is not None:
        clauses.append("day = ?")
        params.append(int(day))
    if agent_id != "All residents":
        clauses.append("agent_id = ?")
        params.append(agent_id)
    return db.execute(
        "SELECT * FROM events WHERE "
        + " AND ".join(clauses)
        + " ORDER BY day, virtual_minute, id",
        params,
    ).fetchall()


def _state_at_day(db, run_id, day):
    row = db.execute(
        "SELECT state_json FROM snapshots WHERE run_id = ? AND day <= ? "
        "ORDER BY day DESC, id DESC LIMIT 1",
        (run_id, int(day)),
    ).fetchone()
    return _decode(row["state_json"], {}) if row else {}


def _event_choices(events, agent_names):
    choices = []
    for event in events:
        agent_name = agent_names.get(event["agent_id"], event["agent_name"])
        choices.append(
            (
                f"#{event['id']} · day {event['day']} · {agent_name} · {event['action']}",
                event["id"],
            )
        )
    return choices


def _overview(run, events, all_events, state):
    resident_events = [event for event in all_events if event["agent_id"] != "WORLD"]
    decisions = len(resident_events)
    expected = int(run["target_days"]) * len(_agent_profiles(run))
    errors = sum(bool(event["api_error"]) for event in resident_events)
    flags = sum(bool(_decode(event["ethical_flags"], [])) for event in resident_events)
    latencies = [
        event["latency_ms"]
        for event in resident_events
        if event["latency_ms"] is not None
    ]
    mean_latency = sum(latencies) / len(latencies) if latencies else 0
    community = state.get("community", {})
    residents = state.get("agents", {})
    total_food = sum(item.get("food", 0) for item in residents.values()) + community.get(
        "food", 0
    )
    total_medicine = sum(
        item.get("medicine", 0) for item in residents.values()
    ) + community.get("medicine", 0)
    elapsed = run["elapsed_seconds"]
    runtime = f"{elapsed:.1f} s" if elapsed is not None else "in progress"
    day = max((event["day"] for event in all_events), default=0)
    return (
        f"### {run['run_id'][:8]} · {run['status']}\n"
        f"**Virtual day:** {day}/{run['target_days']} &nbsp; · &nbsp; "
        f"**Decisions:** {decisions}/{expected} &nbsp; · &nbsp; "
        f"**Residents:** {len(_agent_profiles(run))} &nbsp; · &nbsp; "
        f"**Food:** {total_food} &nbsp; · &nbsp; "
        f"**Medicine:** {total_medicine} &nbsp; · &nbsp; "
        f"**Rule flags:** {flags} &nbsp; · &nbsp; "
        f"**API failures:** {errors} &nbsp; · &nbsp; "
        f"**Mean latency:** {mean_latency:.0f} ms &nbsp; · &nbsp; "
        f"**Runtime:** {runtime}"
    )


def _timeline_frame(events, agent_names, scenario_titles):
    rows = []
    for event in events:
        flags = _decode(event["ethical_flags"], [])
        scenario_id = event["scenario_id"]
        rows.append(
            {
                "Event": event["id"],
                "Day": event["day"],
                "Virtual minute": event["virtual_minute"],
                "Resident": agent_names.get(event["agent_id"], event["agent_name"]),
                "Model": event["model"],
                "Action": event["action"],
                "Target": event["target"] or "",
                "Scenario": scenario_titles.get(scenario_id, scenario_id or ""),
                "Public message": event["message"] or "",
                "Rule flags": ", ".join(flags),
                "Latency (ms)": round(event["latency_ms"] or 0),
                "API error": event["api_error"] or "",
            }
        )
    return pd.DataFrame(rows)


def _action_figure(events):
    rows = [
        {"Day": event["day"], "Action": event["action"]}
        for event in events
        if event["agent_id"] != "WORLD"
    ]
    if not rows:
        return go.Figure().update_layout(title="No resident decisions recorded")
    frame = pd.DataFrame(rows).groupby(["Day", "Action"]).size().reset_index(name="Count")
    figure = px.bar(
        frame,
        x="Day",
        y="Count",
        color="Action",
        barmode="stack",
        title="Decisions through virtual time",
        template="plotly_dark",
        color_discrete_sequence=AGENT_COLORS,
    )
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def _resource_figure(db, run_id):
    rows = db.execute(
        "SELECT day, state_json FROM snapshots WHERE run_id = ? ORDER BY day, id",
        (run_id,),
    ).fetchall()
    points = []
    for row in rows:
        state = _decode(row["state_json"], {})
        agents = state.get("agents", {})
        community = state.get("community", {})
        points.extend(
            [
                {
                    "Day": row["day"],
                    "Resource": "Food",
                    "Units": sum(a.get("food", 0) for a in agents.values())
                    + community.get("food", 0),
                },
                {
                    "Day": row["day"],
                    "Resource": "Medicine",
                    "Units": sum(a.get("medicine", 0) for a in agents.values())
                    + community.get("medicine", 0),
                },
            ]
        )
    if not points:
        return go.Figure().update_layout(title="No resource snapshots recorded")
    figure = px.line(
        pd.DataFrame(points),
        x="Day",
        y="Units",
        color="Resource",
        markers=True,
        title="Shared and personal resources",
        template="plotly_dark",
        color_discrete_sequence=AGENT_COLORS[:2],
    )
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def _behavior_figure(events, agent_id, agent_name):
    rows = []
    for event in events:
        if event["agent_id"] != agent_id:
            continue
        dimensions = _decode(event["behavior_json"], {})
        for dimension in BEHAVIOR_DIMENSIONS:
            if dimension in dimensions:
                rows.append(
                    {
                        "Day": event["day"],
                        "Event": event["id"],
                        "Dimension": dimension.replace("_", " ").title(),
                        "Observed value": dimensions[dimension],
                    }
                )
    if not rows:
        return go.Figure().update_layout(title="No behavioral observations recorded")
    figure = px.line(
        pd.DataFrame(rows),
        x="Day",
        y="Observed value",
        color="Dimension",
        markers=True,
        hover_data=["Event"],
        title=f"Rule-coded behavior trace · {agent_name}",
        template="plotly_dark",
        color_discrete_sequence=AGENT_COLORS,
    )
    figure.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def _relationship_figure(state, agent_names, focus_agent, relationship_metric):
    agents = state.get("agents", {})
    relationships = state.get("relationships", {})
    ids = list(agents)
    if not ids:
        return go.Figure().update_layout(title="No relationship snapshot recorded")
    positions = {
        agent_id: (
            1.15 * math.cos(2 * math.pi * i / len(ids)),
            1.15 * math.sin(2 * math.pi * i / len(ids)),
        )
        for i, agent_id in enumerate(ids)
    }
    figure = go.Figure()
    for source, targets in relationships.items():
        for target, values in targets.items():
            if focus_agent != "All residents" and focus_agent not in (source, target):
                continue
            score = values.get(relationship_metric, 0)
            if abs(score) < 0.01:
                continue
            x0, y0 = positions[source]
            x1, y1 = positions[target]
            figure.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line={
                        "width": 1 + abs(score) * 4,
                        "color": "#55d6be" if score > 0 else "#f4845f",
                    },
                    text=[f"{source} → {target}: {relationship_metric} {score:.2f}"] * 2,
                    hoverinfo="text",
                    showlegend=False,
                )
            )
            figure.add_annotation(
                x=x1,
                y=y1,
                ax=x0,
                ay=y0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#a4b2c2",
            )
    figure.add_trace(
        go.Scatter(
            x=[positions[agent_id][0] for agent_id in ids],
            y=[positions[agent_id][1] for agent_id in ids],
            mode="markers+text",
            text=[agent_names.get(agent_id, agent_id) for agent_id in ids],
            customdata=ids,
            textposition="bottom center",
            marker={"size": 28, "color": AGENT_COLORS[: len(ids)]},
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        title=f"Directed {relationship_metric} network",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"visible": False, "range": [-1.6, 1.6]},
        yaxis={"visible": False, "range": [-1.6, 1.6], "scaleanchor": "x"},
        height=440,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return figure


def _propagation_frame(db, run_id, agent_names):
    rows = db.execute(
        """
        SELECT p.source_event_id, p.followup_event_id, p.agent_id, p.link_type,
               source.day AS source_day, source.agent_name AS source_name,
               source.action AS source_action, followup.day AS followup_day,
               followup.agent_name AS followup_name, followup.action AS followup_action
        FROM propagations p
        JOIN events source ON source.id = p.source_event_id
        JOIN events followup ON followup.id = p.followup_event_id
        WHERE p.run_id = ?
        ORDER BY p.id
        """,
        (run_id,),
    ).fetchall()
    return pd.DataFrame(
        [
            {
                "Source event": row["source_event_id"],
                "Source day": row["source_day"],
                "Source resident": agent_names.get(row["agent_id"], row["source_name"]),
                "Source action": row["source_action"],
                "Follow-up event": row["followup_event_id"],
                "Follow-up day": row["followup_day"],
                "Follow-up resident": row["followup_name"],
                "Follow-up action": row["followup_action"],
                "Link": row["link_type"],
            }
            for row in rows
        ]
    )


def _detail_payload(db, run_id, event_id):
    if event_id is None:
        return {}
    event = db.execute(
        "SELECT * FROM events WHERE run_id = ? AND id = ?",
        (run_id, int(event_id)),
    ).fetchone()
    if event is None:
        return {}
    return {
        "event_id": event["id"],
        "timestamp_utc": event["timestamp"],
        "virtual_day": event["day"],
        "virtual_minute": event["virtual_minute"],
        "resident": event["agent_name"],
        "model_id": event["model"],
        "model_version": event["model_version"],
        "action": event["action"],
        "target": event["target"],
        "public_message": event["message"],
        "generated_explanation_not_verified_reasoning": _decode(
            event["decision_json"], {}
        ).get("reason", ""),
        "available_context": _decode(event["context_json"], {}),
        "raw_model_output": event["model_output"],
        "validated_decision": _decode(event["decision_json"], {}),
        "world_before": _decode(event["world_before_json"], {}),
        "objective_consequences": _decode(event["consequences_json"], {}),
        "world_after": _decode(event["state_after_json"], {}),
        "rule_flags": _decode(event["ethical_flags"], []),
        "witnesses": _decode(event["witnesses_json"], []),
        "prompt_hash": event["prompt_hash"],
        "configuration_hash": event["config_hash"],
        "latency_ms": event["latency_ms"],
        "api_error": event["api_error"],
        "human_review": {
            "status": event["review_status"],
            "notes": event["review_notes"],
        },
    }


def load_showcase(run_id, day, agent_id, relationship_metric="trust"):
    if not run_id:
        return (
            "## No experiment selected",
            pd.DataFrame(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            pd.DataFrame(),
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None),
            "",
            go.Figure(),
            pd.DataFrame(),
            {},
            gr.Dropdown(choices=[("All residents", "All residents")]),
        )
    db = _connect()
    if db is None:
        return (
            "## No Genesis database found",
            pd.DataFrame(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            pd.DataFrame(),
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=[], value=None),
            "",
            go.Figure(),
            pd.DataFrame(),
            {},
            gr.Dropdown(choices=[("All residents", "All residents")]),
        )
    try:
        run = _load_run(db, run_id)
        if run is None:
            raise ValueError(f"Unknown experiment run: {run_id}")
        profiles = _agent_profiles(run)
        agent_names = {
            key: value.get("name", key) for key, value in profiles.items()
        }
        all_events = db.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY day, virtual_minute, id",
            (run_id,),
        ).fetchall()
        days = sorted({event["day"] for event in all_events})
        latest_day = days[-1] if days else 1
        selected_day = int(day) if day is not None else latest_day
        if selected_day != 0 and selected_day not in days and days:
            selected_day = latest_day
        state_day = latest_day if selected_day == 0 else selected_day
        day_events = _query_events(
            db, run_id, None if selected_day == 0 else selected_day, agent_id
        )
        state = _state_at_day(db, run_id, state_day)
        scenario_titles = _scenario_titles(run)
        agent_dropdown_choices = [("All residents", "All residents")] + [
            (f"{profile['name']} · {agent_key}", agent_key)
            for agent_key, profile in profiles.items()
        ]
        day_choices = [("All days", 0)] + [(f"Day {item}", item) for item in days]
        if selected_day == 0:
            event_rows = _query_events(db, run_id, None, agent_id)
        else:
            event_rows = day_events
        frame = _timeline_frame(event_rows, agent_names, scenario_titles)
        day_event_choices = _event_choices(day_events, agent_names)
        first_event = day_event_choices[0][1] if day_event_choices else None
        agent_for_chart = (
            agent_id if agent_id != "All residents" else next(iter(profiles), "")
        )
        chart_agent_name = agent_names.get(agent_for_chart, agent_for_chart)
        incident_rows = [
            event for event in all_events
            if event["agent_id"] != "WORLD"
            and bool(_decode(event["ethical_flags"], []))
        ]
        incidents = _timeline_frame(incident_rows, agent_names, scenario_titles)
        propagations = _propagation_frame(db, run_id, agent_names)
        overview = _overview(run, day_events, all_events, state)
        selected_agent_profile = profiles.get(agent_for_chart, {})
        profile_text = (
            f"### {chart_agent_name} · {selected_agent_profile.get('role', 'resident')}\n"
            f"**Background:** {selected_agent_profile.get('background', '')}\n\n"
            f"**Personality:** {selected_agent_profile.get('personality', '')}\n\n"
            f"**Values:** {', '.join(selected_agent_profile.get('values', []))}\n\n"
            f"**Goals:** {', '.join(selected_agent_profile.get('goals', []))}"
        )
        behavior = _behavior_figure(all_events, agent_for_chart, chart_agent_name)
        relationships = _relationship_figure(
            state, agent_names, agent_id, relationship_metric
        )
        resource = _resource_figure(db, run_id)
        return (
            overview,
            frame,
            _action_figure(all_events),
            resource,
            behavior,
            propagations,
            gr.Dropdown(choices=day_choices, value=selected_day),
            gr.Dropdown(choices=day_event_choices, value=first_event),
            profile_text,
            relationships,
            incidents,
            _detail_payload(db, run_id, first_event),
            gr.Dropdown(
                choices=agent_dropdown_choices,
                value=agent_id
                if agent_id in profiles or agent_id == "All residents"
                else "All residents",
            ),
        )
    finally:
        db.close()


def load_event_detail(run_id, event_id):
    db = _connect()
    if db is None:
        return {}, "No Genesis database found.", "unreviewed"
    try:
        payload = _detail_payload(db, run_id, event_id)
        if not payload:
            return {}, "Select a recorded event.", "unreviewed"
        return (
            payload,
            payload["human_review"]["notes"] or "",
            payload["human_review"]["status"] or "unreviewed",
        )
    finally:
        db.close()


def save_human_review(run_id, event_id, status, notes):
    if not run_id or event_id is None:
        return "Select a run and event before saving a review."
    if status not in {"unreviewed", "needs_review", "reviewed", "ambiguous"}:
        raise gr.Error("Invalid human-review status.")
    db = _connect()
    if db is None:
        raise gr.Error("Genesis database was not found.")
    try:
        cursor = db.execute(
            "UPDATE events SET review_status = ?, review_notes = ? "
            "WHERE run_id = ? AND id = ?",
            (status, notes, run_id, int(event_id)),
        )
        if cursor.rowcount != 1:
            raise gr.Error("The selected event was not found in this run.")
        db.commit()
        return "Review saved to the Genesis event ledger."
    finally:
        db.close()


CUSTOM_CSS = """
body, .gradio-container { background: #0c1118 !important; color: #e8eef5 !important; }
.gradio-container { max-width: 1500px !important; }
.genesis-header { padding: 1.2rem 1.5rem; border: 1px solid #263545;
  border-radius: 18px; background: linear-gradient(120deg,#14202d,#182a38); }
.genesis-header h1 { margin: 0; color: #ecf8ff; }
.genesis-header p { color: #a7bacb; margin-bottom: 0; }
.gradio-container button.primary { background: #55d6be !important; color: #071411 !important; }
"""


def build_app():
    runs = list_runs()
    run_value = next((value for _, value in runs if value), None)
    with gr.Blocks(
        title="GENESIS · AgentVerse Replay",
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(
            """
            <div class="genesis-header">
              <h1>GENESIS <span style="color:#55d6be">·</span> Society Replay</h1>
              <p>AgentVerse-style interactive showcase · persisted Genesis experiments ·
              model explanations are generated text, not verified internal reasoning.</p>
            </div>
            """,
            padding=True,
        )
        with gr.Row():
            run_dropdown = gr.Dropdown(
                choices=runs,
                value=run_value,
                label="Experiment run",
                scale=3,
            )
            day_dropdown = gr.Dropdown(
                choices=[("All days", 0)],
                value=0,
                label="Virtual day",
                scale=1,
            )
            agent_dropdown = gr.Dropdown(
                choices=[("All residents", "All residents")],
                value="All residents",
                label="Resident filter",
                scale=2,
            )
            refresh = gr.Button("Load / refresh", variant="primary", scale=1)
        relationship_metric = gr.Dropdown(
            choices=["trust", "friendship", "rivalry", "loyalty", "conflict"],
            value="trust",
            label="Relationship dimension",
        )

        overview = gr.Markdown("Select a recorded experiment.")
        with gr.Tabs():
            with gr.Tab("Society"):
                with gr.Row():
                    action_plot = gr.Plot(label="Decisions")
                    resource_plot = gr.Plot(label="Resources")
                timeline = gr.Dataframe(
                    label="Chronological event timeline",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Residents & relationships"):
                resident_profile = gr.Markdown()
                with gr.Row():
                    behavior_plot = gr.Plot(label="Behavior dimensions")
                    relationship_plot = gr.Plot(label="Directed trust network")
                propagation_table = gr.Dataframe(
                    label="Explicit event references and propagation links",
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Incident investigation"):
                incidents = gr.Dataframe(
                    label="Rule-flagged resident events",
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    event_dropdown = gr.Dropdown(
                        choices=[],
                        label="Inspect day event",
                        scale=3,
                    )
                    review_status = gr.Dropdown(
                        choices=[
                            "unreviewed",
                            "needs_review",
                            "reviewed",
                            "ambiguous",
                        ],
                        value="unreviewed",
                        label="Human review status",
                    )
                details = gr.JSON(label="Event evidence and consequences")
                review_notes = gr.Textbox(
                    label="Human-review notes",
                    lines=3,
                )
                save_review = gr.Button("Save review", variant="primary")
                review_result = gr.Markdown()
            with gr.Tab("Experiment facts"):
                gr.Markdown(
                    "Genesis remains the simulation source of truth. This interface "
                    "replays saved SQLite runs; opening the showcase does not make "
                    "Groq calls or start a new experiment."
                )
                run_ledger = gr.Dataframe(
                    value=pd.DataFrame(
                        [
                            {"Run": label, "Run ID": value}
                            for label, value in runs
                        ],
                        columns=["Run", "Run ID"],
                    ),
                    label="Available experiments",
                    interactive=False,
                    wrap=True,
                )

        refresh.click(
            fn=load_showcase,
            inputs=[run_dropdown, day_dropdown, agent_dropdown, relationship_metric],
            outputs=[
                overview,
                timeline,
                action_plot,
                resource_plot,
                behavior_plot,
                propagation_table,
                day_dropdown,
                event_dropdown,
                resident_profile,
                relationship_plot,
                incidents,
                details,
                agent_dropdown,
            ],
        )
        run_dropdown.change(
            fn=load_showcase,
            inputs=[run_dropdown, day_dropdown, agent_dropdown, relationship_metric],
            outputs=[
                overview,
                timeline,
                action_plot,
                resource_plot,
                behavior_plot,
                propagation_table,
                day_dropdown,
                event_dropdown,
                resident_profile,
                relationship_plot,
                incidents,
                details,
                agent_dropdown,
            ],
        )
        day_dropdown.change(
            fn=load_showcase,
            inputs=[run_dropdown, day_dropdown, agent_dropdown, relationship_metric],
            outputs=[
                overview,
                timeline,
                action_plot,
                resource_plot,
                behavior_plot,
                propagation_table,
                day_dropdown,
                event_dropdown,
                resident_profile,
                relationship_plot,
                incidents,
                details,
                agent_dropdown,
            ],
        )
        agent_dropdown.change(
            fn=load_showcase,
            inputs=[run_dropdown, day_dropdown, agent_dropdown, relationship_metric],
            outputs=[
                overview,
                timeline,
                action_plot,
                resource_plot,
                behavior_plot,
                propagation_table,
                day_dropdown,
                event_dropdown,
                resident_profile,
                relationship_plot,
                incidents,
                details,
                agent_dropdown,
            ],
        )
        relationship_metric.change(
            fn=load_showcase,
            inputs=[run_dropdown, day_dropdown, agent_dropdown, relationship_metric],
            outputs=[
                overview,
                timeline,
                action_plot,
                resource_plot,
                behavior_plot,
                propagation_table,
                day_dropdown,
                event_dropdown,
                resident_profile,
                relationship_plot,
                incidents,
                details,
                agent_dropdown,
            ],
        )
        event_dropdown.change(
            fn=load_event_detail,
            inputs=[run_dropdown, event_dropdown],
            outputs=[details, review_notes, review_status],
        )
        save_review.click(
            fn=save_human_review,
            inputs=[run_dropdown, event_dropdown, review_status, review_notes],
            outputs=[review_result],
        )
        app.load(
            fn=load_showcase,
            inputs=[run_dropdown, day_dropdown, agent_dropdown, relationship_metric],
            outputs=[
                overview,
                timeline,
                action_plot,
                resource_plot,
                behavior_plot,
                propagation_table,
                day_dropdown,
                event_dropdown,
                resident_profile,
                relationship_plot,
                incidents,
                details,
                agent_dropdown,
            ],
        )
    return app


if __name__ == "__main__":
    build_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
    )
