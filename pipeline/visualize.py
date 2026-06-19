from __future__ import annotations

from html import escape
from pathlib import Path
import queue
import time
from typing import Any


def _polyline_points(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{point[0]:.2f},{point[1]:.2f}" for point in points)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_strategy_panel(strategy: str, state: dict[str, Any]) -> str:
    render = state.get("render")
    metrics_html = ""
    svg_html = '<div class="placeholder">Waiting for first rollout...</div>'
    if render:
        width, height = render["canvas_size"]
        track_points = _polyline_points(render["track_polyline"])
        trajectory_points = _polyline_points(render["trajectory"])
        car_x, car_y = render["car_position"]
        svg_html = f"""
<svg viewBox="0 0 {width} {height}" class="track" role="img" aria-label="{escape(strategy)} rollout">
  <rect width="100%" height="100%" fill="#141418" />
  <polyline points="{track_points}" fill="none" stroke="#444444" stroke-width="{render["track_half_width"] * 2:.2f}" stroke-linecap="round" stroke-linejoin="round" />
  <polyline points="{track_points}" fill="none" stroke="#d8d8d8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  <polyline points="{trajectory_points}" fill="none" stroke="#00dc78" stroke-width="5" stroke-linecap="round" stroke-linejoin="round" />
  <circle cx="{car_x:.2f}" cy="{car_y:.2f}" r="10" fill="#ff5a5a" />
</svg>
"""
        metrics = render["metrics"]
        metrics_html = f"""
<div class="stats">
  <div>Round best fitness: <strong>{state["best_training_fitness"]:.2f}</strong></div>
  <div>Current round: <strong>{state["generation"]}/{state["completed_generations"]}</strong></div>
  <div>Seed attempt: <strong>{state.get("attempt", 1)}</strong></div>
  <div>Best validation gen: <strong>{state["best_validation_generation"]}</strong></div>
  <div>Validation progress: <strong>{state["validation_summary"]["avg_max_track_progress"]:.3f}</strong></div>
  <div>Validation finish time: <strong>{_format_value(state["validation_summary"]["avg_finish_time"])}</strong></div>
  <div>Replay progress: <strong>{metrics["max_track_progress"]:.3f}</strong></div>
  <div>Replay finish time: <strong>{_format_value(metrics["finish_time"])}</strong></div>
  <div>Replay collisions: <strong>{metrics["collision_count"]}</strong></div>
</div>
"""

    return f"""
<section class="panel">
  <div class="header">
    <h2>{escape(strategy)}</h2>
    <div class="sub">Showing the best car from the current training round on the map</div>
  </div>
  {svg_html}
  {metrics_html}
</section>
"""


def _write_dashboard(
    dashboard_path: Path,
    run_name: str,
    states: dict[str, dict[str, Any]],
    strategies: list[str],
    finished: int,
) -> None:
    panels = "\n".join(_render_strategy_panel(strategy, states[strategy]) for strategy in strategies)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="1">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(run_name)} dashboard</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0f1115;
      --panel: #171a21;
      --text: #f2f4f8;
      --muted: #9aa4b2;
      --accent: #00dc78;
      --border: #242937;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: radial-gradient(circle at top, #1a2230, var(--bg) 55%);
      color: var(--text);
      font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0; font-size: 24px; }}
    .meta {{ color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .panel {{
      background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 18px 40px rgba(0,0,0,0.25);
    }}
    .header h2 {{ margin: 0 0 4px; font-size: 18px; }}
    .sub {{ color: var(--muted); margin-bottom: 12px; }}
    .track {{
      width: 100%;
      aspect-ratio: 16 / 10;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #141418;
      display: block;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 16px;
      margin-top: 12px;
    }}
    .placeholder {{
      display: grid;
      place-items: center;
      min-height: 240px;
      border: 1px dashed var(--border);
      border-radius: 12px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <div class="top">
    <div>
      <h1>{escape(run_name)} training dashboard</h1>
      <div class="meta">Panels update every generation. Refresh is automatic while training is running.</div>
    </div>
    <div class="meta">Completed strategies: {finished}/{len(strategies)}</div>
  </div>
  <div class="grid">
    {panels}
  </div>
</body>
</html>
"""
    dashboard_path.write_text(html, encoding="utf-8")


def run_dashboard(
    run_name: str,
    dashboard_path: Path,
    strategies: list[str],
    progress_queue: Any,
    processes: list[Any],
) -> list[dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {
        strategy: {
            "generation": 0,
            "attempt": 1,
            "completed_generations": 0,
            "render": None,
            "validation_summary": {"avg_max_track_progress": 0.0, "avg_finish_time": None},
            "best_training_fitness": 0.0,
            "best_validation_generation": 0,
        }
        for strategy in strategies
    }
    results: list[dict[str, Any]] = []
    finished = 0

    _write_dashboard(dashboard_path, run_name, states, strategies, finished)

    while True:
        drained = False
        while True:
            try:
                message = progress_queue.get_nowait()
            except queue.Empty:
                break
            drained = True
            if message["type"] == "progress":
                states[message["strategy_name"]] = message
            elif message["type"] == "result":
                results.append(message["result"])
                finished += 1

        if drained:
            _write_dashboard(dashboard_path, run_name, states, strategies, finished)

        if finished >= len(strategies) and not any(process.is_alive() for process in processes):
            break
        time.sleep(0.25)

    _write_dashboard(dashboard_path, run_name, states, strategies, finished)
    results.sort(key=lambda item: item["strategy_name"])
    return results
