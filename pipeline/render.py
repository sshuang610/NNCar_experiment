from __future__ import annotations

from pathlib import Path

from .track import Point, Track


def _polyline_points(points: list[Point]) -> str:
    return " ".join(f"{point[0]:.2f},{point[1]:.2f}" for point in points)


def write_replay_svg(
    track: Track,
    trajectory: list[Point],
    output_path: str | Path,
    car_position: Point | None = None,
) -> None:
    width, height = track.canvas_size
    path_points = _polyline_points(track.polyline)
    trajectory_points = _polyline_points(trajectory) if trajectory else ""
    car_circle = ""
    if car_position is not None:
        car_circle = (
            f'<circle cx="{car_position[0]:.2f}" cy="{car_position[1]:.2f}" '
            'r="8" fill="#ff5050" />'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#141418" />
  <polyline points="{path_points}" fill="none" stroke="#444444" stroke-width="{track.half_width * 2:.2f}" stroke-linecap="round" stroke-linejoin="round" />
  <polyline points="{path_points}" fill="none" stroke="#d8d8d8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  <polyline points="{trajectory_points}" fill="none" stroke="#00dc78" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
  {car_circle}
</svg>
"""
    Path(output_path).write_text(svg, encoding="utf-8")
