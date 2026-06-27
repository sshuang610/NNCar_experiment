from __future__ import annotations

from dataclasses import dataclass
import math
import random


Point = tuple[float, float]


def _move(point: Point, dx: float, dy: float) -> Point:
    return point[0] + dx, point[1] + dy


@dataclass
class Track:
    seed: int
    polyline: list[Point]
    total_length: float
    start_position: Point
    start_angle: float
    cell_size: int
    half_width: float
    canvas_size: tuple[int, int]

    def point_at_progress(self, progress: float) -> Point:
        progress = max(0.0, min(progress, self.total_length))
        remaining = progress
        for start, end in zip(self.polyline[:-1], self.polyline[1:]):
            seg_len = math.dist(start, end)
            if remaining <= seg_len:
                if seg_len == 0:
                    return start
                ratio = remaining / seg_len
                return (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            remaining -= seg_len
        return self.polyline[-1]

    def heading_at_progress(self, progress: float) -> float:
        progress = max(0.0, min(progress, self.total_length))
        remaining = progress
        for start, end in zip(self.polyline[:-1], self.polyline[1:]):
            seg_len = math.dist(start, end)
            if remaining <= seg_len:
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                return (-math.degrees(math.atan2(dx, dy))) % 360
            remaining -= seg_len
        start, end = self.polyline[-2], self.polyline[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        return (-math.degrees(math.atan2(dx, dy))) % 360

    def project(self, point: Point) -> tuple[float, float]:
        best_progress = 0.0
        best_distance = float("inf")
        traversed = 0.0
        for start, end in zip(self.polyline[:-1], self.polyline[1:]):
            vx = end[0] - start[0]
            vy = end[1] - start[1]
            seg_len_sq = (vx * vx) + (vy * vy)
            seg_len = math.sqrt(seg_len_sq)
            if seg_len_sq == 0:
                continue
            ux = point[0] - start[0]
            uy = point[1] - start[1]
            t = max(0.0, min(1.0, ((ux * vx) + (uy * vy)) / seg_len_sq))
            proj = (start[0] + (vx * t), start[1] + (vy * t))
            dist = math.dist(point, proj)
            if dist < best_distance:
                best_distance = dist
                best_progress = traversed + (seg_len * t)
            traversed += seg_len
        return best_progress, best_distance

    def _segment_index(self):
        cached = getattr(self, "_seg_index_cache", None)
        if cached is not None:
            return cached
        bucket = float(self.cell_size) if self.cell_size > 0 else 1.0
        segments = list(zip(self.polyline[:-1], self.polyline[1:]))
        grid: dict[tuple[int, int], list[int]] = {}
        hw = self.half_width
        for idx, (s, e) in enumerate(segments):
            min_x = min(s[0], e[0]) - hw
            max_x = max(s[0], e[0]) + hw
            min_y = min(s[1], e[1]) - hw
            max_y = max(s[1], e[1]) + hw
            for cx in range(int(math.floor(min_x / bucket)), int(math.floor(max_x / bucket)) + 1):
                for cy in range(int(math.floor(min_y / bucket)), int(math.floor(max_y / bucket)) + 1):
                    grid.setdefault((cx, cy), []).append(idx)
        cached = (grid, bucket, segments)
        self._seg_index_cache = cached
        return cached

    def is_on_track(self, point: Point) -> bool:
        grid, bucket, segments = self._segment_index()
        px, py = point
        hw_sq = self.half_width * self.half_width
        cell = (int(math.floor(px / bucket)), int(math.floor(py / bucket)))
        candidates = grid.get(cell)
        if not candidates:
            return False
        for idx in candidates:
            start, end = segments[idx]
            vx = end[0] - start[0]
            vy = end[1] - start[1]
            seg_len_sq = (vx * vx) + (vy * vy)
            if seg_len_sq == 0.0:
                dx = px - start[0]
                dy = py - start[1]
                if (dx * dx) + (dy * dy) <= hw_sq:
                    return True
                continue
            ux = px - start[0]
            uy = py - start[1]
            t = max(0.0, min(1.0, ((ux * vx) + (uy * vy)) / seg_len_sq))
            cx = start[0] + (vx * t)
            cy = start[1] + (vy * t)
            dx = px - cx
            dy = py - cy
            if (dx * dx) + (dy * dy) <= hw_sq:
                return True
        return False


@dataclass
class Cell:
    x: int
    y: int
    walls: dict[str, bool]

    def has_all_walls(self) -> bool:
        return all(self.walls.values())

    def knock_down_wall(self, other: "Cell", wall: str) -> None:
        wall_pairs = {"N": "S", "S": "N", "E": "W", "W": "E"}
        self.walls[wall] = False
        other.walls[wall_pairs[wall]] = False


class Maze:
    def __init__(self, nx: int, ny: int) -> None:
        self.nx = nx
        self.ny = ny
        self.cells = [
            [Cell(x=x, y=y, walls={"N": True, "S": True, "E": True, "W": True}) for y in range(ny)]
            for x in range(nx)
        ]

    def cell_at(self, x: int, y: int) -> Cell:
        return self.cells[x][y]

    def valid_neighbours(self, cell: Cell) -> list[tuple[str, Cell]]:
        delta = [("W", (-1, 0)), ("E", (1, 0)), ("S", (0, 1)), ("N", (0, -1))]
        neighbours: list[tuple[str, Cell]] = []
        for direction, (dx, dy) in delta:
            x2 = cell.x + dx
            y2 = cell.y + dy
            if 0 <= x2 < self.nx and 0 <= y2 < self.ny:
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours


def generate_track(
    seed: int,
    cell_size: int = 120,
    half_width: float = 34.0,
    grid_width: int = 10,
    grid_height: int = 5,
) -> Track:
    rng = random.Random(seed)
    maze = Maze(grid_width, grid_height)
    start = maze.cell_at(0, 3)
    current = start
    ordered_cells = [(current.x, current.y)]
    track_length = 1

    while True:
        neighbours = maze.valid_neighbours(current)
        if neighbours:
            if current.x == 0 and current.y == 3:
                next_cell = maze.cell_at(0, 2)
                current.knock_down_wall(next_cell, "N")
                current = next_cell
                ordered_cells.append((current.x, current.y))
                track_length += 1
                continue

            direction, next_cell = rng.choice(neighbours)
            current.knock_down_wall(next_cell, direction)
            current = next_cell
            ordered_cells.append((current.x, current.y))
            track_length += 1
            continue

        if current.x == 0 and current.y == 4 and track_length > 40:
            break

        track_length = 1
        maze = Maze(grid_width, grid_height)
        for forced_x in range(3, 7):
            maze.cell_at(forced_x, 3).walls["N"] = False
        start = maze.cell_at(0, 3)
        current = start
        ordered_cells = [(current.x, current.y)]

    margin_x = 70
    margin_y = 85
    polyline: list[Point] = []
    for cell_x, cell_y in ordered_cells:
        polyline.append(
            (
                margin_x + (cell_x * cell_size) + (cell_size / 2),
                margin_y + (cell_y * cell_size) + (cell_size / 2),
            )
        )

    total_length = 0.0
    for start_point, end_point in zip(polyline[:-1], polyline[1:]):
        total_length += math.dist(start_point, end_point)

    start_position = polyline[0]
    next_position = polyline[1]
    dx = next_position[0] - start_position[0]
    dy = next_position[1] - start_position[1]
    start_angle = (-math.degrees(math.atan2(dx, dy))) % 360
    canvas_width = (grid_width * cell_size) + (margin_x * 2)
    canvas_height = (grid_height * cell_size) + (margin_y * 2)
    return Track(
        seed=seed,
        polyline=polyline,
        total_length=total_length,
        start_position=start_position,
        start_angle=start_angle,
        cell_size=cell_size,
        half_width=half_width,
        canvas_size=(canvas_width, canvas_height),
    )
