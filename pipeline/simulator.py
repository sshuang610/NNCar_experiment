from __future__ import annotations

from dataclasses import dataclass
import math

from .fitness import FitnessStrategy, StepContext
from .nn import NeuralNetwork
from .track import Point, Track


def _move(point: Point, angle: float, unit: float) -> Point:
    rad = math.radians(-angle % 360)
    return point[0] + (unit * math.sin(rad)), point[1] + (unit * math.cos(rad))


def _rotate(origin: Point, point: Point, angle_radians: float) -> Point:
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle_radians) * (px - ox) - math.sin(angle_radians) * (py - oy)
    qy = oy + math.sin(angle_radians) * (px - ox) + math.cos(angle_radians) * (py - oy)
    return qx, qy


def _angle_delta(angle_a: float, angle_b: float) -> float:
    return ((angle_a - angle_b + 180.0) % 360.0) - 180.0


@dataclass
class EpisodeMetrics:
    finished_within_30s: bool
    finish_time: float | None
    max_track_progress: float
    collision_count: int
    stall_time: float
    spin_time: float
    reset_count: int
    training_fitness: float
    frames: int


@dataclass
class EpisodeResult:
    metrics: EpisodeMetrics
    trajectory: list[Point]


class SimCar:
    def __init__(self, track: Track) -> None:
        self.track = track
        self.x, self.y = track.start_position
        self.angle = track.start_angle
        self.velocity = 0.0
        self.acceleration = 0.0
        self.width = 17.0
        self.height = 35.0
        self.showlines = False
        self.collided = False
        self.maxspeed = 10.0
        self.last_progress = 0.0
        self.progress = 0.0
        self.center_offset = 0.0
        self.sensor_points: list[Point] = []
        self.sensor_distances: list[float] = []
        self.corners: list[Point] = []
        self.update_geometry()

    def update_geometry(self) -> None:
        d = (self.x - (self.width / 2), self.y - (self.height / 2))
        c = (self.x + (self.width / 2), self.y - (self.height / 2))
        b = (self.x + (self.width / 2), self.y + (self.height / 2))
        a = (self.x - (self.width / 2), self.y + (self.height / 2))
        angle_radians = math.radians(self.angle)
        self.corners = [
            _rotate((self.x, self.y), vertex, angle_radians) for vertex in [a, b, c, d]
        ]
        self.sensor_points = []
        self.sensor_distances = []
        for sensor_angle in (0, 45, -45, 90, -90):
            point = _move((self.x, self.y), self.angle + sensor_angle, 10)
            for _ in range(1000):
                if not self.track.is_on_track(point):
                    point = _move(point, self.angle + sensor_angle, -1)
                    break
                point = _move(point, self.angle + sensor_angle, 4)
            self.sensor_points.append(point)
            self.sensor_distances.append(math.dist((self.x, self.y), point))
        self.progress, self.center_offset = self.track.project((self.x, self.y))

    def reset_position(self) -> None:
        self.x, self.y = self.track.start_position
        self.angle = self.track.start_angle
        self.velocity = 0.0
        self.acceleration = 0.0
        self.collided = False
        self.progress = 0.0
        self.last_progress = 0.0
        self.update_geometry()

    def set_accel(self, accel: float) -> None:
        self.acceleration = accel

    def rotate(self, rot: float) -> None:
        self.angle += rot
        if self.angle > 360:
            self.angle = 0
        if self.angle < 0:
            self.angle = 360 + self.angle

    def feedforward(self, network: NeuralNetwork) -> list[float]:
        inputs = [*self.sensor_distances, self.velocity]
        outputs = network.feedforward(inputs)
        return [outputs.item(idx) for idx in range(outputs.shape[0])]

    def take_action(self, outputs: list[float]) -> None:
        if outputs[0] > 0.5:
            self.set_accel(0.2)
        else:
            self.set_accel(0.0)
        if outputs[1] > 0.5:
            self.set_accel(-0.2)
        if outputs[2] > 0.5:
            self.rotate(-5)
        if outputs[3] > 0.5:
            self.rotate(5)

    def update(self) -> None:
        self.last_progress = self.progress
        if self.acceleration != 0:
            self.velocity += self.acceleration
            if self.velocity > self.maxspeed:
                self.velocity = self.maxspeed
            elif self.velocity < 0:
                self.velocity = 0
        else:
            self.velocity *= 0.92
        self.x, self.y = _move((self.x, self.y), self.angle, self.velocity)
        self.update_geometry()

    def collision(self) -> bool:
        return any(not self.track.is_on_track(point) for point in self.corners)


class Simulator:
    def __init__(self, track: Track, fps: int, time_limit_seconds: float) -> None:
        self.track = track
        self.fps = fps
        self.time_limit_seconds = time_limit_seconds

    def run_episode(self, network: NeuralNetwork, strategy: FitnessStrategy) -> EpisodeResult:
        strategy.reset()
        car = SimCar(self.track)
        dt = 1.0 / self.fps
        max_frames = int(self.fps * self.time_limit_seconds)
        trajectory: list[Point] = [(car.x, car.y)]
        training_fitness = 0.0
        collision_count = 0
        stall_time = 0.0
        spin_time = 0.0
        max_progress = 0.0
        finished = False
        finish_time: float | None = None

        for frame_idx in range(max_frames):
            outputs = car.feedforward(network)
            previous_angle = car.angle
            car.take_action(outputs)
            car.update()
            collided = car.collision()
            if collided:
                car.collided = True
                collision_count += 1

            progress_delta = max(0.0, car.progress - car.last_progress)
            max_progress = max(max_progress, car.progress)
            progress_ratio = car.progress / self.track.total_length if self.track.total_length else 0.0
            target_heading = self.track.heading_at_progress(car.progress)
            heading_delta = _angle_delta(car.angle, target_heading)
            heading_alignment = math.cos(math.radians(heading_delta))
            front_clearance = car.sensor_distances[0] if car.sensor_distances else 0.0
            min_clearance = min(car.sensor_distances) if car.sensor_distances else 0.0
            right_clearance = car.sensor_distances[3] if len(car.sensor_distances) > 3 else 0.0
            left_clearance = car.sensor_distances[4] if len(car.sensor_distances) > 4 else 0.0
            side_clearance_balance = abs(right_clearance - left_clearance)
            turn_amount = abs(_angle_delta(car.angle, previous_angle))
            finished_now = car.progress >= self.track.total_length
            is_stalled = car.velocity < 0.5
            is_spinning = abs(car.angle - previous_angle) >= 5 and progress_delta < 0.1
            if is_stalled:
                stall_time += dt
            if is_spinning:
                spin_time += dt

            training_fitness += strategy.score_step(
                StepContext(
                    velocity=car.velocity,
                    progress_delta=progress_delta,
                    progress_ratio=progress_ratio,
                    center_offset=car.center_offset,
                    normalized_center_offset=car.center_offset / self.track.half_width,
                    heading_alignment=heading_alignment,
                    front_clearance=front_clearance,
                    min_clearance=min_clearance,
                    side_clearance_balance=side_clearance_balance,
                    turn_amount=turn_amount,
                    collided=collided,
                    finished=finished_now,
                    is_stalled=is_stalled,
                    is_spinning=is_spinning,
                    frame=frame_idx + 1,
                    time_elapsed=(frame_idx + 1) * dt,
                )
            )
            trajectory.append((car.x, car.y))

            if collided:
                break
            if finished_now:
                finished = True
                finish_time = (frame_idx + 1) * dt
                break

        metrics = EpisodeMetrics(
            finished_within_30s=finished and finish_time is not None and finish_time <= self.time_limit_seconds,
            finish_time=finish_time,
            max_track_progress=max_progress / self.track.total_length if self.track.total_length else 0.0,
            collision_count=collision_count,
            stall_time=stall_time,
            spin_time=spin_time,
            reset_count=0,
            training_fitness=training_fitness,
            frames=len(trajectory) - 1,
        )
        return EpisodeResult(metrics=metrics, trajectory=trajectory)
