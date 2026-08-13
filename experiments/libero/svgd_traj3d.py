#!/usr/bin/env python
"""Interactive 3D viewer for how SVGD rollout trajectories evolve across iterations.

Serves a small local UI that indexes every trial under a runs root and, for the
selected trial, animates the per-iteration end-effector rollouts in 3D: the whole
particle population, a single particle, or the running best particle.

Two optimizer families are supported and normalised into one bundle format:

* endpoint search (``svgd_endpoint.py``) -- particles are 3D target endpoints,
  stored in ``particles_before_update`` / ``terminal_eefs``;
* path search (``svgd_obstacle_path.py``) -- particles are path parameters, so the
  3D quantities are recovered from the saved rollout traces instead.

Scene landmarks (table, robot base, arm links, tracked objects) come from a
``scene.json`` written next to a run by ``scene_geometry.capture_scene``; runs
recorded before that exists still load, just without landmarks.

Usage::

    python experiments/libero/svgd_traj3d.py --runs-root runs --port 8000
    python experiments/libero/svgd_traj3d.py --list
    python experiments/libero/svgd_traj3d.py --verify
    python experiments/libero/svgd_traj3d.py --export-bundle runs/.../token_cosine_y040 \
        --out bundle.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import socket
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

VIEWER_HTML = Path(__file__).with_name("svgd_traj3d_viewer.html")
COORD_DECIMALS = 5  # 10 micrometres -- well below the ~1 mm tracking error
METRIC_DECIMALS = 6


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #


@dataclass
class TrialInfo:
    """One optimizer run that can be visualised."""

    key: str  # runs-root-relative posix path of the trial directory
    scene: str  # top-level run directory (grouping label in the UI)
    label: str  # remainder of the path, shown in the picker
    kind: str  # "endpoint" | "path"
    objective: str | None
    particles: int
    iterations: int
    traced_iterations: int
    modified: float
    source: str = "run"  # "run" = live trial directory, "bundle" = exported JSON


BUNDLE_SUFFIX = ".traj3d.json"


def _history_payload(payload: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return metadata and records from wrapped or legacy list histories."""
    metadata = payload if isinstance(payload, dict) else {}
    records = payload if isinstance(payload, list) else metadata.get("history")
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        return metadata, []
    return metadata, records


def _summarize(history_path: Path, runs_root: Path) -> TrialInfo | None:
    try:
        raw_payload = json.loads(history_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload, records = _history_payload(raw_payload)
    if not records:
        return None

    particles = max(len(record.get("energies") or []) for record in records)
    if particles == 0:
        return None

    traced = sum(1 for record in records if any(_trace_names(record)))
    if traced == 0:
        return None

    kind = "endpoint" if "particles_before_update" in records[0] else "path"
    config = payload.get("config") or {}
    trial_dir = history_path.parent
    relative = trial_dir.relative_to(runs_root).as_posix()
    scene, _, remainder = relative.partition("/")
    return TrialInfo(
        key=relative,
        scene=scene,
        label=remainder or relative,
        kind=kind,
        objective=config.get("latent_distance") or records[0].get("objective"),
        particles=particles,
        iterations=len(records),
        traced_iterations=traced,
        modified=history_path.stat().st_mtime,
    )


def _summarize_bundle(bundle_path: Path, runs_root: Path) -> TrialInfo | None:
    """Index a pre-exported bundle -- the shareable form of a trial."""
    try:
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    frames = payload.get("frames") or []
    if not frames or not payload.get("particles"):
        return None
    relative = bundle_path.relative_to(runs_root).as_posix()[: -len(BUNDLE_SUFFIX)]
    scene, _, remainder = relative.partition("/")
    return TrialInfo(
        key=bundle_path.relative_to(runs_root).as_posix(),
        scene=scene,
        label=remainder or relative,
        kind=payload.get("kind", "endpoint"),
        objective=payload.get("objective"),
        particles=int(payload["particles"]),
        iterations=len(frames),
        traced_iterations=sum(1 for f in frames if any(f.get("paths") or [])),
        modified=bundle_path.stat().st_mtime,
        source="bundle",
    )


def discover_trials(runs_root: Path) -> list[TrialInfo]:
    """Index every trial under ``runs_root``: live run directories and exported bundles."""
    found = [_summarize(path, runs_root) for path in sorted(runs_root.rglob("history.json"))]
    found += [
        _summarize_bundle(path, runs_root)
        for path in sorted(runs_root.rglob("*" + BUNDLE_SUFFIX))
    ]
    trials = [info for info in found if info is not None]
    trials.sort(key=lambda info: (info.scene, info.label))
    return trials


# --------------------------------------------------------------------------- #
# bundle construction
# --------------------------------------------------------------------------- #


def _trace_names(record: dict[str, Any]) -> list[Any]:
    names = record.get("rollout_trace_files")
    if names is None:
        names = record.get("trace_files")
    return list(names or [])


def _round(value: Any, decimals: int) -> Any:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        array = np.where(np.isfinite(array), array, np.nan)
    return np.round(array, decimals).tolist()


def _opt_round(value: Any, decimals: int) -> Any:
    return None if value is None else _round(value, decimals)


def _apex(path: list[list[float]] | None) -> list[float] | None:
    """Point of a rollout furthest from its own start->end chord.

    Path-search particles all share one fixed endpoint, so the endpoint says
    nothing about how they differ; the arc apex is the parameter they actually
    move.
    """
    if not path or len(path) < 3:
        return None
    points = np.asarray(path, dtype=np.float64)
    chord = points[-1] - points[0]
    norm = float(np.linalg.norm(chord))
    if norm < 1e-9:
        return None
    offsets = points - points[0]
    projected = np.outer(offsets @ (chord / norm), chord / norm)
    return _round(points[int(np.argmax(np.linalg.norm(offsets - projected, axis=1)))],
                  COORD_DECIMALS)


def _metric_column(record: dict[str, Any], name: str) -> list[float] | None:
    metrics = record.get("latent_metrics") or {}
    column = metrics.get(name)
    if isinstance(column, list):
        return _round(column, METRIC_DECIMALS)
    return None


def _keep_indices(count: int, stride: int) -> list[int]:
    """Sample indices at ``stride``, always keeping the first and last state."""
    if stride <= 1 or count <= 2:
        return list(range(count))
    keep = list(range(0, count - 1, stride))
    if keep[-1] != count - 1:
        keep.append(count - 1)
    return keep


def _read_trace(path: Path, stride: int, arm_stride: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as trace:
        full = np.asarray(trace["eef_path"], dtype=np.float64)
        keep = _keep_indices(len(full), stride)
        entry: dict[str, Any] = {
            "path": _round(full[keep], COORD_DECIMALS),
            "path_steps": keep,
            "steps": int(len(full)),
        }
        for key, out in (("target_eef", "target"), ("terminal_eef", "terminal")):
            if key in trace:
                entry[out] = _round(trace[key], COORD_DECIMALS)
        entry.setdefault("terminal", _round(full[-1], COORD_DECIMALS))

        # Objects move while the arm does; the whole track is what shows whether a
        # rollout disturbed them, not just its endpoints.
        objects = np.asarray(trace.get("object_positions", np.zeros((0, 0, 3))))
        if objects.ndim == 3 and objects.shape[1] > 0:
            tracks = objects[keep] if len(objects) == len(full) else objects
            entry["objects"] = _round(np.swapaxes(tracks, 0, 1), COORD_DECIMALS)
            entry["object_start"] = _round(objects[0], COORD_DECIMALS)
            entry["object_end"] = _round(objects[-1], COORD_DECIMALS)

        # Arm skeletons are the heaviest payload by far, so they get their own
        # (coarser) stride: enough poses to read the motion, not every step.
        links = np.asarray(trace.get("arm_link_positions", np.zeros((0, 0, 3))))
        if arm_stride > 0 and links.ndim == 3 and links.shape[1] > 0:
            arm_keep = _keep_indices(len(links), max(arm_stride, 1))
            entry["arm"] = _round(links[arm_keep], COORD_DECIMALS)
            entry["arm_steps"] = arm_keep
    return entry


def _load_scene(trial_dir: Path) -> dict[str, Any] | None:
    """Nearest ``scene.json`` at or above the trial -- the run's static landmarks.

    Trials nest under a shared run directory (``<run>/trials/<name>``) and every
    trial in a run shares one simulator scene, so a capture written anywhere on
    the way up applies.
    """
    for directory in (trial_dir, *trial_dir.parents[:4]):
        candidate = directory / "scene.json"
        if candidate.is_file():
            try:
                scene = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scene["source"] = candidate.name
            return scene
    return None


def _manifest_goal(trial_dir: Path) -> list[float] | None:
    """``physical_goal_eef`` from the nearest test manifest.

    Path-search runs written before the goal was recorded in ``history.json``
    would otherwise show no goal at all, which is exactly the case where the plot
    is hardest to read.
    """
    for directory in trial_dir.parents[:4]:
        candidate = directory / "manifest.json"
        if not candidate.is_file():
            continue
        try:
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        goal = manifest.get("physical_goal_eef")
        if goal is not None:
            return list(goal)
    return None


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _frame_images(
    trial_dir: Path, iteration: int, particles: int, record: dict[str, Any]
) -> dict[str, Any] | None:
    """Terminal renders written for one iteration, by the shared file convention.

    Every optimizer here saves ``iter_XXX/particle_NN.png`` (the composed
    ``[agentview | wrist]`` view that was actually scored) plus ``best.png``.
    Paths are relative to the trial so the viewer can request them by key.
    """
    directory = f"iter_{iteration:03d}"
    found = [
        name if (trial_dir / (name := f"{directory}/particle_{index:02d}.png")).is_file()
        else None
        for index in range(particles)
    ]
    best = record.get("best_image") or f"{directory}/best.png"
    if not (trial_dir / best).is_file():
        best = None
    if not any(found) and best is None:
        return None
    return {"particles": found, "best": best}


def _reference_image(trial_dir: Path) -> str | None:
    """The goal render every particle in the trial is being scored against."""
    for name in ("goal_reference.png", "goal_latent_decoded.png", "goal.png"):
        if (trial_dir / name).is_file():
            return name
    return None


def build_bundle(trial_dir: Path, stride: int = 1, arm_stride: int = 8) -> dict[str, Any]:
    """Normalise one trial into the JSON payload the viewer consumes."""
    history_path = trial_dir / "history.json"
    raw_payload = json.loads(history_path.read_text(encoding="utf-8"))
    payload, records = _history_payload(raw_payload)
    if not records:
        raise ValueError(f"{history_path} has no iterations")

    config = payload.get("config") or {}
    objective = config.get("latent_distance") or records[0].get("objective")
    kind = "endpoint" if "particles_before_update" in records[0] else "path"
    particles = max(len(record.get("energies") or []) for record in records)

    object_names: list[str] = []
    arm_link_names: list[str] = []
    rollout_steps = 0
    path_steps: list[int] = []
    arm_steps: list[int] = []
    frames: list[dict[str, Any]] = []
    for record in records:
        energies = record.get("energies") or []
        targets = record.get("particles_before_update")
        terminals = record.get("terminal_eefs")
        names = _trace_names(record)

        paths: list[Any] = []
        derived_targets: list[Any] = []
        derived_terminals: list[Any] = []
        object_starts: list[Any] = []
        object_ends: list[Any] = []
        object_tracks: list[Any] = []
        arms: list[Any] = []
        for index in range(len(energies)):
            name = names[index] if index < len(names) else None
            trace = _read_trace(trial_dir / name, stride, arm_stride) if name else None
            paths.append(trace["path"] if trace else None)
            derived_targets.append(trace.get("target") if trace else None)
            derived_terminals.append(trace.get("terminal") if trace else None)
            object_starts.append(trace.get("object_start") if trace else None)
            object_ends.append(trace.get("object_end") if trace else None)
            object_tracks.append(trace.get("objects") if trace else None)
            arms.append(trace.get("arm") if trace else None)
            if trace:
                rollout_steps = max(rollout_steps, trace["steps"])
                path_steps = path_steps or trace["path_steps"]
                arm_steps = arm_steps or trace.get("arm_steps") or []

        frame: dict[str, Any] = {
            "iteration": int(record.get("iteration", len(frames))),
            "energies": _round(energies, METRIC_DECIMALS),
            "targets": _opt_round(targets, COORD_DECIMALS) or derived_targets,
            "terminals": _opt_round(terminals, COORD_DECIMALS) or derived_terminals,
            "paths": paths,
            "markers": [_apex(path) for path in paths] if kind == "path" else None,
            "best_particle": record.get("best_particle"),
            "global_best_energy": record.get("global_best_energy"),
            "global_best_iteration": record.get("global_best_iteration"),
            "global_best_particle": record.get("global_best_particle"),
        }
        for key in ("goal_errors_m", "target_tracking_errors_m", "particle_spread_m"):
            if record.get(key) is not None:
                frame[key] = _round(record[key], METRIC_DECIMALS)
        metrics = {
            name: column
            for name in ("rms", "cosine", "token_cosine")
            if (column := _metric_column(record, name)) is not None
        }
        if metrics:
            frame["metrics"] = metrics
        if any(start is not None for start in object_starts):
            frame["object_starts"] = object_starts
            frame["object_ends"] = object_ends
            frame["object_paths"] = object_tracks
        if any(arm is not None for arm in arms):
            frame["arms"] = arms
        images = _frame_images(trial_dir, frame["iteration"], len(energies), record)
        if images:
            frame["images"] = images["particles"]
            frame["best_image"] = images["best"]
        if (not object_names or not arm_link_names) and any(names):
            first = next(name for name in names if name)
            with np.load(trial_dir / first, allow_pickle=True) as trace:
                object_names = object_names or [
                    str(name) for name in trace.get("object_names", [])
                ]
                arm_link_names = arm_link_names or [
                    str(name) for name in trace.get("arm_link_names", [])
                ]
        frames.append(frame)

    start = payload.get("actual_start_eef")
    if start is None:
        first_path = next(
            (path for frame in frames for path in frame["paths"] if path), None
        )
        start = first_path[0] if first_path else None

    goal = payload.get("diagnostic_goal_eef") or payload.get("manifest_physical_goal_eef")
    goal_source = "history.json"
    if goal is None:
        goal = _manifest_goal(trial_dir)
        goal_source = "test manifest.json" if goal is not None else "none"

    scene = _load_scene(trial_dir)
    bundle = {
        "key": trial_dir.name,
        "kind": kind,
        "marker_label": "path apexes" if kind == "path" else "endpoints",
        "objective": objective,
        "particles": particles,
        "created_at_utc": payload.get("created_at_utc"),
        "start_eef": _opt_round(start, COORD_DECIMALS),
        "goal_eef": _opt_round(goal, COORD_DECIMALS),
        "goal_source": goal_source,
        "target_eef": _opt_round(payload.get("fixed_target_eef"), COORD_DECIMALS),
        "goal_tolerance_m": payload.get("goal_tolerance_m"),
        "goal_is_diagnostic_only": payload.get(
            "diagnostic_goal_is_optimizer_input", False
        )
        is False,
        "bounds": config.get("bounds"),
        "object_names": object_names,
        "scene": scene,
        # Pixels stay on disk and are fetched per view: a run holds hundreds of
        # renders, far more than belongs inside a JSON payload.
        "goal_image": _reference_image(trial_dir),
        "images_available": any(frame.get("images") for frame in frames),
        "rollout": {
            "steps": rollout_steps,
            "path_steps": path_steps,
            "arm_steps": arm_steps,
            "arm_link_names": arm_link_names,
            "arm_link_parents": ((scene or {}).get("arm") or {}).get("link_parents"),
        },
        "config": {
            key: config.get(key)
            for key in (
                "particles",
                "iterations",
                "transport",
                "latent_distance",
                "latent_views",
                "latent_weight",
                "repulsion_weight",
                "step_size",
                "temperature",
                "init_mode",
                "init_radius",
                "max_update_norm",
                "seed",
            )
            if config.get(key) is not None
        },
        "frames": frames,
    }
    bundle["checks"] = _checks(bundle)
    return bundle


# --------------------------------------------------------------------------- #
# scene checks
# --------------------------------------------------------------------------- #


def _distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    first, second = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,):
        return None
    return round(float(np.linalg.norm(first - second)), METRIC_DECIMALS)


def _checks(bundle: dict[str, Any]) -> dict[str, Any]:
    """Answer, in metres, "is the plot showing what I think it is?".

    Every anchor drawn in the viewer is asserted against the rollouts themselves:
    the start against where paths actually begin, each particle's target against
    where its rollout actually ended, and the goal against the best terminal.
    The viewer surfaces these so a mislabelled anchor is visible instead of
    quietly framing the whole scene wrong.
    """
    frames = bundle["frames"]
    first = next((f for f in frames if any(f["paths"])), None)
    last = next((f for f in reversed(frames) if any(f["paths"])), None)
    checks: dict[str, Any] = {
        "goal_source": bundle["goal_source"],
        "start_vs_first_waypoint_m": None,
        "target_tracking_max_m": None,
        "goal_vs_best_terminal_m": None,
        "goal_vs_target_m": _distance(bundle["goal_eef"], bundle["target_eef"]),
        "goal_reachable_by_particles": None,
        "particle_span_m": None,
    }
    if first is not None:
        starts = [path[0] for path in first["paths"] if path]
        checks["start_vs_first_waypoint_m"] = max(
            (_distance(bundle["start_eef"], point) or 0.0) for point in starts
        )
    if last is not None:
        pairs = [
            (target, terminal)
            for target, terminal in zip(last["targets"], last["terminals"])
            if target is not None and terminal is not None
        ]
        errors = [
            distance
            for target, terminal in pairs
            if (distance := _distance(target, terminal)) is not None
        ]
        if errors:
            checks["target_tracking_max_m"] = round(max(errors), METRIC_DECIMALS)
        best = last.get("best_particle")
        if best is not None and best < len(last["terminals"]):
            checks["goal_vs_best_terminal_m"] = _distance(
                bundle["goal_eef"], last["terminals"][best]
            )
        points = np.asarray(
            [point for point in last["terminals"] if point is not None], dtype=np.float64
        )
        if points.size:
            checks["particle_span_m"] = round(
                float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))),
                METRIC_DECIMALS,
            )
    tolerance = bundle.get("goal_tolerance_m") or 0.03
    reached = checks["goal_vs_best_terminal_m"]
    if reached is not None:
        checks["goal_reachable_by_particles"] = bool(reached <= tolerance)
    return checks


# --------------------------------------------------------------------------- #
# http server
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _plotly_js() -> bytes | None:
    """Bundled copy of plotly.min.js, so the viewer works without network access."""
    try:
        import plotly
    except ImportError:
        return None
    candidate = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    return candidate.read_bytes() if candidate.exists() else None


class _ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], runs_root: Path, stride: int, arm_stride: int
    ) -> None:
        super().__init__(address, _ViewerHandler)
        self.runs_root = runs_root
        self.stride = stride
        self.arm_stride = arm_stride
        self.bundle_cache: dict[tuple[str, int, int], tuple[float, bytes]] = {}


class _ViewerHandler(BaseHTTPRequestHandler):
    server: _ViewerServer
    server_version = "svgd-traj3d/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter console
        if not self.path.startswith("/static/"):
            sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    # -- helpers ----------------------------------------------------------- #

    def _send(
        self, body: bytes, content_type: str, *, cache: bool = False, compress: bool = True
    ) -> None:
        headers = [("Content-Type", content_type)]
        if compress and "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 4096:
            body = gzip.compress(body, 6)
            headers.append(("Content-Encoding", "gzip"))
        headers.append(("Cache-Control", "max-age=86400" if cache else "no-store"))
        self.send_response(200)
        for key, value in headers + [("Content-Length", str(len(body)))]:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json")

    def _fail(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve(self, key: str) -> Path:
        """Map a request key to a trial directory or bundle file, never outside the root."""
        root = self.server.runs_root.resolve()
        candidate = (root / key).resolve()
        if not candidate.is_relative_to(root):
            raise FileNotFoundError(key)
        if key.endswith(BUNDLE_SUFFIX) and candidate.is_file():
            return candidate
        if (candidate / "history.json").exists():
            return candidate
        raise FileNotFoundError(key)

    # -- routes ------------------------------------------------------------ #

    def do_GET(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
        route = urlparse(self.path)
        query = parse_qs(route.query)
        try:
            if route.path in ("/", "/index.html"):
                self._send(VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
            elif route.path == "/static/plotly.min.js":
                script = _plotly_js()
                if script is None:
                    self._fail(404, "plotly is not installed in this environment")
                else:
                    self._send(script, "text/javascript", cache=True)
            elif route.path == "/api/trials":
                trials = discover_trials(self.server.runs_root)
                self._send_json(
                    {
                        "runs_root": str(self.server.runs_root),
                        "trials": [asdict(info) for info in trials],
                    }
                )
            elif route.path == "/api/bundle":
                self._send_bundle(query.get("key", [""])[0])
            elif route.path == "/api/image":
                self._send_image(query.get("key", [""])[0], query.get("path", [""])[0])
            else:
                self._fail(404, f"no route {route.path}")
        except FileNotFoundError as error:
            self._fail(404, f"unknown trial: {error}")
        except Exception as error:  # surface loader errors in the UI
            self._fail(500, f"{type(error).__name__}: {error}")

    def _send_image(self, key: str, relative: str) -> None:
        """Serve one render from inside a trial directory, and nothing else.

        The renders are immutable once an iteration is written, so they are
        cacheable -- which is what keeps playback from re-fetching every frame.
        """
        trial = self._resolve(key)
        if trial.is_file():  # an exported bundle carries no pixels
            raise FileNotFoundError(f"{key} is a bundle, not a run directory")
        candidate = (trial / relative).resolve()
        if not candidate.is_relative_to(trial.resolve()):
            raise FileNotFoundError(relative)
        if candidate.suffix.lower() not in IMAGE_SUFFIXES or not candidate.is_file():
            raise FileNotFoundError(relative)
        suffix = candidate.suffix.lower().lstrip(".")
        self._send(
            candidate.read_bytes(),
            f"image/{'jpeg' if suffix == 'jpg' else suffix}",
            cache=True,
            compress=False,  # already-compressed pixels
        )

    def _send_bundle(self, key: str) -> None:
        source = self._resolve(key)
        stride, arm_stride = self.server.stride, self.server.arm_stride
        is_bundle = source.is_file()
        stamp = (source if is_bundle else source / "history.json").stat().st_mtime
        cached = self.server.bundle_cache.get((key, stride, arm_stride))
        if cached is None or cached[0] != stamp:
            if is_bundle:
                bundle = json.loads(source.read_text(encoding="utf-8"))
                # An exported bundle is a lone JSON file; the renders it was
                # built from are not reachable from here.
                bundle["images_available"] = False
            else:
                bundle = build_bundle(source, stride=stride, arm_stride=arm_stride)
            bundle["key"] = key
            payload = json.dumps(bundle).encode("utf-8")
            self.server.bundle_cache[(key, stride, arm_stride)] = (stamp, payload)
            cached = (stamp, payload)
        self._send(cached[1], "application/json")


def serve(runs_root: Path, host: str, port: int, stride: int, arm_stride: int) -> None:
    trials = discover_trials(runs_root)
    server = _ViewerServer((host, port), runs_root, stride, arm_stride)
    shown = host if host not in ("0.0.0.0", "::") else socket.gethostname()
    print(f"runs root : {runs_root}")
    print(f"trials    : {len(trials)} with saved rollout traces")
    print(f"viewer    : http://{shown}:{server.server_address[1]}/")
    if _plotly_js() is None:
        print("warning   : plotly not importable -- viewer will load plotly from the CDN")
    print("Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def verify(runs_root: Path, stride: int = 8) -> None:
    """Print, per trial, whether the anchors the viewer draws match the rollouts.

    ``start`` should coincide with where every path begins, ``track`` (the gap
    between a particle's own target endpoint and where its rollout stopped)
    should be at controller-error scale, and ``goal`` is the distance from the
    final best terminal to the goal marker -- large by design in runs where the
    goal is diagnostic only.
    """
    print(
        f"{'trial':64s} {'start':>8s} {'track':>8s} {'goal':>8s} "
        f"{'span':>8s}  goal source / scene"
    )
    for info in discover_trials(runs_root):
        if info.source != "run":
            continue
        try:
            bundle = build_bundle(runs_root / info.key, stride=stride, arm_stride=0)
        except (OSError, ValueError, KeyError) as error:
            print(f"{info.key[:64]:64s} {type(error).__name__}: {error}")
            continue
        checks, scene = bundle["checks"], bundle.get("scene")
        cell = lambda value: "       -" if value is None else f"{value:8.4f}"  # noqa: E731
        notes = [checks["goal_source"]]
        if scene:
            table = scene.get("table")
            notes.append(f"table z={table['top_z']:.3f}" if table else "scene, no table")
        else:
            notes.append("no scene.json")
        if checks["goal_reachable_by_particles"] is False:
            notes.append("goal not reached")
        print(
            f"{info.key[-64:]:64s} {cell(checks['start_vs_first_waypoint_m'])} "
            f"{cell(checks['target_tracking_max_m'])} "
            f"{cell(checks['goal_vs_best_terminal_m'])} "
            f"{cell(checks['particle_span_m'])}  {' · '.join(notes)}"
        )


def export_all(
    runs_root: Path, out_dir: Path, stride: int = 1, arm_stride: int = 8
) -> None:
    """Export every live trial as ``<key>.traj3d.json`` under ``out_dir``.

    The result is self-contained: point ``--runs-root`` at ``out_dir`` and the
    viewer works with no access to the original traces, which is what makes a
    run small enough to commit.
    """
    total = 0
    for info in discover_trials(runs_root):
        if info.source != "run":
            continue
        out = out_dir / (info.key + BUNDLE_SUFFIX)
        out.parent.mkdir(parents=True, exist_ok=True)
        bundle = build_bundle(runs_root / info.key, stride=stride, arm_stride=arm_stride)
        out.write_text(json.dumps(bundle), encoding="utf-8")
        size = out.stat().st_size
        total += size
        print(f"{size / 1e6:6.2f} MB  {out.relative_to(out_dir)}")
    print(f"{total / 1e6:6.2f} MB  total in {out_dir}")


def _trial_dir(argument: str) -> Path:
    path = Path(argument).expanduser()
    if path.name == "history.json":
        path = path.parent
    if not (path / "history.json").exists():
        raise argparse.ArgumentTypeError(f"{path} does not contain history.json")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="directory scanned recursively for trial history.json files",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="keep every Nth rollout waypoint (1 = full 49-step path)",
    )
    parser.add_argument(
        "--arm-stride",
        type=int,
        default=8,
        help="keep every Nth arm-skeleton pose (0 disables arm data, which is the "
        "bulkiest part of a bundle)",
    )
    parser.add_argument("--list", action="store_true", help="print the trial index and exit")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check every trial's start/target/goal anchors against its rollouts",
    )
    parser.add_argument(
        "--export-all",
        type=Path,
        help="write every discovered run as a shareable bundle under this directory",
    )
    parser.add_argument(
        "--export-bundle",
        type=_trial_dir,
        help="write one trial's viewer bundle to --out instead of serving",
    )
    parser.add_argument("--out", type=Path, help="output path for --export-bundle")
    args = parser.parse_args()

    runs_root = args.runs_root.expanduser().resolve()
    if args.export_all is not None:
        export_all(
            runs_root,
            args.export_all.expanduser(),
            stride=args.stride,
            arm_stride=args.arm_stride,
        )
        return

    if args.export_bundle is not None:
        bundle = build_bundle(
            args.export_bundle, stride=args.stride, arm_stride=args.arm_stride
        )
        out = args.out or args.export_bundle / "traj3d_bundle.json"
        out.write_text(json.dumps(bundle), encoding="utf-8")
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
        return

    if not runs_root.is_dir():
        parser.error(f"--runs-root {runs_root} is not a directory")

    if args.list:
        for info in discover_trials(runs_root):
            print(
                f"{info.scene}/{info.label:60s} {info.kind:8s} "
                f"{info.objective or '-':12s} "
                f"{info.particles:3d}p x {info.iterations:3d}it "
                f"({info.traced_iterations} traced)"
            )
        return

    if args.verify:
        verify(runs_root, stride=max(args.stride, 8))
        return

    serve(runs_root, args.host, args.port, args.stride, args.arm_stride)


if __name__ == "__main__":
    main()
