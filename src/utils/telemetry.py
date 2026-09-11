""" Structured, analysis-ready telemetry for the EAGS-SLAM pipeline.

This module writes CSV logs that answer three questions which the ordinary run
artefacts (``rendering_metrics.json``, ``ate.json``, the submap checkpoints)
cannot answer:

1. ``tracking_init_candidates.csv`` / ``tracking_frames.csv``
   Did the tracker pick the *right* pose candidate, and how much ground-truth
   error was left on the table by picking a different one?  This bounds any
   improvement to the initialization scoring in ``Tracker.init_pose_min_loss``
   (tracker.py:130-169).

2. ``mapping_frames.csv``
   How is the wall-clock inside ``Mapper.optimize_submap`` (mapper.py:143-214)
   split between rendering, loss evaluation, ``backward`` and
   ``optimizer.step``?  This decides whether gradient masking can buy time at
   all, because only ``step`` is affected by a mask.

3. ``gradient_stats.csv``
   Where does the per-Gaussian gradient mass live?  If Gaussians in flat
   regions carry negligible ``||dL/dxyz||``, an explicit update-priority
   mechanism is largely redundant; if they carry large but unhelpful gradients,
   the mechanism is justified.

Design notes
------------
* Everything is gated by a single ``logging.enabled`` config flag, and every
  public method is a no-op when telemetry is disabled or absent.
* Instrumentation must never be able to break a long run: any internal failure
  disables telemetry and prints a single warning instead of propagating.
* Section timings use CUDA events when a GPU is available (with one
  synchronisation per optimizer iteration) and ``time.perf_counter`` otherwise.
  Section timings are therefore accurate, but the absolute wall-clock of an
  instrumented run is somewhat inflated by the extra synchronisation.
  ``t_loop_wall_ms`` is logged so that overhead can be quantified against a
  non-instrumented baseline.
"""
import csv
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CSV schemas.  Columns are declared here so that every writer emits a stable
# header and rows are emitted in header order (missing keys become empty cells).
# ---------------------------------------------------------------------------
CANDIDATES_HEADER = [
    "frame_id", "candidate", "is_chosen", "is_oracle",
    "loss_total", "loss_color", "loss_depth",
    "t_err_m", "r_err_deg",
]

TRACKING_FRAMES_HEADER = [
    "frame_id", "n_candidates", "chosen", "oracle", "choice_is_oracle",
    "chosen_t_err_m", "oracle_t_err_m", "headroom_m",
    "best_loss", "second_best_loss", "loss_margin",
    "fallback_fired", "num_iters",
    "final_t_err_m", "final_r_err_deg", "track_ms",
]

MAPPING_FRAMES_HEADER = [
    "frame_id", "is_new_submap", "max_iterations", "iterations_run",
    "early_stopped", "n_gaussians_before", "n_new_points_candidate",
    "n_added", "n_gaussians_after", "n_pruned_early", "n_pruned_final",
    "t_render_ms", "t_loss_ms", "t_backward_ms", "t_gradstats_ms",
    "t_step_ms", "t_bookkeeping_ms", "t_loop_wall_ms", "t_per_iter_ms",
    "timing_mode", "optimization_time_s",
]

GRADIENT_STATS_HEADER = [
    "frame_id", "iteration", "n_gaussians",
    "grad_sum", "grad_mean", "grad_max",
    "n_below_1e-04", "n_below_1e-03", "n_below_1e-02", "n_below_1e-01",
]

# Raw candidate poses, so that the initialization headroom can be recomputed
# offline under a different metric without re-running the tracker.
POSES_HEADER = ["frame_id", "candidate", "is_gt"] + [
    "c2w_{:d}{:d}".format(row, col) for row in range(4) for col in range(4)
]

GRADIENT_THRESHOLDS = (1e-4, 1e-3, 1e-2, 1e-1)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def rotation_error_deg(R_est, R_gt) -> float:
    """ Geodesic angle (degrees) between two rotation matrices.

    Uses the trace formula rather than scipy so that telemetry stays
    dependency-light and safe to import from anywhere.

    Args:
        R_est: Estimated 3x3 rotation matrix.
        R_gt: Ground-truth 3x3 rotation matrix.
    Returns:
        The rotation error in degrees, or NaN if either input is unusable.
    """
    if R_est is None or R_gt is None:
        return float("nan")
    try:
        A = np.asarray(R_est, dtype=np.float64).reshape(3, 3)
        B = np.asarray(R_gt, dtype=np.float64).reshape(3, 3)
    except (ValueError, TypeError):
        return float("nan")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        return float("nan")
    cos_theta = (np.trace(A.T @ B) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))


def pose_errors(c2w_est, c2w_gt):
    """ Translation and rotation error between two camera-to-world poses.

    Args:
        c2w_est: Estimated camera-to-world 4x4 matrix.
        c2w_gt: Ground-truth camera-to-world 4x4 matrix.
    Returns:
        Tuple (translation error in metres, rotation error in degrees).
    """
    if c2w_est is None or c2w_gt is None:
        return float("nan"), float("nan")
    try:
        est = np.asarray(c2w_est, dtype=np.float64)
        gt = np.asarray(c2w_gt, dtype=np.float64)
    except (ValueError, TypeError):
        return float("nan"), float("nan")
    if est.shape != (4, 4) or gt.shape != (4, 4):
        return float("nan"), float("nan")
    t_err = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
    return t_err, rotation_error_deg(est[:3, :3], gt[:3, :3])


def xyz_gradient_stats(xyz_param, thresholds=GRADIENT_THRESHOLDS) -> dict:
    """ Per-Gaussian summary statistics of ``||dL/dxyz||``.

    The per-Gaussian gradient of the position parameter is the canonical
    "importance" signal (it is what 3DGS accumulates in
    ``GaussianModel.xyz_gradient_accum``), so its distribution is what tells us
    whether flat-region Gaussians carry meaningful gradient mass.

    Args:
        xyz_param: The ``_xyz`` parameter tensor of the Gaussian model.
        thresholds: Magnitude thresholds at which to count Gaussians.
    Returns:
        A dict of scalars, or None when no gradient is available.
    """
    if xyz_param is None:
        return None
    grad = getattr(xyz_param, "grad", None)
    if grad is None:
        return None
    with torch.no_grad():
        norms = grad.detach().norm(dim=1)
        n = int(norms.shape[0])
        if n == 0:
            return None
        stats = {
            "n_gaussians": n,
            "grad_sum": float(norms.sum().item()),
            "grad_max": float(norms.max().item()),
        }
        stats["grad_mean"] = stats["grad_sum"] / n
        for threshold in thresholds:
            # Column name must match GRADIENT_STATS_HEADER, e.g. "n_below_1e-02".
            key = "n_below_{:.0e}".format(threshold)
            stats[key] = int((norms < threshold).sum().item())
    return stats


# ---------------------------------------------------------------------------
# Telemetry writer
# ---------------------------------------------------------------------------
class Telemetry(object):
    """ Writes analysis-ready CSV telemetry into ``<output_path>/telemetry``.

    All public methods are safe to call unconditionally: they return early when
    telemetry is disabled, and any write failure disables telemetry (with one
    warning) rather than interrupting the run.
    """

    def __init__(self, output_path, config=None, verbose: bool = False):
        """ Args:
                output_path: Run output directory (already created by GaussianSLAM).
                config: The full run configuration dict. The optional ``logging``
                    section controls this object; missing sections use defaults.
                verbose: Whether to echo telemetry status to stdout.
        """
        cfg = {}
        if isinstance(config, dict):
            cfg = config.get("logging") or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.verbose = verbose
        self.profile_optimization = bool(cfg.get("profile_optimization", True))
        self.gradient_stats_every = int(cfg.get("gradient_stats_every", 10) or 0)
        self.gradient_dump_frames = {
            int(f) for f in (cfg.get("gradient_dump_frames") or [])
        }
        self.root = Path(output_path) / str(cfg.get("subdir", "telemetry"))
        self._writer_cache = {}
        self._handles = {}
        self._failed = False

        if self.enabled:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # pragma: no cover - defensive
                self._disable(f"could not create telemetry directory: {exc}")
                return
            if self.verbose:
                print(f"Telemetry enabled -> {self.root}", flush=True)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        """ Flush and close every open CSV handle. """
        for handle in list(self._handles.values()):
            try:
                handle.close()
            except Exception:
                pass
        self._handles.clear()
        self._writer_cache.clear()

    def _disable(self, reason: str) -> None:
        """ Turn telemetry off after an internal failure, warning only once. """
        if not self._failed:
            self._failed = True
            print(f"[telemetry] disabled: {reason}", flush=True)
        self.enabled = False
        self.close()

    # -- low level ---------------------------------------------------------
    def _writer(self, name: str, header: list):
        """ Return (csv writer, header) for a named file, creating it if needed. """
        if not self.enabled:
            return None
        cached = self._writer_cache.get(name)
        if cached is not None:
            return cached
        try:
            path = self.root / f"{name}.csv"
            is_new = (not path.exists()) or path.stat().st_size == 0
            handle = open(path, "a", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            if is_new:
                writer.writerow(header)
            self._handles[name] = handle
            self._writer_cache[name] = (writer, header)
            return self._writer_cache[name]
        except Exception as exc:  # pragma: no cover - defensive
            self._disable(f"could not open {name}.csv: {exc}")
            return None

    def _write(self, name: str, header: list, row: dict) -> None:
        """ Append one row, emitting values in header order. """
        if not self.enabled:
            return
        entry = self._writer(name, header)
        if entry is None:
            return
        writer, columns = entry
        try:
            writer.writerow([row.get(column, "") for column in columns])
            self._handles[name].flush()
        except Exception as exc:  # pragma: no cover - defensive
            self._disable(f"could not write to {name}.csv: {exc}")

    # -- tracking ----------------------------------------------------------
    def log_init_candidate(self, frame_id, candidate: str, is_chosen: bool,
                           is_oracle: bool, loss_total, loss_color, loss_depth,
                           t_err_m, r_err_deg) -> None:
        """ One row per (frame, pose candidate) evaluated by the tracker. """
        self._write("tracking_init_candidates", CANDIDATES_HEADER, {
            "frame_id": frame_id,
            "candidate": candidate,
            "is_chosen": int(bool(is_chosen)),
            "is_oracle": int(bool(is_oracle)),
            "loss_total": loss_total,
            "loss_color": loss_color,
            "loss_depth": loss_depth,
            "t_err_m": t_err_m,
            "r_err_deg": r_err_deg,
        })

    def log_tracking_frame(self, row: dict) -> None:
        """ One row per tracked frame, summarising the initialization decision. """
        self._write("tracking_frames", TRACKING_FRAMES_HEADER, row)

    def log_pose(self, frame_id, candidate: str, c2w, is_gt: bool = False) -> None:
        """ One row per (frame, pose) in camera-to-world form.

        Candidate rows carry the poses the tracker had to choose between; a single
        ``candidate == "gt"`` row per frame carries the ground truth, so the file is
        self-contained for offline analysis.
        """
        if not self.enabled or c2w is None:
            return
        try:
            values = np.asarray(c2w, dtype=np.float64)
            if values.shape != (4, 4):
                return
        except (ValueError, TypeError):
            return
        row = {"frame_id": frame_id, "candidate": candidate, "is_gt": int(bool(is_gt))}
        for index, value in enumerate(values.reshape(-1)):
            row[POSES_HEADER[3 + index]] = float(value)
        self._write("tracking_poses", POSES_HEADER, row)

    # -- mapping -----------------------------------------------------------
    def log_mapping_frame(self, row: dict) -> None:
        """ One row per mapping frame: seeding counts plus the timing breakdown. """
        self._write("mapping_frames", MAPPING_FRAMES_HEADER, row)

    def log_gradient_stats(self, frame_id, iteration: int, stats: dict) -> None:
        """ One row per sampled optimizer iteration. """
        if not stats:
            return
        row = dict(stats)
        row["frame_id"] = frame_id
        row["iteration"] = iteration
        self._write("gradient_stats", GRADIENT_STATS_HEADER, row)

    def wants_gradient_stats(self, iteration: int) -> bool:
        """ Whether per-Gaussian gradient statistics should be taken this iteration. """
        if not self.enabled or self.gradient_stats_every <= 0:
            return False
        return iteration % self.gradient_stats_every == 0

    def wants_gradient_dump(self, frame_id) -> bool:
        """ Whether this frame was explicitly requested for a full gradient dump. """
        return self.enabled and frame_id in self.gradient_dump_frames

    def dump_gradient_norms(self, frame_id, iteration: int, xyz_param) -> None:
        """ Write the full per-Gaussian gradient-norm vector for one iteration.

        Useful for joining per-Gaussian statistics against a class map computed
        offline, which is why it is opt-in via ``logging.gradient_dump_frames``.
        """
        if not self.wants_gradient_dump(frame_id) or xyz_param is None:
            return
        grad = getattr(xyz_param, "grad", None)
        if grad is None:
            return
        try:
            with torch.no_grad():
                norms = grad.detach().norm(dim=1).cpu()
            path = self.root / f"grad_norms_{int(frame_id):06d}_{int(iteration):04d}.pt"
            torch.save(norms, path)
        except Exception as exc:  # pragma: no cover - defensive
            self._disable(f"could not dump gradient norms: {exc}")


class NullTelemetry(Telemetry):
    """ A telemetry object that is always disabled.

    Used as the default so that Mapper/Tracker can call telemetry methods
    unconditionally without every call site needing a None check.
    """

    def __init__(self):
        self.enabled = False
        self.verbose = False
        self.profile_optimization = False
        self.gradient_stats_every = 0
        self.gradient_dump_frames = set()
        self.root = Path(".")
        self._writer_cache = {}
        self._handles = {}
        self._failed = False
