# Structured telemetry

Reference for the CSV telemetry added to the baseline pipeline, and for the three
questions it exists to answer.

Implementation: `src/utils/telemetry.py`.
Write path: `<output_path>/telemetry/`.

This instrumentation is **independent of `verbose`**. `verbose` prints to stdout
(scrollback only, gone when the terminal closes); telemetry writes analysis-ready CSVs
that survive the run and can be loaded directly into pandas. Both can be enabled at once.

---

## 1. Why it exists

Three specific claims need evidence that the ordinary run artefacts
(`rendering_metrics.json`, `ate.json`, the submap checkpoints) cannot supply:

| Question | Why the existing artefacts can't answer it | Evidence produced here |
|---|---|---|
| **Was the init-pose choice good?** `Tracker.init_pose_min_loss` (`tracker.py:130-169`) renders 2–3 candidate poses and picks by L1 loss. On a flat scene all candidates score alike, so the pick may be noise. | `ate.json` is a single trajectory-wide number; it says nothing about whether an *individual frame's* choice was right, nor how much better an alternative would have been | `tracking_init_candidates.csv`, `tracking_frames.csv`, `tracking_poses.csv` |
| **Can a gradient mask buy time?** A mask only skips `optimizer.step()` (`mapper.py:170`); render and `backward` still scale with N. | `Map time avg` in the log bundles seeding + 100–900 iterations into one number | `mapping_frames.csv` |
| **Is an update-priority mechanism redundant?** Adam already under-moves parameters with persistently small gradients. | Nothing records per-Gaussian gradient magnitude anywhere (the `xyz_gradient_accum` field at `gaussian_model.py:51` is dead code in this fork) | `gradient_stats.csv` |

---

## 2. Configuration

New top-level `logging:` section, present in all four base configs
(`configs/Replica/replica.yaml`, `configs/ScanNet/scannet.yaml`,
`configs/scannetpp/scannetpp.yaml`, `configs/TUM_RGBD/tum_rgbd.yaml`):

```yaml
logging:
  enabled: True                  # master switch; every call site is a no-op when False
  subdir: "telemetry"            # directory under output_path
  profile_optimization: True     # CUDA-event timing breakdown of optimize_submap
  gradient_stats_every: 10       # iterations between per-Gaussian |dL/dxyz| samples (0 = off)
  gradient_dump_frames: []       # frame ids to dump full per-Gaussian gradient vectors for
```

Config flows through the existing path: `io_utils.load_config` →
`GaussianSLAM(config)` (`gaussian_slam.py:35-74`) → constructed once and passed to both
`Mapper` and `Tracker`. Per-scene configs inherit the section automatically via
`inherit_from`.

**Failure isolation.** Any internal error (unwritable path, malformed row) disables
telemetry and prints one warning — it never interrupts a run. Writers append, and each
row is flushed immediately, so a crashed run still leaves usable data.

---

## 3. Output files

### 3.1 `tracking_init_candidates.csv` — one row per (frame, candidate)

| Column | Meaning |
|---|---|
| `frame_id` | Frame index |
| `candidate` | `const_speed`, `previous`, or `odometer` (`tracker.py:286-289`) |
| `is_chosen` | 1 if the L1 loss selected this candidate |
| `is_oracle` | 1 if this candidate has the **lowest ground-truth pose error** of the set |
| `loss_total` | `w_color_loss * color + (1 - w) * depth`, the value used for selection |
| `loss_color`, `loss_depth` | The two components |
| `t_err_m` | Translation error vs ground truth, metres |
| `r_err_deg` | Geodesic rotation error vs ground truth, degrees |

`is_chosen` vs `is_oracle` is the crux: the disagreement rate is how often the scoring
function picks a pose that is *not* the best available one.

### 3.2 `tracking_frames.csv` — one row per tracked frame

| Column | Meaning |
|---|---|
| `n_candidates` | 2 or 3 |
| `chosen`, `oracle` | Winning candidate by loss / by ground truth |
| `choice_is_oracle` | 1 when they agree |
| `chosen_t_err_m`, `oracle_t_err_m` | Pose error of each |
| **`headroom_m`** | `chosen_t_err_m - oracle_t_err_m` — **accuracy left on the table this frame** |
| `best_loss`, `second_best_loss`, `loss_margin` | Loss margin between the top two candidates — a direct measure of whether the score discriminates at all |
| `fallback_fired` | 1 if the high-init-loss fallback triggered (iteration doubling / odometer re-init, `tracker.py:297-303`) |
| `num_iters` | Refinement iterations configured for this frame |
| `final_t_err_m`, `final_r_err_deg` | Pose error **after** refinement |
| `track_ms` | Wall-clock duration of the tracking call |

> **The headline number is the aggregate of `headroom_m`.** If the oracle is barely better
> than the chosen candidate across the run, no improvement to the scoring function can
> help, and the idea is dead before any code is written.

### 3.3 `tracking_poses.csv` — one row per (frame, pose)

`frame_id`, `candidate`, `is_gt`, then 16 columns `c2w_00 … c2w_33` holding the flattened
camera-to-world matrix. One `candidate == "gt"` row per frame carries ground truth, so the
file is self-contained: the headroom can be recomputed offline under any metric without
re-running the tracker.

### 3.4 `mapping_frames.csv` — one row per mapping frame

**Growth / seeding:**

| Column | Meaning |
|---|---|
| `is_new_submap` | 1 if this frame started a new submap |
| `max_iterations` | Requested budget (`iterations` or `new_submap_iterations`) |
| `iterations_run` | Iterations actually executed (early stop may cut it short) |
| `early_stopped` | 1 if the early-stop branch fired (`mapper.py:243-250`) |
| `n_gaussians_before` | Model size before this frame's seeding |
| `n_new_points_candidate` | Candidate points surviving novelty filtering (`new_pts_num`) |
| `n_added` | Gaussians actually added by seeding |
| `n_gaussians_after` | Model size after optimization and the final prune |
| `n_pruned_early`, `n_pruned_final` | Counts removed at the 30 %/60 % prunes and the final prune |

**Cost breakdown** (all milliseconds, summed over the frame's iterations):

| Column | Section of `optimize_submap` |
|---|---|
| `t_render_ms` | `render_gaussian_model` + exposure correction (`mapper.py:198-205`) |
| `t_loss_ms` | Mask, L1 + D-SSIM, depth L1, isotropic reg (`mapper.py:208-214`) |
| `t_backward_ms` | `total_loss.backward()` (`mapper.py:217`) |
| `t_gradstats_ms` | Telemetry's own gradient sampling (0 when disabled) |
| **`t_step_ms`** | **`optimizer.step()` — the *only* section a gradient mask can remove** |
| `t_bookkeeping_ms` | `.item()` calls, early-stop, checkpoint, prune, verbose logging |
| `t_loop_wall_ms` | Wall-clock of the whole iteration loop |
| `t_per_iter_ms` | `t_loop_wall_ms / iterations_run` |
| `timing_mode` | `cuda_event` (GPU) or `perf_counter` (CPU fallback) |
| `optimization_time_s` | The pre-existing `optimization_time` from `losses_dict` |

**The decisive ratio is `sum(t_step_ms) / sum(t_loop_wall_ms)`.** That is Adam's share of
the optimization loop. Everything else — render and backward — is paid regardless of how
many Gaussians a mask excludes.

### 3.5 `gradient_stats.csv` — one row per sampled iteration

| Column | Meaning |
|---|---|
| `frame_id`, `iteration` | Position in the run |
| `n_gaussians` | Model size at that moment |
| `grad_sum`, `grad_mean`, `grad_max` | Statistics of per-Gaussian `‖∂L/∂xyz‖` |
| `n_below_1e-04 … n_below_1e-01` | Count of Gaussians below each magnitude threshold |

`‖∂L/∂xyz‖` is the canonical per-Gaussian importance signal — the quantity 3DGS
accumulates in `xyz_gradient_accum`, which is dead code here. It is read from
`_xyz.grad` immediately after `backward()` and before `step()`, and each sample reflects
that single iteration because `zero_grad(set_to_none=True)` (`mapper.py:189`) clears the
field at the top of every iteration.

**Reading it:** if `n_below_1e-03 / n_gaussians` is near 1, the optimizer already ignores
most Gaussians and an explicit priority mechanism is largely redundant. If the
distribution is heavy-tailed with substantial mass far from zero, there is something for a
mechanism to act on.

### 3.6 Optional gradient dumps

With frame ids listed in `logging.gradient_dump_frames`, each iteration of those frames
writes `grad_norms_<frame>_<iter>.pt` — the **full** per-Gaussian vector, not just
summary statistics. Use this to join per-Gaussian gradient magnitudes against an
edge/class map computed offline, which is what makes the "gradient mass by class" analysis
possible once the 4-class machinery exists. Off by default: these files are N floats each.

---

## 4. Analysis recipes

```python
import pandas as pd
tel = "output/TUM_RGBD/rgbd_dataset_freiburg1_desk/<timestamp>/telemetry/"

# --- 1. Is there headroom in the initialization choice? ---
f = pd.read_csv(tel + "tracking_frames.csv")
print("loss picks the GT-best candidate: {:.1%}".format(f.choice_is_oracle.mean()))
print(f.headroom_m.describe())              # metres left on the table
print(f.loss_margin.describe())             # is the score discriminating at all?
print(f.groupby("chosen").num_iters.mean()) # does a worse init cost more iterations?
print(f.groupby("fallback_fired").headroom_m.mean())

# --- 2. Where does the mapping time actually go? ---
m = pd.read_csv(tel + "mapping_frames.csv")
secs = ["t_render_ms", "t_loss_ms", "t_backward_ms", "t_gradstats_ms", "t_step_ms", "t_bookkeeping_ms"]
share = m[secs].sum() / m[secs].sum().sum()
print(share)                                # t_step_ms share == the ceiling on any mask speedup
print("iterations run:", m.iterations_run.sum())

# --- 3. Where does the gradient mass live? ---
g = pd.read_csv(tel + "gradient_stats.csv")
g["frac_near_zero"] = g.n_below_1e-03 / g.n_gaussians
print(g.groupby("frame_id").frac_near_zero.mean().describe())
print(g.groupby("frame_id").grad_mean.median())
```

---

## 5. Caveats

* **Timing instrumentation perturbs wall-clock.** `cuda_event` mode issues one
  `torch.cuda.synchronize()` per iteration, removing CPU/GPU overlap. Section timings are
  accurate; the absolute duration of an instrumented run is somewhat higher than an
  uninstrumented one. Compare `t_loop_wall_ms` against the baseline `Map time avg`, or set
  `profile_optimization: False` to keep everything except the timing breakdown.
* **Gradient sampling costs O(N) reductions.** With `gradient_stats_every: 10`, roughly
  10 % of iterations pay a few extra reductions over N Gaussians. Raise it for long runs.
* **`headroom_m` needs ground truth.** It relies on `dataset[frame_id]` returning the GT
  pose. Tracking configurations that use GT poses directly short-circuit before telemetry
  is reached (`tracker.py:196-197`).
* **Only `_xyz` gradients are sampled.** That is deliberate: the colour (SH) groups are
  force-frozen by `requires_grad = False` at `mapper.py:451-452`, so they carry no gradient
  at all, and position is the parameter that matters for structure.
* **`n_added` is the true seeding addition**, taken before pruning; `n_gaussians_after` is
  post-prune, so `n_gaussians_after - n_gaussians_before` is the *net* change and will not
  equal `n_added` once pruning has removed points.

## 6. What telemetry does *not* capture

* **The rendered images per candidate.** Without them the candidate losses can be
  recomputed only under metrics that depend on stored data, not on re-rendering. Storing
  them would be large (H×W×3 per candidate per frame).
* **Per-Gaussian class labels.** The 4-class edge taxonomy does not exist in the code yet;
  `gradient_stats.csv` gives the unlabelled distribution now, and
  `gradient_dump_frames` is the hook for joining classes in later.
* **Per-section timings inside seeding** (`create_point_cloud`, FAISS novelty search,
  `distCUDA2` in `add_points_with_edge`). `optimization_time_s` covers `optimize_submap`
  only; the remainder of `Map time` is seeding.
