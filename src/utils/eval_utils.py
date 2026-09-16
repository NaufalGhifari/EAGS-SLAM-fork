import json
import os
import inspect
import evo
import numpy as np

import torch
from errno import EEXIST
from os import makedirs, path
from evo.core import metrics, trajectory
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import plot
from evo.tools.plot import PlotMode
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from tqdm import tqdm

import wandb

import rich

_log_styles = {
    "Eval": "bold red",
}

def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="MonoGS"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise


def plot_ate_trajectory(traj_ref, traj_est_aligned, errors, ape_stats, ape_stat, out_path):
    """ Renders and saves the xy trajectory plot with a per-pose error colour map.

    Kept separate from the metric computation so that a plotting failure cannot be
    confused with an evaluation failure - the ATE statistic is computed and written to
    ``stats_<label>.json`` before this is ever called.

    evo's helpers do not share a single calling convention: ``plot.traj`` takes the axes
    first, while ``traj_colormap`` takes the trajectory first and accepts ``ax`` by
    keyword. The order is therefore selected from the installed signature rather than
    hard-coded, so this keeps working across evo versions.

    Args:
        traj_ref: Reference (ground truth) trajectory as an evo PosePath3D.
        traj_est_aligned: Estimated trajectory, aligned to the reference.
        errors: Per-pose error array used to colour the estimated trajectory.
        ape_stats: Statistics dict (needs "min" and "max" for the colour scale).
        ape_stat: RMSE value, used as the plot title.
        out_path: Where to write the PNG.
    """
    plot_mode = evo.tools.plot.PlotMode.xy
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")

    colormap_kwargs = {"min_map": ape_stats["min"], "max_map": ape_stats["max"]}
    parameters = list(inspect.signature(evo.tools.plot.traj_colormap).parameters)
    if parameters and parameters[0] == "traj":
        evo.tools.plot.traj_colormap(
            traj_est_aligned, errors, plot_mode, ax=ax, **colormap_kwargs)
    else:
        evo.tools.plot.traj_colormap(
            ax, traj_est_aligned, errors, plot_mode, **colormap_kwargs)

    ax.legend()
    plt.savefig(out_path, dpi=90)
    plt.close(fig)  # otherwise figures accumulate over repeated loop-closure passes


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est_aligned = trajectory.align_trajectory(
        traj_est, traj_ref, correct_scale=monocular
    )

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    # The trajectory plot is a diagnostic artefact only; the ATE statistic has already
    # been computed and written to stats_<label>.json above. Plotting must therefore never
    # abort the run: an evo/matplotlib backend problem (e.g. a headless host where
    # `evo_config set plot_backend agg` was not applied) previously raised out of the
    # loop-closure thread, through Loop_closure.check_futures (lc.py:662), and killed the
    # entire SLAM run for the sake of a PNG.
    try:
        plot_ate_trajectory(traj_ref, traj_est_aligned, ape_metric.error, ape_stats, ape_stat,
                            os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))))
    except Exception as exc:
        Log(f"skipping ATE trajectory plot ({type(exc).__name__}: {exc})", tag="Eval")

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

        trj_id.append(int(frames[kf_id].uid))
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    return ate