import time
from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import torch
import torchvision
import cv2

from src.entities.arguments import OptimizationParams
from src.entities.datasets import TUM_RGBD, BaseDataset, ScanNet
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.mapper_utils import (calc_psnr, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask,
                                    sample_pixels_based_on_gradient)
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.telemetry import NullTelemetry, xyz_gradient_stats
from src.utils.vis_utils import *  # noqa - needed for debugging

class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger, device:int=0, verbose:bool=False,
                 telemetry=None) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
            telemetry: Optional Telemetry object for structured CSV logging.
        """
        self.device = device
        self.config = config
        self.VERBOSE:bool = verbose
        self.logger = logger
        self.telemetry = telemetry if telemetry is not None else NullTelemetry()
        self.dataset = dataset
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_thre = config["alpha_thre"]
        self.pruning_thre = config["pruning_thre"]
        self.current_view_opt_iterations = config["current_view_opt_iterations"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = []
        self.DEPTH_THRES:float = config["edge_depth_thres"] if "edge_depth_thres" in config else 0.025

    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        Args:
            gaussian_model: The current submap
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        seeding_mask = None
        if new_submap:
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        else:
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask

    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) \
         -> Tuple[np.ndarray, np.ndarray]:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)
            np.ndarray: The point's sample index of the whole point cloud
        """
        pts = create_point_cloud(gt_color, 1.005 * gt_depth, intrinsics, estimate_c2w)
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.  # need filter if zero depth pixels in gt_depth
        valid_ids = np.flatnonzero(seeding_mask) # get flatten none zero index of seeding_mask
        if is_new_submap:
            if self.new_submap_points_num < 0:
                uniform_ids = np.arange(pts.shape[0])
            else:
                uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            combined_ids = np.concatenate((combined_ids, valid_ids))
            sample_ids = np.unique(combined_ids)
        else:
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        return pts[sample_ids, :].astype(np.float32), sample_ids

    def optimize_submap(self, keyframes:list, gaussian_model: GaussianModel, iterations:int=100,
                        mapping_frame_id:int=None) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes (list): A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model (GaussianModel): An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations (int): The number of iterations to perform the optimization process. Defaults to 100.
            mapping_frame_id (int): The frame currently being mapped, used for telemetry only. Note that the loop
                below rebinds a local ``frame_id`` to the *sampled keyframe*, hence the distinct name.
        Returns:
            losses_dict (dict): Dictionary with the optimization statistics
        """

        losses_dict = {}
        lowest_loss = float("inf")

        ckp = None  # Checkpoint
        ckp_iter = int(0)
        SAVE_CKP_EVERY_ITER = int(0.05*iterations)
            
        early_stop_cnt = int(0)
        EARLY_STOP_CNT_THRE = int(0.05*iterations)

        PRUNE_ITERS:list[int] = [int(0.3*iterations), int(0.6*iterations)]
        current_frame_iters = self.current_view_opt_iterations * iterations
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)

        start_time = time.time()

        # --- telemetry: timing boundaries ------------------------------------
        # Boundary events are reused every iteration; CUDA events give accurate
        # per-section GPU timings at the cost of one synchronisation per
        # iteration, which is what makes the absolute wall-clock of an
        # instrumented run slightly higher than an uninstrumented one.
        telemetry = self.telemetry
        row_frame_id = mapping_frame_id if mapping_frame_id is not None else keyframes[0][0]
        timing_mode = "off"
        events = None
        marks = [0.0] * 7
        if telemetry.enabled and telemetry.profile_optimization:
            if torch.cuda.is_available():
                events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
                timing_mode = "cuda_event"
            else:
                timing_mode = "perf_counter"

        def _stamp(index:int) -> None:
            """ Marks a timing boundary (CUDA event, or timestamp on CPU). """
            if events is not None:
                events[index].record()
            elif timing_mode == "perf_counter":
                marks[index] = time.perf_counter()

        def _elapsed(index:int) -> float:
            """ Milliseconds between boundary ``index`` and ``index + 1``. """
            if events is not None:
                return events[index].elapsed_time(events[index + 1])
            if timing_mode == "perf_counter":
                return (marks[index + 1] - marks[index]) * 1000.0
            return 0.0
        # Sections: 0 render, 1 loss, 2 backward, 3 gradient stats, 4 step, 5 bookkeeping
        t_sections = [0.0] * 6
        loop_start = time.perf_counter()
        iterations_run = int(0)
        early_stopped = False
        n_pruned_early = int(0)

        for iteration in range(0,iterations):
            gaussian_model.optimizer.zero_grad(set_to_none=True)

            if iteration < 5:
                keyframe_id = 0
            else:
                keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)
            frame_id, keyframe = keyframes[keyframe_id]

            _stamp(0)
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            image, depth = render_pkg["color"], render_pkg["depth"]

            if keyframe["exposure_ab"] is not None:
                image = torch.clamp(image * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)

            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]
            _stamp(1)

            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            reg_loss = isotropic_loss(gaussian_model.get_scaling())
            total_loss = color_loss + depth_loss + reg_loss
            _stamp(2)

            total_loss.backward()
            _stamp(3)

            # Telemetry: per-Gaussian gradient distribution of the position parameter.
            # This is the signal that decides whether an explicit update-priority
            # mechanism is redundant (gradients already negligible) or justified.
            if telemetry.enabled:
                if telemetry.wants_gradient_dump(row_frame_id):
                    telemetry.dump_gradient_norms(row_frame_id, iteration, gaussian_model.get_xyz())
                if telemetry.wants_gradient_stats(iteration):
                    telemetry.log_gradient_stats(row_frame_id, iteration,
                                                 xyz_gradient_stats(gaussian_model.get_xyz()))
            _stamp(4)

            gaussian_model.optimizer.step()
            _stamp(5)

            with torch.no_grad():
                c_loss = color_loss.item()
                d_loss = depth_loss.item()
                r_loss = reg_loss.item()
                t_loss = total_loss.item()
                losses_dict[frame_id] = {"color_loss": c_loss,
                                        "depth_loss": d_loss,
                                        "total_loss": t_loss}
                
                # Early stop
                if iteration>PRUNE_ITERS[-1] and ckp is not None:
                        if t_loss-lowest_loss > 0.15*lowest_loss:
                            early_stop_cnt += 1
                            if early_stop_cnt>EARLY_STOP_CNT_THRE:
                                if self.VERBOSE:
                                    print(f"Early stop at iter:{iteration:d}")
                                early_stopped = True
                                break
                        else:
                            early_stop_cnt = int(0)

                # Save checkpoint
                if iteration%SAVE_CKP_EVERY_ITER==0 and iteration!=0:
                    if t_loss < lowest_loss:
                        lowest_loss = t_loss
                        ckp = gaussian_model.optimizer.state_dict()
                        ckp_iter = iteration
                        early_stop_cnt = int(0)
                
                # Prune gaussians, load checkpoint if needed
                if iteration in PRUNE_ITERS:
                    if lowest_loss < t_loss and ckp is not None:
                        gaussian_model.optimizer.load_state_dict(ckp)
                        if self.VERBOSE:
                            print(f"Resume checkpoint at iter {ckp_iter:d}, loss:{lowest_loss:.5f}")
                    prune_mask = (gaussian_model.get_opacity() < self.pruning_thre).squeeze()
                    n_pruned_early += int(prune_mask.sum().item())
                    gaussian_model.prune_points(prune_mask)
                    lowest_loss = float("inf")
                    ckp = None

                # Log
                if self.VERBOSE and iteration % 10 == 0:
                    print(f"Iter:{iteration:3d}, loss T:{t_loss:.5f}, C:{c_loss:.5f}, D:{d_loss:.5f}, R:{r_loss:f}",
                          flush=True)
                _stamp(6)

            # Telemetry: close the iteration's timing boundaries and accumulate.
            # A ``break`` above skips this for the final (partial) iteration,
            # which is intended: the loop is over either way.
            if events is not None:
                torch.cuda.synchronize(self.device)
            for _section in range(6):
                t_sections[_section] += _elapsed(_section)
            iterations_run += 1

        # Load checkpoint if needed
        if lowest_loss < t_loss and ckp is not None:
            gaussian_model.optimizer.load_state_dict(ckp)
            if self.VERBOSE:
                print(f"Resume checkpoint at iter {ckp_iter:d}, loss:{lowest_loss:.5f}")

        prune_mask = (gaussian_model.get_opacity() < 0.01).squeeze()
        n_pruned_final = int(prune_mask.sum().item())
        gaussian_model.prune_points(prune_mask)

        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations

        # Telemetry: what the outer Mapper.map() needs to write one row per mapping frame.
        losses_dict["iterations_run"] = iterations_run
        losses_dict["early_stopped"] = int(early_stopped)
        losses_dict["n_pruned_early"] = n_pruned_early
        losses_dict["n_pruned_final"] = n_pruned_final
        if timing_mode != "off":
            losses_dict["timing"] = {
                "mode": timing_mode,
                "render_ms": t_sections[0],
                "loss_ms": t_sections[1],
                "backward_ms": t_sections[2],
                "gradstats_ms": t_sections[3],
                "step_ms": t_sections[4],
                "bookkeeping_ms": t_sections[5],
                "loop_wall_ms": (time.perf_counter() - loop_start) * 1000.0,
            }
        return losses_dict

    def grow_submap(self, gt_depth: np.ndarray, estimate_c2w: np.ndarray, gaussian_model: GaussianModel,
                    pts: np.ndarray, filter_cloud: bool) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        Args:
            gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
            estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
            gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
            pts: The current set of 3D points in the keyframe of shape (N, 3)
            filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
                outliers or noise before integrating it into the map.
        Returns:
            int: The number of points added to the submap
        """
        gaussian_points = gaussian_model.get_xyz() # get all gaussians
        camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics)
        reused_pts_ids = compute_frustum_point_ids( # points that lay in frustum
            gaussian_points, np2torch(camera_frustum_corners), device=self.device)
        new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
                                             radius=self.new_points_radius, device=self.device) # remove too close points
        new_pts_ids = torch2np(new_pts_ids)
        if new_pts_ids.shape[0] > 0:
            cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0) # points to pcl with color
            if filter_cloud:
                cloud_to_add, inlier_ids = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
            gaussian_model.add_points(cloud_to_add)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        if self.VERBOSE:
            print("Gaussian model size", gaussian_model.get_size())
        return new_pts_ids.shape[0]

    def map(self, frame_id:int, estimate_c2w:np.ndarray, gaussian_model:GaussianModel,
            is_new_submap:bool, exposure_ab=None, edge_img:np.ndarray=None) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
            edge_img (np.ndarray): The edge image
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """

        _, gt_color, gt_depth, _ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)

        edge_none:bool = edge_img is None
        if edge_none:
            edge_bool = np.zeros_like(gt_depth, dtype=bool)
        else:
            edge_bool = (edge_img != 0) # convert edge_img to bool
            edge_bool[[0,-1],:] = False # Set the outermost pixel of the image 
            edge_bool[:,[0,-1]] = False # to False to avoid boundary problems

        color_transform = torchvision.transforms.ToTensor()
        render_setting = get_render_settings(self.dataset.width, self.dataset.height,
                                             self.dataset.intrinsics, estimate_w2c, device=self.device)
        keyframe = {
            "color": color_transform(gt_color).to(self.device, non_blocking=True),
            "depth": torch.from_numpy(gt_depth).float().to(self.device, non_blocking=True),
            "edge": torch.from_numpy(edge_bool).bool().to(self.device, non_blocking=True),
            "render_settings": render_setting,
            "exposure_ab": exposure_ab,
        }
        torch.cuda.synchronize(self.device)

        # 1. Compute seeding alphs and depth_error mask. The points in mask will be seeding
        if is_new_submap:
            if edge_none:
                # color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
                seeding_mask = (geometric_edge_mask(gt_color, RGB=True) != 0)
            else:
                kernel = np.ones((2, 2), np.uint8)
                seeding_mask = (cv2.dilate(edge_img, kernel, iterations=1) != 0)
        else:
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = keyframe["depth"][None] # H*W -> 1*H*W
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) & (depth_error > 40 * depth_error.median())
            seeding_mask = (alpha_mask | depth_error_mask)[0] # 1*H*W -> H*W
            seeding_mask = torch2np(seeding_mask)


        # 2. Seeding new gaussians: Get 3D points of new gaussians
        all_pts = create_point_cloud(gt_color, 1.0001 * gt_depth, self.dataset.intrinsics, estimate_c2w)
        valid_ids = np.flatnonzero(seeding_mask) # get flatten none zero index of seeding_mask
        if is_new_submap:
            if self.new_submap_points_num <= 0 or self.new_submap_points_num >= len(all_pts):
                uniform_ids = np.arange(all_pts.shape[0])
            else:
                uniform_ids = np.random.choice(all_pts.shape[0], self.new_submap_points_num, replace=False)
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            sample_ids = np.unique(np.concatenate((uniform_ids, gradient_ids, valid_ids)))
        else:
            if self.new_frame_sample_size <= 0 or len(valid_ids) <= self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        non_zero_depth_mask = gt_depth.flatten() > 0  # need filter if zero depth pixels in gt_depth
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        pts = all_pts[sample_ids, :].astype(np.float32)


        # 3. Grow submap
        n_gaussians_before = gaussian_model.get_size()  # telemetry: size before this frame's seeding
        gaussian_points = gaussian_model.get_xyz() # get all gaussians
        camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics)
        reused_pts_ids = compute_frustum_point_ids( # points that lay in frustum
            gaussian_points, np2torch(camera_frustum_corners), device=self.device)
        new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
                                             radius=self.new_points_radius, device=self.device) # remove too close points
        new_pts_ids = torch2np(new_pts_ids)
        new_pts_num = new_pts_ids.shape[0]
        if new_pts_num > 0:
            cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0) # points to pcl with color
            if isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap:
                cloud_to_add, inlier_ids = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
                sample_ids = sample_ids[new_pts_ids[inlier_ids]]
            else:
                sample_ids = sample_ids[new_pts_ids]
            if edge_none:
                gaussian_model.add_points(cloud_to_add)
            else:
                gaussian_model.add_points_with_edge(all_pts, sample_ids,
                    keyframe["edge"], keyframe["depth"], depth_thres=self.DEPTH_THRES)
        gaussian_model._features_dc.requires_grad = False
        gaussian_model._features_rest.requires_grad = False
        n_added = gaussian_model.get_size() - n_gaussians_before  # telemetry: true seeding additions
        if self.VERBOSE:
            print(f"Gaussian model size: {gaussian_model.get_size()}")


        # 4. Optimize submap
        max_iterations = self.iterations
        if is_new_submap:
            max_iterations = self.new_submap_iterations
        if self.VERBOSE:
            t_start = time.perf_counter()
        opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations,
                                        mapping_frame_id=frame_id)
        if self.VERBOSE:
            optimization_time = time.perf_counter()-t_start
            print(f"Optimization time: {int(optimization_time*1000):d} ms")

        self._log_mapping_frame(frame_id, is_new_submap, max_iterations, n_gaussians_before, n_added,
                                new_pts_num, gaussian_model.get_size(), opt_dict)

        self.keyframes.append((frame_id, keyframe))

        if self.VERBOSE:
            # Visualise the mapping for the current frame
            with torch.no_grad():
                render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"])
                image_vis = render_pkg_vis["color"]
                depth_vis = render_pkg_vis["depth"]
                if keyframe["exposure_ab"] is not None:
                    image_vis = torch.clamp(image_vis * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
                psnr_value = calc_psnr(image_vis, keyframe["color"]).item()
                opt_dict["psnr_render"] = psnr_value
                print(f"PSNR this frame: {psnr_value}")
                self.logger.vis_mapping_iteration(
                    frame_id, max_iterations,
                    image_vis.clone().detach().permute(1, 2, 0),
                    depth_vis.clone().detach().permute(1, 2, 0),
                    keyframe["color"].permute(1, 2, 0),
                    keyframe["depth"].unsqueeze(-1),
                    seeding_mask=seeding_mask)
                # Log the mapping numbers for the current frame
                self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                                optimization_time/max_iterations, opt_dict)

        return opt_dict

    def _log_mapping_frame(self, frame_id:int, is_new_submap:bool, max_iterations:int,
                           n_before:int, n_added:int, n_new_candidates:int, n_after:int,
                           opt_dict:dict) -> None:
        """ Writes one telemetry row per mapping frame.

        The row combines the seeding counts (known here) with the optimization
        statistics returned by ``optimize_submap``, so that a single table holds
        the full per-frame cost and growth picture.

        Args:
            frame_id: The frame being mapped.
            is_new_submap: Whether this frame started a new submap.
            max_iterations: The iteration budget requested for this frame.
            n_before: Number of Gaussians before this frame's seeding.
            n_added: Gaussian count added by this frame's seeding (before pruning).
            n_new_candidates: Number of candidate points that survived novelty filtering.
            n_after: Number of Gaussians after optimization and the final prune.
            opt_dict: The dictionary returned by ``optimize_submap``.
        """
        if not self.telemetry.enabled:
            return
        timing = opt_dict.get("timing") or {}
        iterations_run = opt_dict.get("iterations_run") or 0
        loop_wall_ms = timing.get("loop_wall_ms", "")
        row = {
            "frame_id": frame_id,
            "is_new_submap": int(bool(is_new_submap)),
            "max_iterations": max_iterations,
            "iterations_run": iterations_run,
            "early_stopped": opt_dict.get("early_stopped", ""),
            "n_gaussians_before": n_before,
            "n_new_points_candidate": n_new_candidates,
            "n_added": n_added,
            "n_gaussians_after": n_after,
            "n_pruned_early": opt_dict.get("n_pruned_early", ""),
            "n_pruned_final": opt_dict.get("n_pruned_final", ""),
            "t_render_ms": timing.get("render_ms", ""),
            "t_loss_ms": timing.get("loss_ms", ""),
            "t_backward_ms": timing.get("backward_ms", ""),
            "t_gradstats_ms": timing.get("gradstats_ms", ""),
            "t_step_ms": timing.get("step_ms", ""),
            "t_bookkeeping_ms": timing.get("bookkeeping_ms", ""),
            "t_loop_wall_ms": loop_wall_ms,
            "t_per_iter_ms": (loop_wall_ms / iterations_run) if (iterations_run and loop_wall_ms != "") else "",
            "timing_mode": timing.get("mode", "off"),
            "optimization_time_s": opt_dict.get("optimization_time", ""),
        }
        self.telemetry.log_mapping_frame(row)

    def report(self):
        pass
