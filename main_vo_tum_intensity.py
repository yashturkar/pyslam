#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import copy
import numpy as np
import cv2
import os
import math
import time
import platform
import argparse
import glob
import sys

from pyslam.config import Config

from pyslam.slam.visual_odometry import VisualOdometryEducational
from pyslam.slam.visual_odometry_rgbd import (
    VisualOdometryRgbd,
    VisualOdometryRgbdTensor,
)
from pyslam.slam.camera import PinholeCamera

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import DatasetType, SensorType

from pyslam.viz.mplot_thread import Mplot2d, Mplot3d
from pyslam.viz.qplot_thread import Qplot2d
from pyslam.viz.rerun_interface import Rerun

from pyslam.local_features.feature_tracker import (
    feature_tracker_factory,
    FeatureTrackerTypes,
)
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.utilities.utils_sys import Printer

# Transform from the VO/TUM convention (Forward=X, Left=Y, Up=Z) to Rerun's RDF (Right, Down, Forward)
kVoToRerunWorld = np.array(
    [
        [0.0, -1.0, 0.0],   # X_rdf (right) <- -Y_vo (left)
        [0.0, 0.0, -1.0],   # Y_rdf (down)  <- -Z_vo (up)
        [1.0, 0.0, 0.0],    # Z_rdf (fwd)   <-  X_vo (forward)
    ],
    dtype=np.float64,
)


def _convert_pose_to_rerun(pose: np.ndarray):
    """Convert a pose (4x4) from VO world frame (FLU) to Rerun's RDF world frame."""
    if pose is None:
        return None
    pose_rr = np.eye(4, dtype=np.float64)
    pose_rr[:3, :3] = kVoToRerunWorld @ pose[:3, :3]
    pose_rr[:3, 3] = kVoToRerunWorld @ pose[:3, 3]
    return pose_rr


def _convert_traj_to_rerun(traj_points):
    if traj_points is None or len(traj_points) == 0:
        return traj_points
    pts = np.asarray(traj_points, dtype=np.float64)
    return (kVoToRerunWorld @ pts.T).T


def _collect_tracker_choices():
    choices = []
    for name, value in FeatureTrackerConfigs.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, dict):
            choices.append(name)
    return sorted(choices)


kTrackerChoices = _collect_tracker_choices()


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kResultsFolder = kRootFolder + "/results"

# Dataset base path
kDatasetBasePath = "/mnt/share/nas/eph/lucid/Corridor_A/Datasets/Datasets/TUM_Format"

kUseRerun = True
# check rerun does not have issues
if kUseRerun and not Rerun.is_ok:
    kUseRerun = False

"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = True
if platform.system() == "Darwin":
    kUsePangolin = (
        True  # Under mac force pangolin to be used since Mplot3d() has some reliability issues
    )
if kUsePangolin:
    from pyslam.viz.viewer3D import Viewer3D

kUseQplot2d = False
if platform.system() == "Darwin":
    kUseQplot2d = True  # Under mac force the usage of Qtplot2d: It is smoother


def factory_plot2d(*args, **kwargs):
    if kUseRerun:
        return None
    if kUseQplot2d:
        return Qplot2d(*args, **kwargs)
    else:
        return Mplot2d(*args, **kwargs)


def discover_available_intensities(base_path):
    """Discover available intensity values from directory names."""
    if not os.path.exists(base_path):
        Printer.red(f"Dataset base path does not exist: {base_path}")
        return []
    
    intensities = []
    for item in os.listdir(base_path):
        # Look for pattern: Corridor_A_D_XXX_T1 (ignore ENH)
        if item.startswith("Corridor_A_D_") and item.endswith("_T1") and "_ENH" not in item:
            try:
                # Extract intensity value
                parts = item.split("_")
                if len(parts) >= 4:
                    intensity = int(parts[3])  # D_XXX -> XXX
                    intensities.append(intensity)
            except ValueError:
                continue
    
    return sorted(intensities)


def generate_associations_file(sequence_path, times_file, images_dir):
    """Generate associations.txt file from times.txt and images."""
    associations_path = os.path.join(sequence_path, "associations.txt")
    
    # Check if already exists
    if os.path.exists(associations_path):
        Printer.green(f"Associations file already exists: {associations_path}")
        return associations_path
    
    Printer.orange(f"Generating associations file: {associations_path}")
    
    # Read times.txt
    if not os.path.exists(times_file):
        Printer.red(f"times.txt not found: {times_file}")
        return None
    
    # Get sorted image files
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
    
    if not image_files:
        Printer.red(f"No images found in: {images_dir}")
        return None
    
    # Read timestamps from times.txt
    timestamps = []
    with open(times_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                timestamp = float(parts[1])
                timestamps.append(timestamp)
    
    # Match timestamps with images
    if len(timestamps) != len(image_files):
        Printer.yellow(f"Warning: {len(timestamps)} timestamps but {len(image_files)} images")
        min_len = min(len(timestamps), len(image_files))
        timestamps = timestamps[:min_len]
        image_files = image_files[:min_len]
    
    # Write associations file
    # Format: timestamp rgb_image_path depth_image_path
    # For monocular, we use empty string for depth
    # Note: TumDataset.getDepth() returns None early for MONOCULAR, so depth field won't be accessed
    with open(associations_path, 'w') as f:
        for timestamp, img_path in zip(timestamps, image_files):
            # Get relative path from sequence_path
            rel_img_path = os.path.relpath(img_path, sequence_path)
            # TUM format: timestamp rgb_path depth_path
            # For monocular, use empty string for depth (standard TUM format includes this field)
            # Note: getDepth() returns None early for MONOCULAR, so depth field won't be accessed
            f.write(f"{timestamp:.9f} {rel_img_path} \n")
    
    Printer.green(f"Generated associations file with {len(timestamps)} entries")
    return associations_path


def setup_groundtruth_file(sequence_path, sequence_name):
    """Setup groundtruth.txt file (create symlink if needed)."""
    gt_file = os.path.join(sequence_path, f"{sequence_name}_GT.txt")
    groundtruth_txt = os.path.join(sequence_path, "groundtruth.txt")
    
    if os.path.exists(groundtruth_txt):
        Printer.green(f"groundtruth.txt already exists: {groundtruth_txt}")
        return groundtruth_txt
    
    if not os.path.exists(gt_file):
        Printer.red(f"Ground truth file not found: {gt_file}")
        return None
    
    # Create symlink
    try:
        os.symlink(os.path.basename(gt_file), groundtruth_txt)
        Printer.green(f"Created symlink: {groundtruth_txt} -> {os.path.basename(gt_file)}")
    except OSError as e:
        # If symlink fails, try copying
        Printer.yellow(f"Symlink failed ({e}), trying to copy file...")
        import shutil
        shutil.copy2(gt_file, groundtruth_txt)
        Printer.green(f"Copied ground truth file: {groundtruth_txt}")
    
    return groundtruth_txt


def configure_dataset_settings(config, intensity, base_path, settings_file="settings/TUM1.yaml"):
    """Configure dataset settings for the selected intensity."""
    sequence_name = f"Corridor_A_D_{intensity}_T1"
    sequence_path = os.path.join(base_path, sequence_name)
    
    if not os.path.exists(sequence_path):
        Printer.red(f"Sequence path does not exist: {sequence_path}")
        return None
    
    # Setup groundtruth file
    setup_groundtruth_file(sequence_path, sequence_name)
    
    # Generate associations file if needed
    times_file = os.path.join(sequence_path, "times.txt")
    images_dir = os.path.join(sequence_path, "images")
    generate_associations_file(sequence_path, times_file, images_dir)
    
    # Modify config to use TUM dataset
    config.config["DATASET"]["type"] = "TUM_DATASET"
    
    # Update TUM_DATASET settings
    tum_settings = {
        "type": "tum",
        "sensor_type": "mono",  # Monocular since no depth images found
        "base_path": base_path,
        "name": sequence_name,
        "settings": settings_file,
        "associations": "associations.txt",
        "groundtruth_file": "auto"
    }
    
    config.config["TUM_DATASET"] = tum_settings
    
    # Re-read dataset settings
    config.get_dataset_settings()
    config.get_general_system_settings()
    
    Printer.green(f"Configured dataset: {sequence_name}")
    Printer.green(f"  Path: {sequence_path}")
    Printer.green(f"  Settings: {settings_file}")
    
    return sequence_path


def main():
    parser = argparse.ArgumentParser(description="Run PySLAM VO on TUM format dataset with intensity selection")
    parser.add_argument(
        "--intensity", "-i",
        type=int,
        help="Light intensity value (e.g., 100, 127, 190, etc.)"
    )
    parser.add_argument(
        "--settings", "-s",
        type=str,
        default="settings/TUM1.yaml",
        help="Camera settings file (default: settings/TUM1.yaml)"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=kDatasetBasePath,
        help=f"Base path to TUM format dataset (default: {kDatasetBasePath})"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        choices=kTrackerChoices,
        default="LK_SHI_TOMASI",
        help="Feature tracker preset to use (see pyslam.local_features.feature_tracker_configs)",
    )
    parser.add_argument(
        "--num-features",
        type=int,
        default=None,
        help="Override number of features to detect/track (falls back to settings file or tracker preset)",
    )
    
    args = parser.parse_args()
    
    # Discover available intensities
    available_intensities = discover_available_intensities(args.base_path)
    
    if not available_intensities:
        Printer.red(f"No valid intensity sequences found in: {args.base_path}")
        sys.exit(1)
    
    Printer.green(f"Available intensities: {available_intensities}")
    
    # Select intensity
    intensity = args.intensity
    if intensity is None:
        print("\nAvailable intensities:")
        for i, val in enumerate(available_intensities):
            print(f"  {i+1}. D_{val}")
        while True:
            try:
                choice = input(f"\nSelect intensity (1-{len(available_intensities)}) or enter value: ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_intensities):
                        intensity = available_intensities[idx]
                        break
                    else:
                        print(f"Invalid choice. Please enter 1-{len(available_intensities)}")
                else:
                    # Try to parse as intensity value
                    intensity = int(choice)
                    if intensity in available_intensities:
                        break
                    else:
                        print(f"Intensity {intensity} not available. Available: {available_intensities}")
            except (ValueError, KeyboardInterrupt):
                print("\nExiting...")
                sys.exit(0)
    else:
        if intensity not in available_intensities:
            Printer.red(f"Intensity {intensity} not available. Available: {available_intensities}")
            sys.exit(1)
    
    Printer.green(f"Selected intensity: D_{intensity}")
    
    # Load and configure config
    config = Config()
    sequence_path = configure_dataset_settings(config, intensity, args.base_path, args.settings)
    
    if sequence_path is None:
        Printer.red("Failed to configure dataset settings")
        sys.exit(1)
    
    # Create dataset and groundtruth
    dataset = dataset_factory(config)
    groundtruth = groundtruth_factory(config.dataset_settings)
    
    cam = PinholeCamera(config)
    
    tracker_template = FeatureTrackerConfigs.get_config_from_name(args.tracker)
    if tracker_template is None:
        Printer.red(f"Tracker configuration '{args.tracker}' not found. Available: {kTrackerChoices}")
        sys.exit(1)
    tracker_config = copy.deepcopy(tracker_template)
    
    # Determine number of features priority: CLI > settings file > tracker preset
    num_features = tracker_config.get("num_features", 2000)
    if args.num_features is not None:
        num_features = args.num_features
    elif config.num_features_to_extract > 0:
        num_features = config.num_features_to_extract
    tracker_config["num_features"] = num_features
    
    Printer.green(f"Using tracker {args.tracker} with {num_features} features")
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # create visual odometry object
    if dataset.sensor_type == SensorType.RGBD:
        vo = VisualOdometryRgbdTensor(cam, groundtruth)  # only for RGBD
        Printer.green("Using VisualOdometryRgbdTensor")
    else:
        vo = VisualOdometryEducational(cam, groundtruth, feature_tracker)
        Printer.green("Using VisualOdometryEducational")
    time.sleep(1)  # time to read the message
    
    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5 * traj_img_size)
    draw_scale = 1
    
    plt3d = None
    
    viewer3D = None
    
    is_draw_3d = True
    is_draw_with_rerun = kUseRerun
    if is_draw_with_rerun:
        Rerun.init_vo()
    else:
        if kUsePangolin:
            viewer3D = Viewer3D(scale=dataset.scale_viewer_3d * 10)
        else:
            plt3d = Mplot3d(title="3D trajectory")
    
    is_draw_err = True
    err_plt = factory_plot2d(xlabel="img id", ylabel="m", title="error")
    
    is_draw_matched_points = True
    matched_points_plt = factory_plot2d(xlabel="img id", ylabel="# matches", title="# matches")
    
    img_id = 0
    while True:
        
        img = None
        
        if dataset.is_ok:
            timestamp = dataset.getTimestamp()  # get current timestamp
            img = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)
            img_right = (
                dataset.getImageColorRight(img_id)
                if dataset.sensor_type == SensorType.STEREO
                else None
            )
        
        if img is not None:
            
            vo.track(img, img_right, depth, img_id, timestamp)  # main VO function
            
            if (
                len(vo.traj3d_est) > 1
            ):  # start drawing from the third image (when everything is initialized and flows in a normal way)
                
                x, y, z = vo.traj3d_est[-1]
                gt_x, gt_y, gt_z = vo.traj3d_gt[-1]
                
                if is_draw_traj_img:  # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(
                        draw_scale * x
                    ) + half_traj_img_size, half_traj_img_size - int(draw_scale * z)
                    draw_gt_x, draw_gt_y = int(
                        draw_scale * gt_x
                    ) + half_traj_img_size, half_traj_img_size - int(draw_scale * gt_z)
                    cv2.circle(
                        traj_img,
                        (draw_x, draw_y),
                        1,
                        (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0),
                        1,
                    )  # estimated from green to blue
                    cv2.circle(
                        traj_img, (draw_gt_x, draw_gt_y), 1, (0, 0, 255), 1
                    )  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(
                        traj_img,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 255, 255),
                        1,
                        8,
                    )
                    # show
                    
                    if is_draw_with_rerun:
                        Rerun.log_img_seq("trajectory_img/2d", img_id, traj_img)
                    else:
                        cv2.imshow("Trajectory", traj_img)
                
                if is_draw_with_rerun:
                    pose_rr = _convert_pose_to_rerun(vo.poses[-1]) if vo.poses else None
                    traj_est_rr = _convert_traj_to_rerun(vo.traj3d_est)
                    traj_gt_rr = _convert_traj_to_rerun(vo.traj3d_gt)
                    Rerun.log_2d_seq_scalar("trajectory_error/err_x", img_id, math.fabs(gt_x - x))
                    Rerun.log_2d_seq_scalar("trajectory_error/err_y", img_id, math.fabs(gt_y - y))
                    Rerun.log_2d_seq_scalar("trajectory_error/err_z", img_id, math.fabs(gt_z - z))
                    
                    Rerun.log_2d_seq_scalar(
                        "trajectory_stats/num_matches", img_id, vo.num_matched_kps
                    )
                    Rerun.log_2d_seq_scalar("trajectory_stats/num_inliers", img_id, vo.num_inliers)
                    
                    if pose_rr is not None:
                        Rerun.log_3d_camera_img_seq(img_id, vo.draw_img, None, cam, pose_rr)
                    if traj_est_rr is not None:
                        Rerun.log_3d_trajectory(img_id, traj_est_rr, "estimated", color=[0, 0, 255])
                    if traj_gt_rr is not None:
                        Rerun.log_3d_trajectory(img_id, traj_gt_rr, "ground_truth", color=[255, 0, 0])
                else:
                    if is_draw_3d:  # draw 3d trajectory
                        if kUsePangolin:
                            viewer3D.draw_vo(vo)
                        else:
                            plt3d.draw(vo.traj3d_gt, "ground truth", color="r", marker=".")
                            plt3d.draw(vo.traj3d_est, "estimated", color="g", marker=".")
                    
                    if is_draw_err:  # draw error signals
                        errx = [img_id, math.fabs(gt_x - x)]
                        erry = [img_id, math.fabs(gt_y - y)]
                        errz = [img_id, math.fabs(gt_z - z)]
                        err_plt.draw(errx, "err_x", color="g")
                        err_plt.draw(erry, "err_y", color="b")
                        err_plt.draw(errz, "err_z", color="r")
                    
                    if is_draw_matched_points:
                        matched_kps_signal = [img_id, vo.num_matched_kps]
                        inliers_signal = [img_id, vo.num_inliers]
                        matched_points_plt.draw(matched_kps_signal, "# matches", color="b")
                        matched_points_plt.draw(inliers_signal, "# inliers", color="g")
            
            # draw camera image
            if not is_draw_with_rerun:
                cv2.imshow("Camera", vo.draw_img)
        
        else:
            time.sleep(0.1)
        
        # get keys
        key = matched_points_plt.get_key() if matched_points_plt is not None else None
        if key == "" or key is None:
            key = err_plt.get_key() if err_plt is not None else None
        if key == "" or key is None:
            key = plt3d.get_key() if plt3d is not None else None
        
        # press 'q' to exit!
        key_cv = cv2.waitKey(1) & 0xFF
        if key == "q" or (key_cv == ord("q")):
            break
        if viewer3D and viewer3D.is_closed():
            break
        img_id += 1
    
    # print('press a key in order to exit...')
    # cv2.waitKey(0)
    
    if is_draw_traj_img:
        if not os.path.exists(kResultsFolder):
            os.makedirs(kResultsFolder, exist_ok=True)
        output_file = f"{kResultsFolder}/map_D_{intensity}_T1.png"
        print(f"saving {output_file}")
        cv2.imwrite(output_file, traj_img)
    if plt3d:
        plt3d.quit()
    if viewer3D:
        viewer3D.quit()
    if err_plt:
        err_plt.quit()
    if matched_points_plt:
        matched_points_plt.quit()
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

