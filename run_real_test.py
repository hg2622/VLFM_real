#!/usr/bin/env python3
import os
import threading
import time
import math
from typing import Optional, List, Tuple, Dict, Any, Callable

import numpy as np
import quaternion
import torch

import rclpy
from rclpy.node import Node

import cv2

from ros_camera_bridge import ROSCameraBridge, ROSWaypointPub
from vlfm.policy.reality_policies import RealityITMPolicyV2
from omegaconf import OmegaConf

# =======================================================
#                    VISUALIZATION HELPERS
# =======================================================


def visualize_and_save_point_cloud(point_cloud: np.ndarray, save_path: str) -> None:
    """Visualizes an array of 3D points and saves the visualization as a PNG image.

    Args:
        point_cloud (np.ndarray): Array of 3D points with shape (N, 3) or (N, >=3).
        save_path (str): Path to save the PNG image.
    """
    import matplotlib.pyplot as plt

    if point_cloud is None or point_cloud.size == 0:
        return

    pts = point_cloud[:, :3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]

    ax.scatter(x, y, z, c="b", marker="o")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def monochannel_to_inferno_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Map a single-channel array to an RGB image using the 'inferno' colormap.
    Returns uint8 RGB.
    """
    import matplotlib.pyplot as plt

    arr = arr.astype(np.float32)
    amin = float(arr.min())
    amax = float(arr.max())
    denom = (amax - amin) + 1e-8
    norm = (arr - amin) / denom

    cmap = plt.get_cmap("inferno")
    rgba = cmap(norm)               # (..., 4)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def visualize_obstacle_map(obstacle_map, save_path: str) -> None:
    """
    Re-implementation of ObstacleMap.visualize(self) using the object's internals.
    """
    vis_img = np.ones((*obstacle_map._map.shape[:2], 3), dtype=np.uint8) * 255

    # Draw explored area in light green
    vis_img[obstacle_map.explored_area == 1] = (200, 255, 200)

    # Draw unnavigable areas in gray (radius padding color)
    vis_img[obstacle_map._navigable_map == 0] = obstacle_map.radius_padding_color

    # Draw obstacles in black
    vis_img[obstacle_map._map == 1] = (0, 0, 0)

    # Draw frontiers in blue-ish
    for frontier in getattr(obstacle_map, "_frontiers_px", []):
        cv2.circle(
            vis_img,
            tuple(int(i) for i in frontier),
            5,
            (200, 0, 0),
            2,
        )

    # Flip vertically to match their convention
    vis_img = cv2.flip(vis_img, 0)

    # Draw trajectory if available
    if len(getattr(obstacle_map, "_camera_positions", [])) > 0:
        obstacle_map._traj_vis.draw_trajectory(
            vis_img,
            obstacle_map._camera_positions,
            obstacle_map._last_camera_yaw,
        )

    cv2.imwrite(save_path, vis_img)


def visualize_value_map(value_map, obstacle_map, save_path: str) -> None:
    """
    Re-implementation of ValueMap.visualize(self, obstacle_map=...)
    """
    vm = value_map._value_map
    reduced_map = np.max(vm, axis=-1).copy()

    if obstacle_map is not None:
        reduced_map[obstacle_map.explored_area == 0] = 0

    # Must negate y to get correct orientation
    map_img = np.flipud(reduced_map)

    # Handle zeros
    zero_mask = map_img == 0
    if np.any(map_img):
        map_img[zero_mask] = np.max(map_img)

    map_img_rgb = monochannel_to_inferno_rgb(map_img)
    map_img_rgb[zero_mask] = (255, 255, 255)

    # Draw trajectory if available
    if len(getattr(value_map, "_camera_positions", [])) > 0:
        value_map._traj_vis.draw_trajectory(
            map_img_rgb,
            value_map._camera_positions,
            value_map._last_camera_yaw,
        )

    # Save as BGR for OpenCV
    cv2.imwrite(save_path, map_img_rgb[..., ::-1])


# =======================================================
#                  GEOMETRY / OBS HELPERS
# =======================================================

def quat_to_yaw(q: np.quaternion) -> float:
    """Extract yaw from quaternion (world frame)."""
    w, x, y, z = q.w, q.x, q.y, q.z
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def pose_to_tf(position: np.ndarray, rot: np.quaternion) -> np.ndarray:
    """4x4 transform matrix from position + quaternion."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quaternion.as_rotation_matrix(rot)
    T[:3, 3] = position.astype(np.float32)
    return T


def build_vlfm_obs(ros_obs: Dict[str, Any], target_object: str = "chair") -> Optional[Dict[str, Any]]:
    """
    ros_obs is whatever ROSCameraBridge.get_obs(...) returns.
    We expect keys:
      - "rgb":      HxWx3 uint8
      - "depth":    HxW float or uint16 (meters)
      - "position": np.array([x,y,z], float32)
      - "rotation": np.quaternion(w,x,y,z)
    """
    rgb = ros_obs.get("rgb", None)
    depth_raw = ros_obs.get("depth", None)
    pos = ros_obs.get("position", None)
    rot = ros_obs.get("rotation", None)

    if rgb is None or depth_raw is None or pos is None or rot is None:
        return None

    # ---- Interpret ROS depth as meters, normalize to [0,1] ----
    depth_m = depth_raw.astype(np.float32)

    # Configure your range here
    min_d = 0.1
    max_d = 5.0

    depth_norm = (depth_m - min_d) / (max_d - min_d)
    depth_norm = np.clip(depth_norm, 0.0, 1.0)
    depth_norm = np.nan_to_num(depth_norm, nan=0.0)

    robot_xy = pos[:2]
    robot_heading = quat_to_yaw(rot)
    tf_cam_to_world = pose_to_tf(pos, rot)

    # 360 camera: FOV = 2π
    fov = 2 * math.pi

    # fx, fy are not used in our spherical projection; kept for API compat and API signature
    fx = fy = 1.0

    # IMPORTANT: for RealityMixin._cache_observations we provide TWO entries:
    #  - first: used to update obstacles (explore=False, update_obstacles=True)
    #  - last:  used to update explored area/frontiers (explore=True, update_obstacles=False)
    obstacle_entries = [
        (depth_norm, tf_cam_to_world, min_d, max_d, fx, fy, fov),
        (depth_norm, tf_cam_to_world, min_d, max_d, fx, fy, fov),
    ]

    obs = {
        "nav_depth": depth_norm,
        "robot_xy": robot_xy,
        "robot_heading": robot_heading,
        "objectgoal": target_object,

        "obstacle_map_depths": obstacle_entries,

        # For semantic value map (BLIP2)
        "value_map_rgbd": [
            (rgb, depth_norm, tf_cam_to_world, min_d, max_d, fov)
        ],

        # For object map (YOLO + SAM)
        "object_map_rgbd": [
            (rgb, depth_norm, tf_cam_to_world, min_d, max_d, fx, fy)
        ],
    }

    return obs


# =======================================================
#                      MAIN NODE
# =======================================================

def main(args=None):
    # --------- Timing / visualization config ----------
    rate_hz = 2.0
    period = 1.0 / rate_hz

    VIS_SAVE_PERIOD = 10      # save visualizations every N steps
    VIS_DIR = "vlfm_debug_vis"
    os.makedirs(VIS_DIR, exist_ok=True)

    target_object = "chair"   # change as needed

    # ---------------- ROS init + shared node ----------------
    rclpy.init(args=args)
    shared_node: Node = rclpy.create_node("vlfm_real")

    print("[VLFM] Created shared ROS node 'vlfm_real'")

    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(shared_node,),
        daemon=True,
    )
    spin_thread.start()
    print("[VLFM] spin_thread started (ROS callbacks running in background)")

    # ---------------- Camera bridge + waypoint pub ----------------
    cam = ROSCameraBridge(
        shared_node,
        rgb_topic="/habitat/rgb",
        depth_topic="/habitat/depth",
        pose_topic="/habitat/state_estimation",
    )

    waypoint_pub = ROSWaypointPub(
        node=shared_node,
        topic="/way_point",
        frame_id="map",
    )

    print("[VLFM] ROSCameraBridge and ROSWaypointPub initialized")

    # ---------------- Load VLFM reality policy ----------------
    cfg = OmegaConf.load("config/experiments/reality.yaml")
    policy = RealityITMPolicyV2.from_config(cfg)
    print("[VLFM] RealityITMPolicyV2 loaded from config/experiments/reality.yaml")

    # (1,1) mask like in habitat baselines
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

    step_idx = 0

    def maybe_save_visualizations(step: int):
        if step % 1 != 0:
            return

        try:
            # Obstacle map
            if hasattr(policy, "_obstacle_map") and policy._obstacle_map is not None:
                ob_path = os.path.join(VIS_DIR, f"obstacle_map_{step:05d}.png")
                visualize_obstacle_map(policy._obstacle_map, ob_path)

            # Value map
            if hasattr(policy, "_value_map") and policy._value_map is not None:
                val_path = os.path.join(VIS_DIR, f"value_map_{step:05d}.png")
                visualize_value_map(policy._value_map, policy._obstacle_map, val_path)

            # Target object point cloud
            cloud = policy._policy_info.get("target_point_cloud", None)
            if cloud is not None and len(cloud) > 0:
                cloud_path = os.path.join(VIS_DIR, f"target_cloud_{step:05d}.png")
                visualize_and_save_point_cloud(cloud, cloud_path)

        except Exception as e:
            print(f"[VLFM] Visualization error at step {step}: {e}")

    def tick():
        nonlocal mask, step_idx

        step_idx += 1

        ros = cam.get_obs(allow_spin=False)
        if ros is None:
            return True

        rgb = ros.get("rgb", None)
        depth_raw = ros.get("depth", None)
        pos = ros.get("position", None)
        rot = ros.get("rotation", None)
        if rgb is None or depth_raw is None or pos is None or rot is None:
            return True

        obs = build_vlfm_obs(ros, target_object=target_object)
        if obs is None:
            return True

        # Get action dict from VLFM policy
        action_dict = policy.get_action(obs, mask, deterministic=True)
        mask[...] = True

        # Publish waypoint if available
        nav_goal = policy._policy_info.get("nav_goal", None)
        if nav_goal is not None:
            gx, gy = nav_goal[:2]
            waypoint_pub.publish_xyz_ros([gx, gy, 0.0])
            print(f"[VLFM] Step {step_idx}: Published waypoint ({gx:.2f}, {gy:.2f})")

        # Occasionally dump some debug vis
        maybe_save_visualizations(step_idx)

        if policy._policy_info.get("target_found", False):
            print(f"[VLFM] Step {step_idx}: Target found, stopping loop.")
            return False

        return True

    try:
        while rclpy.ok():
            if not tick():
                break
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        shared_node.destroy_node()
        rclpy.shutdown()
        print("[VLFM] Shutdown complete")


if __name__ == "__main__":
    main()
