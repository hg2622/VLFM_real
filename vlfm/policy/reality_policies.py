# vlfm/policy/reality_policies.py
# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor

from vlfm.mapping.obstacle_map import ObstacleMap
from vlfm.policy.base_objectnav_policy import VLFMConfig
from vlfm.policy.itm_policy import ITMPolicyV2


INITIAL_ARM_YAWS = np.deg2rad([-90, -60, -30, 0, 30, 60, 90, 0]).tolist()


class RealityMixin:
    """
    Mixin that adds real-world logic on top of ITMPolicyV2.
    """

    _stop_action: Tensor = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    _load_yolo: bool = False
    _non_coco_caption: str = (
        "chair . table . tv . laptop . microwave . toaster . sink . refrigerator . book"
        " . clock . vase . scissors . teddy bear . hair drier . toothbrush ."
    )
    _initial_yaws: List = INITIAL_ARM_YAWS.copy()
    _observations_cache: Dict[str, Any] = {}
    _policy_info: Dict[str, Any] = {}
    _done_initializing: bool = False

    def __init__(self: Union["RealityMixin", ITMPolicyV2], *args: Any, **kwargs: Any) -> None:
        # sync_explored_areas forces ITMPolicyV2 to use the obstacle map/frontiers
        super().__init__(sync_explored_areas=True, *args, **kwargs)  # type: ignore

        # ---- OPTIONAL ZoeDepth LOAD (can fail safely) ----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._depth_model = None
        try:
            self._depth_model = torch.hub.load(
                "isl-org/ZoeDepth",
                "ZoeD_NK",
                config_mode="eval",
                pretrained=True,
            ).to(device)
            print(f"[RealityMixin] ZoeDepth loaded successfully on {device}")
        except Exception as e:
            # This is where your MiDaS/timm error was happening.
            print(
                "[RealityMixin] WARNING: ZoeDepth load failed, depth inference disabled. "
                f"Error: {repr(e)}"
            )
            self._depth_model = None

        # Real robot: object point cloud can be noisy → disable DBSCAN
        self._object_map.use_dbscan = False  # type: ignore

    # ------------------------------------------------------------------
    # CONFIG LOADING (PRINT EVERYTHING WE USE)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: DictConfig, *args_unused: Any, **kwargs_unused: Any) -> Any:
        # config.policy is a DictConfig or dict from your YAML
        policy_cfg = config.policy

        # Dataclass instance with all default values
        default_cfg = VLFMConfig()

        kwargs = {}
        for k in VLFMConfig.kwaarg_names:
            if k in policy_cfg:
                kwargs[k] = policy_cfg[k]
            else:
                # fall back to dataclass default
                kwargs[k] = getattr(default_cfg, k)

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # MAIN INTERFACE
    # ------------------------------------------------------------------
    def act(
        self: Union["RealityMixin", ITMPolicyV2],
        observations: Dict[str, Any],
        rnn_hidden_states: Union[Tensor, Any],
        prev_actions: Any,
        masks: Tensor,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        # Update open-vocab caption with current target
        if observations["objectgoal"] not in self._non_coco_caption:
            self._non_coco_caption = observations["objectgoal"] + " . " + self._non_coco_caption

        parent_cls: ITMPolicyV2 = super()  # type: ignore
        action: Tensor = parent_cls.act(observations, rnn_hidden_states, prev_actions, masks, deterministic)[0]

        # For initialize phase: use yaw channel as arm yaw
        if self._done_initializing:
            action_dict = {
                "angular": action[0][0].item(),
                "linear": action[0][1].item(),
                "arm_yaw": -1,
                "info": self._policy_info,
            }
        else:
            action_dict = {
                "angular": 0,
                "linear": 0,
                "arm_yaw": action[0][0].item(),
                "info": self._policy_info,
            }

        if "rho_theta" in self._policy_info:
            action_dict["rho_theta"] = self._policy_info["rho_theta"]

        self._done_initializing = len(self._initial_yaws) == 0

        return action_dict

    def get_action(self, observations: Dict[str, Any], masks: Tensor, deterministic: bool = True) -> Dict[str, Any]:
        return self.act(observations, None, None, masks, deterministic=deterministic)

    def _reset(self: Union["RealityMixin", ITMPolicyV2]) -> None:
        parent_cls: ITMPolicyV2 = super()  # type: ignore
        parent_cls._reset()
        self._initial_yaws = INITIAL_ARM_YAWS.copy()
        self._done_initializing = False
        self._observations_cache.clear()
        self._policy_info.clear()

    def _initialize(self) -> Tensor:
        yaw = self._initial_yaws.pop(0)
        return torch.tensor([[yaw]], dtype=torch.float32)

    # ------------------------------------------------------------------
    # OBSERVATION CACHING + MAP UPDATES
    # ------------------------------------------------------------------
    def _cache_observations(self: Union["RealityMixin", ITMPolicyV2], observations: Dict[str, Any]) -> None:
        """
        Cache rgb/depth/camera transform and update obstacle + frontier maps
        every step. This is where `frontier_sensor` is produced.
        """
        self._obstacle_map: ObstacleMap

        # obstacle_map_depths: list of tuples
        # (depth_norm, tf_cam_to_world, min_d, max_d, fx, fy, fov)
        #
        # We call update_map twice:
        #  - first: explore=False, update_obstacles=True (obstacle integration)
        #  - last:  explore=True,  update_obstacles=False (fog-of-war + frontiers)
        depths = observations["obstacle_map_depths"]
        if len(depths) == 0:
            frontiers = np.array([])
        else:
            # All but last: update obstacles only
            for depth, tf, min_depth, max_depth, fx, fy, topdown_fov in depths[:-1]:
                self._obstacle_map.update_map(
                    depth,
                    tf,
                    min_depth,
                    max_depth,
                    fx,
                    fy,
                    topdown_fov,
                    explore=True,
                    update_obstacles=True,
                )

            # Last one: update explored area + frontiers only
            # depth, tf, min_depth, max_depth, fx, fy, topdown_fov = depths[-1]
            # self._obstacle_map.update_map(
            #     depth,
            #     tf,
            #     min_depth,
            #     max_depth,
            #     fx,
            #     fy,
            #     topdown_fov,
            #     explore=True,
            #     update_obstacles=False,
            # )

            # Keep camera trajectory for visualization
            self._obstacle_map.update_agent_traj(
                observations["robot_xy"],
                observations["robot_heading"],
            )
            frontiers = self._obstacle_map.frontiers

        # nav_depth: used for internal pointnav
        height, width = observations["nav_depth"].shape
        nav_depth = torch.from_numpy(observations["nav_depth"]).reshape(1, height, width, 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        nav_depth = nav_depth.to(device)

        self._observations_cache = {
            "frontier_sensor": frontiers,
            "nav_depth": nav_depth,
            "robot_xy": observations["robot_xy"],
            "robot_heading": observations["robot_heading"],
            "object_map_rgbd": observations["object_map_rgbd"],
            "value_map_rgbd": observations["value_map_rgbd"],
        }

        print(f"[REALITY] frontiers shape: {frontiers.shape}")

    # ------------------------------------------------------------------
    # DEPTH INFERENCE (ZoeDepth) – OPTIONAL
    # ------------------------------------------------------------------
    def _infer_depth(self, rgb: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
        """
        Infers the depth image from the rgb image using ZoeDepth *if available*.
        In your real pipeline we always have depth from ROS, so this should not
        be called. If it is, and ZoeDepth is disabled, we just return zeros.
        """
        if self._depth_model is None:
            print("[RealityMixin] _infer_depth called but ZoeDepth is disabled; returning zeros.")
            h, w, _ = rgb.shape
            return np.zeros((h, w), dtype=np.float32)

        img_pil = Image.fromarray(rgb)
        with torch.inference_mode():
            depth = self._depth_model.infer_pil(img_pil)
        depth = (np.clip(depth, min_depth, max_depth)) / (max_depth - min_depth)
        return depth


@dataclass
class RealityConfig(DictConfig):
    policy: VLFMConfig = field(default_factory=VLFMConfig)


class RealityITMPolicyV2(RealityMixin, ITMPolicyV2):
    """
    Concrete class combining ITMPolicyV2 with the real-world mixin.
    """
    pass
