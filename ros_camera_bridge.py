import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import threading
import time

from rclpy.time import Time

import quaternion  # enables np.quaternion(...)
from geometry_msgs.msg import PoseStamped




# class ROSCamera:
#     """
#     Holds latest RGB/Depth/Pose and registers subscriptions on an EXISTING node.
#     (No rclpy.init(), no spin, no Node subclassing.)
#     """
#     def __init__(self,
#                  node: Node,
#                  rgb_topic="/habitat/rgb",
#                  depth_topic="/habitat/depth",
#                  pose_topic="/habitat/state_estimation"):
#         self.node = node
#         self.bridge = CvBridge()
#         self.rgb = None
#         self.depth = None
#         self.position = None
#         self.rotation = None
#         self.lock = threading.Lock()

#         self.node.create_subscription(Image,  rgb_topic,  self.rgb_callback,   10)
#         self.node.create_subscription(Image,  depth_topic, self.depth_callback, 10)
#         self.node.create_subscription(PoseStamped, pose_topic, self.pose_callback, 10)
#         # self.node.create_subscription(Odometry, pose_topic, self.pose_callback, 10)


#     def rgb_callback(self, msg):
#         with self.lock:
#             self.rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             self.rgb_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

#     def depth_callback(self, msg):
#         with self.lock:
#             self.depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
#             self.depth_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

#     def pose_callback(self, msg: PoseStamped):
#         with self.lock:
#             self.position = np.array([
#                 msg.pose.position.x,
#                 msg.pose.position.y,
#                 msg.pose.position.z
#             ], dtype=np.float32)
#             q = msg.pose.orientation
#             # numpy-quaternion expects (w, x, y, z)
#             self.rotation = np.quaternion(q.w, q.x, q.y, q.z)
#             self.odom_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

#     # def pose_callback(self, msg: Odometry):
#     #     with self.lock:
#     #         p = msg.pose.pose.position
#     #         q = msg.pose.pose.orientation
#     #         self.position = np.array([p.x, p.y, p.z], dtype=np.float32)
#     #         # numpy-quaternion expects (w, x, y, z)
#     #         self.rotation = np.quaternion(q.w, q.x, q.y, q.z)
#     #         self.odom_stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9

#         # (optional) keep velocities if useful later:
#         # v = msg.twist.twist.linear;  w = msg.twist.twist.angular
#         # self.linear_vel  = np.array([v.x, v.y, v.z], dtype=np.float32)
#         # self.angular_vel = np.array([w.x, w.y, w.z], dtype=np.float32)



# class ROSCameraBridge:
#     """
#     Uses the shared node from the runner. No rclpy.init(), no spin, no node creation here.
#     """
#     def __init__(self, node: Node,
#                  rgb_topic="/habitat/rgb",
#                  depth_topic="/habitat/depth",
#                  pose_topic="/state_estimation"):
#         self.node = node
#         self.camera = ROSCamera(node, rgb_topic, depth_topic, pose_topic)

#     def get_obs(self, timeout_sec=5.0, spin_period=0.05):
#         """
#         Wait until we have rgb, depth, and pose, spinning the node so callbacks fire.
#         Returns a copy of the latest observation dict, or raises TimeoutError.
#         """
#         t0 = time.time()
#         while True:
#             # allow subscriber callbacks to run
#             rclpy.spin_once(self.node, timeout_sec=spin_period)

#             with self.camera.lock:
#                 ready = (self.camera.rgb is not None and
#                         self.camera.depth is not None and
#                         self.camera.position is not None and
#                         self.camera.rotation is not None)
#                 if ready:
#                     return {
#                         "rgb":      self.camera.rgb.copy(),z
#                         "depth":    self.camera.depth.copy(),
#                         "position": self.camera.position.copy(),
#                         "rotation": self.camera.rotation.copy(),
#                         "stamps": {
#                             "rgb":   self.camera.rgb_stamp,
#                             "depth": self.camera.depth_stamp,
#                             "odom":  self.camera.odom_stamp,
#                         },
#                     }

#             if time.time() - t0 > timeout_sec:
#                 # Helpful diagnosis: tell which streams are missing
#                 missing = []
#                 if self.camera.rgb is None:   missing.append("rgb")
#                 if self.camera.depth is None: missing.append("depth")
#                 if self.camera.position is None or self.camera.rotation is None:
#                     missing.append("pose")
#                 raise TimeoutError(f"ROS obs timeout after {timeout_sec}s; missing: {', '.join(missing) or 'none'}")



import time, threading
from collections import deque
import numpy as np
import quaternion
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

WINDOW_NS = 80_000_000  # 0.08 s in nanoseconds

class ROSCamera:
    """
    Holds latest RGB plus small caches for Depth & Odom (Pose) samples.
    Subscriptions attach to an EXISTING node.
    """
    def __init__(self,
                 node: Node,
                 rgb_topic="/habitat/rgb",
                 depth_topic="/habitat/depth",
                 pose_topic="/habitat/state_estimation",
                 cache_len: int = 50):
        self.node = node
        self.bridge = CvBridge()

        # Latest RGB and an RGB cache for timestamp matching.
        self.rgb = None
        self.rgb_stamp_ns = None
        self.rgb_buf = deque(maxlen=cache_len)

        # Latest registered scan (PointCloud2), not synchronized with anything
        self.scan = None
        self.scan_stamp_ns = None

        # Caches for depth and odom
        # depth_buf entries: (stamp_ns, depth_np)
        # odom_buf  entries: (stamp_ns, (pos_np[3], quat))
        self.depth_buf = deque(maxlen=cache_len)
        self.odom_buf  = deque(maxlen=cache_len)

        self.lock = threading.Lock()

        # Keep your proven-good QoS = 10 (default RELIABLE)
        self.node.create_subscription(Image,       rgb_topic,   self.rgb_callback,   10)
        self.node.create_subscription(Image,       depth_topic, self.depth_callback, 10)
        self.node.create_subscription(PoseStamped, pose_topic,  self.pose_callback,  10)


        # Registered scan topic (PointCloud2). No time matching; we just keep the latest.
        self.node.create_subscription(PointCloud2,  '/registered_scan', self.scan_callback, 10)
    def rgb_callback(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.lock:
            self.rgb = img
            self.rgb_stamp_ns = stamp_ns
            self.rgb_buf.append((stamp_ns, img))

    def depth_callback(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.lock:
            self.depth_buf.append((stamp_ns, depth))

    def pose_callback(self, msg: PoseStamped):
        p = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                     dtype=np.float32)
        q = msg.pose.orientation
        rot = np.quaternion(q.w, q.x, q.y, q.z)
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.lock:
            self.odom_buf.append((stamp_ns, (p, rot)))

    def scan_callback(self, msg: PointCloud2):
        # Not synchronized; keep the latest scan message.
        stamp_ns = Time.from_msg(msg.header.stamp).nanoseconds
        with self.lock:
            self.scan = msg
            self.scan_stamp_ns = stamp_ns



# --- Drop-in replacement for ROSCameraBridge (ROSCamera unchanged) ---

import time
from collections import deque

class ROSCameraBridge:
    """
    Snapshot-based matcher:
      1) Take (depth, odom, rgb) SNAPSHOTS under a very short lock.
      2) Release the lock and do the matching work on those snapshots.
      3) Re-lock briefly to verify/prune and return a *new* pair only.
    """
    def __init__(self, node: Node,
                 rgb_topic="/habitat/rgb",
                 depth_topic="/habitat/depth",
                 pose_topic="/state_estimation",
                 cache_len: int = 50):
        self.node = node
        self._last_depth_ns = None
        self._last_odom_ns  = None
        self.camera = ROSCamera(node, rgb_topic, depth_topic, pose_topic, cache_len=cache_len)

    # ---------- Helpers on snapshots (no lock inside) ----------
    def _fmt_ns(self, ns: int | None):
        return "None" if ns is None else f"{ns} ({ns/1e9:.6f}s)"

    def _status_from_snapshots(self, depth_list, odom_list, rgb_ns):
        depth_stamps = [s for s,_ in depth_list]
        odom_stamps  = [s for s,_ in odom_list]

        depth_n = len(depth_stamps)
        odom_n  = len(odom_stamps)

        best_dt = None
        if depth_n and odom_n:
            j = odom_n - 1
            for ds in reversed(depth_stamps):
                while j > 0 and abs(odom_stamps[j-1] - ds) <= abs(odom_stamps[j] - ds):
                    j -= 1
                dt = abs(odom_stamps[j] - ds)
                best_dt = dt if best_dt is None or dt < best_dt else best_dt

        return {
            "depth_n": depth_n,
            "odom_n":  odom_n,
            "rgb_ns":  rgb_ns,
            "depth_min_ns": min(depth_stamps) if depth_n else None,
            "depth_max_ns": max(depth_stamps) if depth_n else None,
            "odom_min_ns":  min(odom_stamps)  if odom_n  else None,
            "odom_max_ns":  max(odom_stamps)  if odom_n  else None,
            "best_dt_ns":   best_dt,
            "best_dt_ms":   (best_dt / 1e6) if best_dt is not None else None,
        }

    def _find_closest_rgb_on_list(self, rgb_list, target_stamp_ns: int, window_ns: int):
        """Find RGB frame closest to target stamp within window_ns.

        Returns:
            (rgb_stamp_ns, rgb_img) or None
        """
        if not rgb_list:
            return None

        best = None
        best_dt = None
        for r_stamp, r_img in reversed(rgb_list):
            dt = abs(r_stamp - target_stamp_ns)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best = (r_stamp, r_img)
            if dt == 0:
                break

        if best is None or best_dt is None or best_dt > window_ns:
            return None
        return best

    def _find_synced_pair_on_lists(self, depth_list, odom_list, window_ns: int):
        """
        Given SNAPSHOT lists, find newest depth matched to closest odom within window.
        Returns (depth_stamp_ns, depth_img, odom_stamp_ns, position, rotation) or None.
        """
        if not depth_list or not odom_list:
            return None

        # Newest→oldest depth; for each, find closest odom (also newest→oldest)
        for d_stamp, d_img in reversed(depth_list):
            closest = None
            closest_dt = None
            for o_stamp, (pos, rot) in reversed(odom_list):
                dt = abs(d_stamp - o_stamp)
                if closest_dt is None or dt < closest_dt:
                    closest_dt = dt
                    closest = (o_stamp, pos, rot)
                if closest_dt == 0:
                    break
            if closest is not None and closest_dt <= window_ns:
                o_stamp, pos, rot = closest
                return (d_stamp, d_img, o_stamp, pos, rot)
        return None

    # ---------- Main API ----------
    def get_obs(self,
                timeout_sec: float = 5.0,
                spin_period: float = 0.05,
                window_sec: float = 0.08,
                allow_spin: bool = True,
                copy: bool = True,
                verbose_every: int = 10):
        """
        Returns only when |depth - odom| <= window_sec, using the snapshot matcher.
        Also returns a 'debug' dict with useful timing metrics.
        """
        window_ns = int(window_sec * 1e9)
        t0 = time.time()
        it = 0

        while True:
            if allow_spin:
                rclpy.spin_once(self.node, timeout_sec=spin_period)
            it += 1

            # 1) SNAPSHOT quickly under the lock (don’t do heavy work here)
            with self.camera.lock:
                depth_list = list(self.camera.depth_buf)
                odom_list  = list(self.camera.odom_buf)
                rgb_list   = list(self.camera.rgb_buf)
                rgb_img    = self.camera.rgb
                rgb_ns     = self.camera.rgb_stamp_ns

            # Optional heartbeat (outside the lock)
            if verbose_every and (it % verbose_every == 0):
                st = self._status_from_snapshots(depth_list, odom_list, rgb_ns)
                # self.node.get_logger().info(
                #     "[SYNC WAIT] dN=%d oN=%d | d=[%s .. %s] o=[%s .. %s] best=%s ms, rgb=%s, window=%.1f ms"
                #     % (st["depth_n"], st["odom_n"],
                #        self._fmt_ns(st["depth_min_ns"]), self._fmt_ns(st["depth_max_ns"]),
                #        self._fmt_ns(st["odom_min_ns"]),  self._fmt_ns(st["odom_max_ns"]),
                #        f"{st['best_dt_ms']:.2f}" if st["best_dt_ms"] is not None else "None",
                #        self._fmt_ns(st["rgb_ns"]),
                #        window_sec*1e3)
                # )

            # 2) MATCH on the snapshots (no lock held)
            matched = self._find_synced_pair_on_lists(depth_list, odom_list, window_ns)
            if matched is None:
                if time.time() - t0 > timeout_sec:
                    st = self._status_from_snapshots(depth_list, odom_list, rgb_ns)
                    # self.node.get_logger().warn(
                    #     "[SYNC TIMEOUT] dN=%d oN=%d | d=[%s .. %s] o=[%s .. %s] best=%s ms, rgb=%s, need <= %.1f ms"
                    #     % (st["depth_n"], st["odom_n"],
                    #        self._fmt_ns(st["depth_min_ns"]), self._fmt_ns(st["depth_max_ns"]),
                    #        self._fmt_ns(st["odom_min_ns"]),  self._fmt_ns(st["odom_max_ns"]),
                    #        f"{st['best_dt_ms']:.2f}" if st["best_dt_ms"] is not None else "None",
                    #        self._fmt_ns(st["rgb_ns"]),
                    #        window_sec*1e3)
                    # )
                    missing = []
                    if st["depth_n"] == 0: missing.append("depth")
                    if st["odom_n"]  == 0: missing.append("odom")
                    raise TimeoutError("Timeout: missing %s" % (", ".join(missing) or "synced pair"))
                continue

            d_stamp, d_img, o_stamp, pos, rot = matched

            # Match RGB to depth stamp to keep detection masks aligned with depth.
            rgb_matched = self._find_closest_rgb_on_list(rgb_list, d_stamp, window_ns)
            if rgb_matched is None:
                if time.time() - t0 > timeout_sec:
                    raise TimeoutError("Timeout: no RGB frame near matched depth stamp")
                continue
            rgb_stamp_ns, rgb_img_from_stamp = rgb_matched

            # 3) New-pair gate (avoid returning the same pair twice)
            if d_stamp == self._last_depth_ns and o_stamp == self._last_odom_ns:
                if time.time() - t0 > timeout_sec:
                    raise TimeoutError("Timeout: no new synced pair")
                continue

            dt_ms = abs(d_stamp - o_stamp) / 1e6

            # 4) RE-LOCK briefly to verify matched stamps still present, PRUNE, then return
            with self.camera.lock:
                # If another thread advanced buffers, the stamps might be gone; if so, loop again
                if not any(s == d_stamp for s,_ in self.camera.depth_buf) or \
                   not any(s == o_stamp for s,_ in self.camera.odom_buf):
                    continue

                # Mark as last returned
                self._last_depth_ns = d_stamp
                self._last_odom_ns  = o_stamp

                # PRUNE: drop everything up to and including matched stamps
                self.camera.depth_buf = deque(
                    [(s, img) for (s, img) in self.camera.depth_buf if s > d_stamp],
                    maxlen=self.camera.depth_buf.maxlen
                )
                self.camera.odom_buf = deque(
                    [(s, val) for (s, val) in self.camera.odom_buf if s > o_stamp],
                    maxlen=self.camera.odom_buf.maxlen
                )

                # Update RGB again in case it advanced since the snapshot
                rgb_ns_cur  = self.camera.rgb_stamp_ns
                rgb_img_cur = self.camera.rgb

                # Registered scan (not synced)
                scan_msg_cur = getattr(self.camera, 'scan', None)
                scan_ns_cur  = getattr(self.camera, 'scan_stamp_ns', None)

            # Log success (outside the lock)
            # self.node.get_logger().info(
            #     "[SYNC OK] |Δt(depth-odom)|=%.2f ms  depth=%s  odom=%s  rgb=%s"
            #     % (dt_ms, self._fmt_ns(d_stamp), self._fmt_ns(o_stamp), self._fmt_ns(rgb_ns_cur))
            # )

            # Build return payload
            rgb_src = rgb_img_from_stamp if rgb_img_from_stamp is not None else rgb_img_cur
            rgb   = rgb_src.copy() if (copy and rgb_src is not None) else rgb_src
            depth = d_img.copy()        if copy else d_img
            pos_o = pos.copy()          if copy else pos

            # Use the *current* debug based on the latest snapshot we have now
            debug_st = self._status_from_snapshots(depth_list, odom_list, rgb_ns_cur)

            return {
                "rgb": rgb,
                "depth": depth,
                "scan": scan_msg_cur,
                "position": pos_o,
                "rotation": rot,
                "stamps": {
                    "rgb":   rgb_stamp_ns,
                    "depth": d_stamp,
                    "odom":  o_stamp,
                    "scan":  scan_ns_cur,
                    "dt_ms":    dt_ms,
                },
                "debug": debug_st
            }

# --- add after ROSCameraBridge class ---


# import time
# import threading
# from collections import deque
# from dataclasses import dataclass
# from typing import Optional, Tuple, Dict

# import numpy as np
# import quaternion  # pip install numpy-quaternion
# import rclpy
# from rclpy.node import Node
# from rclpy.time import Time

# from sensor_msgs.msg import Image, PointCloud2
# from geometry_msgs.msg import PoseStamped
# from nav_msgs.msg import Odometry
# from cv_bridge import CvBridge


# # ------------- Small utilities -------------
# def ros_stamp_to_sec(stamp) -> float:
#     """Convert builtin_interfaces/Time to float seconds."""
#     # Works for both msg.header.stamp and bare 'stamp'
#     if hasattr(stamp, "sec"):
#         return stamp.sec + stamp.nanosec * 1e-9
#     # Or if we got a header
#     return Time.from_msg(stamp).nanoseconds * 1e-9


# def _find_closest_by_time(buffer: deque, target_t: float) -> Optional[Tuple[float, object]]:
#     """
#     Given a deque of (t, data), find the entry with |t - target_t| minimal.
#     Returns (t, data) or None if buffer is empty.
#     """
#     if not buffer:
#         return None
#     # Small buffers are fine with linear scan; robust and simple.
#     best = None
#     best_abs = float("inf")
#     for (t, data) in buffer:
#         d = abs(t - target_t)
#         if d < best_abs:
#             best_abs = d
#             best = (t, data)
#     return best


# @dataclass
# class SyncedObs:
#     rgb: np.ndarray
#     depth: np.ndarray
#     position: np.ndarray        # (3,)
#     rotation: np.quaternion     # (w,x,y,z)
#     stamps: Dict[str, float]    # {"rgb":.., "depth":.., "odom":..}


# # ------------- Camera subscriber with caches -------------
# class ROSCamera:
#     """
#     Subscribes to RGB / Depth / Pose (odom), caching recent frames & timestamps.
#     No rclpy.init() or Node subclassing here; you pass in an existing Node.
#     """

#     def __init__(self,
#                  node: Node,
#                  rgb_topic: str = "/habitat/rgb",
#                  depth_topic: str = "/habitat/depth",
#                  pose_topic: str = "/state_estimation",
#                  pose_msg_type: str = "PoseStamped",
#                  maxlen: int = 200,
#                  max_age_sec: float = 3.0):
#         """
#         pose_msg_type: "PoseStamped" or "Odometry"
#         maxlen: max items kept in each buffer
#         max_age_sec: drop entries older than (now - max_age_sec)
#         """
#         self.node = node
#         self.bridge = CvBridge()

#         # Buffers store tuples: (stamp_sec, data)
#         self.rgb_buf: deque[Tuple[float, np.ndarray]] = deque(maxlen=maxlen)
#         self.depth_buf: deque[Tuple[float, np.ndarray]] = deque(maxlen=maxlen)

#         # Latest odom/pose (we only keep the *latest* pose)
#         self.position: Optional[np.ndarray] = None
#         self.rotation: Optional[np.quaternion] = None
#         self.odom_stamp: Optional[float] = None

#         self.max_age_sec = max_age_sec
#         self.lock = threading.Lock()

#         self.node.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
#         self.node.create_subscription(Image, depth_topic, self.depth_callback, 10)

#         if pose_msg_type == "PoseStamped":
#             self.node.create_subscription(PoseStamped, pose_topic, self.pose_callback_ps, 10)
#         elif pose_msg_type == "Odometry":
#             self.node.create_subscription(Odometry, pose_topic, self.pose_callback_odom, 10)
#         else:
#             raise ValueError("pose_msg_type must be 'PoseStamped' or 'Odometry'")

#     # ---- Callbacks ----
#     def rgb_callback(self, msg: Image):
#         stamp_sec = ros_stamp_to_sec(msg.header.stamp)
#         img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#         with self.lock:
#             self.rgb_buf.append((stamp_sec, img))
#             self._purge_old()

#     def depth_callback(self, msg: Image):
#         stamp_sec = ros_stamp_to_sec(msg.header.stamp)
#         # "passthrough" keeps float16/float32 meters as-is (or uint16 mm if that’s your source)
#         depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
#         with self.lock:
#             self.depth_buf.append((stamp_sec, depth))
#             self._purge_old()

#     def pose_callback_ps(self, msg: PoseStamped):
#         stamp_sec = ros_stamp_to_sec(msg.header.stamp)
#         p = msg.pose.position
#         q = msg.pose.orientation
#         with self.lock:
#             self.position = np.array([p.x, p.y, p.z], dtype=np.float32)
#             self.rotation = np.quaternion(q.w, q.x, q.y, q.z)
#             self.odom_stamp = stamp_sec
#             self._purge_old()

#     def pose_callback_odom(self, msg: Odometry):
#         stamp_sec = ros_stamp_to_sec(msg.header.stamp)
#         p = msg.pose.pose.position
#         q = msg.pose.pose.orientation
#         with self.lock:
#             self.position = np.array([p.x, p.y, p.z], dtype=np.float32)
#             self.rotation = np.quaternion(q.w, q.x, q.y, q.z)
#             self.odom_stamp = stamp_sec
#             self._purge_old()

#     # ---- Housekeeping ----
#     def _purge_old(self):
#         """Drop frames older than now - max_age_sec to keep memory in check."""
#         now = time.time()
#         cutoff = now - self.max_age_sec

#         def _left_trim(buf: deque):
#             while buf and buf[0][0] < cutoff:
#                 buf.popleft()

#         _left_trim(self.rgb_buf)
#         _left_trim(self.depth_buf)

#     # ---- Pull a synchronized packet ----
#     def get_synced(self,
#                    sync_tolerance_sec: float = 0.01,
#                    require_new_odom: bool = True,
#                    last_odom_seen: Optional[float] = None) -> Optional[SyncedObs]:
#         """
#         Pair the *latest* pose (odom) with the *closest* rgb & depth frames in the buffers.

#         sync_tolerance_sec: max |t_frame - t_odom| allowed for both rgb and depth.
#         require_new_odom: if True, ignore when odom hasn't advanced beyond last_odom_seen.
#         last_odom_seen: pass your last served odom stamp if you want strictly increasing odom.

#         Returns SyncedObs or None if not ready / no good match.
#         """
#         with self.lock:
#             if self.position is None or self.rotation is None or self.odom_stamp is None:
#                 return None
#             if not self.rgb_buf or not self.depth_buf:
#                 return None

#             t_odom = self.odom_stamp
#             if require_new_odom and last_odom_seen is not None and t_odom <= last_odom_seen:
#                 return None

#             rgb_pick = _find_closest_by_time(self.rgb_buf, t_odom)
#             depth_pick = _find_closest_by_time(self.depth_buf, t_odom)
#             if rgb_pick is None or depth_pick is None:
#                 return None

#             t_rgb, rgb = rgb_pick
#             t_depth, depth = depth_pick

#             if abs(t_rgb - t_odom) > sync_tolerance_sec or abs(t_depth - t_odom) > sync_tolerance_sec:
#                 # Not within tolerance; caller may spin/wait and try again.
#                 return None

#             return SyncedObs(
#                 rgb=rgb.copy(),
#                 depth=depth.copy(),
#                 position=self.position.copy(),
#                 rotation=self.rotation.copy(),
#                 stamps={"rgb": t_rgb, "depth": t_depth, "odom": t_odom},
#             )


# # ------------- Bridge that spins & serves synced obs -------------
# class ROSCameraBridge:
#     """
#     Uses an existing shared Node. get_obs() spins briefly so callbacks fire,
#     then returns an odom-synchronized packet (closest rgb/depth within tolerance).
#     """

#     def __init__(self, node: Node,
#                  rgb_topic="/habitat/rgb",
#                  depth_topic="/habitat/depth",
#                  pose_topic="/state_estimation",
#                  pose_msg_type: str = "PoseStamped",
#                  maxlen: int = 50,
#                  max_age_sec: float = 3.0):
#         self.node = node
#         self.camera = ROSCamera(
#             node, rgb_topic, depth_topic, pose_topic,
#             pose_msg_type=pose_msg_type, maxlen=maxlen, max_age_sec=max_age_sec
#         )
#         self._last_odom_served: Optional[float] = None  # to avoid repeating the same odom

#     def get_obs(self,
#                 timeout_sec: float = 5.0,
#                 spin_period: float = 0.001,
#                 sync_tolerance_sec: float = 0.080,
#                 require_new_odom: bool = False) -> Dict:
#         """
#         Spin until we can form a synchronized (odom,rgb,depth) packet or timeout.

#         - timeout_sec: max wall-time to wait.
#         - spin_period: time passed to rclpy.spin_once(..., timeout_sec=spin_period)
#         - sync_tolerance_sec: max |t_frame - t_odom| allowed for both streams
#         - require_new_odom: if True, don't reuse the same odom twice

#         Returns a dict { "rgb", "depth", "position", "rotation", "stamps" }
#         Raises TimeoutError on timeout (with a helpful message).
#         """
#         t0 = time.time()
#         while True:
#             rclpy.spin_once(self.node, timeout_sec=spin_period)

#             synced = self.camera.get_synced(
#                 sync_tolerance_sec=sync_tolerance_sec,
#                 require_new_odom=require_new_odom,
#                 last_odom_seen=self._last_odom_served
#             )
#             if synced is not None:
#                 self._last_odom_served = synced.stamps["odom"]
#                 return {
#                     "rgb": synced.rgb,
#                     "depth": synced.depth,
#                     "position": synced.position,
#                     "rotation": synced.rotation,
#                     "stamps": synced.stamps,
#                 }

#             if time.time() - t0 > timeout_sec:
#                 # Construct a helpful message about which parts are missing / out of tolerance
#                 with self.camera.lock:
#                     have_pose = (self.camera.position is not None and
#                                  self.camera.rotation is not None and
#                                  self.camera.odom_stamp is not None)
#                     n_rgb = len(self.camera.rgb_buf)
#                     n_depth = len(self.camera.depth_buf)
#                     odom_str = (f"{self.camera.odom_stamp:.6f}" if self.camera.odom_stamp else "none")

#                 raise TimeoutError(
#                     "ROS obs timeout.\n"
#                     f"- pose/odom available: {have_pose} (odom_stamp={odom_str})\n"
#                     f"- rgb cached: {n_rgb} frames, depth cached: {n_depth} frames\n"
#                     f"- tolerance: ±{sync_tolerance_sec*1000:.1f} ms; "
#                     f"require_new_odom={require_new_odom}"
#                 )


from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker


class ROSWaypointPub:
    """
    Reuse the SAME shared node. Do not create a new node here.
    """
    def __init__(self, node: Node, topic="/way_point", marker_topic="/waypoint_marker", frame_id="map"):
        self.node = node
        self.frame_id = frame_id
        self.pub = self.node.create_publisher(PointStamped, topic, 10)
        self.marker_pub = self.node.create_publisher(Marker, marker_topic, 10)
        self._seq_id = 0  # unique id per waypoint so RViz keeps multiple

    def publish_xyz_ros(self, xyz, frame_id=None, label=True, radius=0.12, lifetime_sec=0.0):
        frame = frame_id or self.frame_id

        # 1) PointStamped (your original)
        msg = PointStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        self.pub.publish(msg)

        # 2) Yellow sphere marker
        m = Marker()
        m.header = msg.header
        m.ns = "waypoints"
        m.id = self._seq_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = msg.point
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = float(radius * 2.0)  # diameter
        m.color.r = 1.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 1.0  # yellow
        if lifetime_sec > 0:
            m.lifetime.sec = int(lifetime_sec)
        self.marker_pub.publish(m)

        # 3) XYZ text label (optional, above the sphere)
        if label:
            t = Marker()
            t.header = msg.header
            t.ns = "waypoints_text"
            t.id = self._seq_id
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position = Point(
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2] + radius + 0.15),
            )
            t.pose.orientation.w = 1.0
            t.scale.z = 0.22  # text height in meters
            t.color.r = 1.0; t.color.g = 1.0; t.color.b = 0.0; t.color.a = 1.0  # yellow
            t.text = f"x={xyz[0]:.2f}, y={xyz[1]:.2f}, z={xyz[2]:.2f}"
            if lifetime_sec > 0:
                t.lifetime.sec = int(lifetime_sec)
            self.marker_pub.publish(t)

        self._seq_id += 1