#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import std_msgs.msg
import threading
import time
import json
import math
import os
import open3d as o3d 
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage

# --- ROS IMPORTS ---
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker 
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry
import tf2_ros
import tf2_geometry_msgs 
from geometry_msgs.msg import Twist, PointStamped, PoseStamped, TwistStamped

# --- CUSTOM IMPORTS ---
from RedNet_model import RedNet 
from utils import utils
from ultralytics import YOLO

# --- CONFIGURATION ---
os.environ['ULTRALYTICS_OFFLINE'] = 'True'
os.environ['YOLO_VERBOSE'] = 'False'

SYSTEM_LATENCY = 0.15 
CHECKPOINT_PATH = '/home/administrator/VLPathPlanning/src/depth_to_pointcloud/src/scripts/model_bestRN1.pth'
SCAN_ANGLE_STEP_DEG = 30.0  
FRAMES_TO_SCAN = 10         

# --- GLOBAL MODEL SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_depth = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# --- Compatibility: checkpoint saved with NumPy ≥2.0 on a NumPy 1.x system ---
import sys
import numpy
if not hasattr(numpy, '_core'):
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
#rospy.loginfo(f"Loading RedNet from {CHECKPOINT_PATH}...")
# --- Load Model (matching visual_inspect.py) ---
model = RedNet(num_classes=2, pretrained=False)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    state_dict = ckpt['state_dict']
else:
    state_dict = ckpt
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.to(device)
model.eval()

camera_intrinsicsrgb = np.array([
    [388.6187, 0, 320.6609],
    [0, 388.6187, 239.7847],
    [0, 0, 1]
])

class HuskyMissionNode:
    def __init__(self):
        rospy.init_node('husky_mission_control')
        
        self.node_start_time = rospy.Time.now()
        
        # State & Logic
        self.yolo_enabled = True
        self.sent_goal_xyz = None
        self.sent_goal_yaw = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.K = np.array([[388.6187, 0, 320.6609],[0, 388.6187, 239.7847],[0, 0, 1]])
        
        self.target_visible = False
        self.mission_complete = False
        self.state = "IDLE" 
        self.action_plan = []
        self.current_step_idx = 0
        self.current_target_id = None
        self.current_dist_thresh = 0.3
        self.is_rotation_move = False 
        
        self.detection_counter = 0 
        self.frames_scanned_in_step = 0 
        self.search_target_yaw = 0.0    
        
        self.last_track_time = 0.0

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.latest_rgb = None
        self.latest_depth = None

        self.bridge = CvBridge()
        self.yolo_model = YOLO('/home/administrator/aed_environment/yolov8n.pt')

        self.pc_pub = rospy.Publisher('/velodyne_points', PointCloud2, queue_size=1)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.wpt_pub = rospy.Publisher('/wpts', PointStamped, queue_size=5, latch=True)
        self.marker_pub = rospy.Publisher('/mission_marker', Marker, queue_size=5)

        depth_sub = Subscriber('/camera/depth/image_rect_raw', Image)
        rgb_sub = Subscriber('/camera/color/image_raw', Image)
        self.sub_odom = rospy.Subscriber('/state_estimation', Odometry, self.odom_callback)

        ats = ApproximateTimeSynchronizer([depth_sub, rgb_sub], queue_size=5, slop=0.1)
        ats.registerCallback(self.camera_callback)

        self.yolo_thread = threading.Thread(target=self.yolo_loop)
        self.yolo_thread.daemon = True
        self.yolo_thread.start()
        
        rospy.sleep(1.0)
        self.load_mission_plan()

    def load_mission_plan(self):
        plan_path = "/home/administrator/aed_environment/src/depth_to_pointcloud/src/scripts/mission_plan.json"
        try:
            with open(plan_path, "r") as f:
                self.action_plan = json.load(f)
            rospy.loginfo(f"📂 Loaded {len(self.action_plan)} steps.")
            self.current_step_idx = 0
            self.process_next_action()
        except Exception as e:
            rospy.logerr(f"❌ Failed to load mission plan: {e}")
            self.mission_complete = True

    def publish_visual_marker(self, x, y, yaw, frame_id="map"):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.id = 0; m.type = Marker.ARROW; m.action = Marker.ADD
        m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = 0.5
        cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
        m.pose.orientation.w = cy; m.pose.orientation.z = sy
        m.scale.x = 0.8; m.scale.y = 0.2; m.scale.z = 0.2
        m.color.a = 1.0; m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
        self.marker_pub.publish(m)

    def process_next_action(self):
        if self.current_step_idx >= len(self.action_plan):
            rospy.loginfo("🏁 Mission Execution Finished.")
            self.mission_complete = True
            self.state = "IDLE" # Prevent loops
            return

        step = self.action_plan[self.current_step_idx]
        rospy.loginfo(f"▶️ Starting Step {self.current_step_idx}: {step}")
        self.detection_counter = 0

        if step["action"] == "MOVE":
            forward_dist = step.get("forward_dist", 0.0) 
            yaw_delta = step.get("yaw_delta", 0.0)

            target_yaw = self.robot_yaw + yaw_delta
            gx = self.robot_x + (forward_dist * math.cos(target_yaw))
            gy = self.robot_y + (forward_dist * math.sin(target_yaw))

            pt_msg = PointStamped()
            pt_msg.header.stamp = rospy.Time(0)
            pt_msg.header.frame_id = "map"
            pt_msg.point.x = gx; pt_msg.point.y = gy; pt_msg.point.z = 0.0
            self.wpt_pub.publish(pt_msg)
            
            self.sent_goal_xyz = (gx, gy, 0.0)
            self.sent_goal_yaw = target_yaw
            self.target_visible = False # Blind move
            
            if abs(forward_dist) < 0.1:
                self.current_dist_thresh = 0.05
                self.is_rotation_move = True
                rospy.loginfo(f"🔄 ROTATING {yaw_delta:.2f} rad")
            else:
                self.current_dist_thresh = 0.5 
                self.is_rotation_move = False
                rospy.loginfo(f"🚗 MOVING {forward_dist}m to ({gx:.2f}, {gy:.2f})")
                
            self.state = "WAIT_FOR_REACH"

        elif step["action"] == "FIND":
            self.current_target_id = step["id"]
            self.current_dist_thresh = step.get("thresh", 2.0)
            self.target_visible = False
            self.yolo_enabled = True
            self.sent_goal_xyz = None 
            self.is_rotation_move = False
            
            # Start in SCAN mode
            self.state = "SEARCH_SCAN"
            self.frames_scanned_in_step = 0
            rospy.loginfo(f"👀 Scanning for ID {self.current_target_id}...")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        if self.state != "WAIT_FOR_REACH": return
        
        # --- FIX: IGNORE ODOM FOR VISUAL TRACKING ---
        if self.target_visible:
            return # Let YOLO decide when we arrive based on visual depth
        
        if self.sent_goal_xyz is None: return

        gx, gy, _ = self.sent_goal_xyz
        dist_error = np.hypot(self.robot_x - gx, self.robot_y - gy)
        
        reached = False
        if self.is_rotation_move:
             yaw_error = abs(self.sent_goal_yaw - self.robot_yaw)
             if yaw_error > math.pi: yaw_error = 2*math.pi - yaw_error
             if yaw_error < 0.2: reached = True 
        else:
            active_thresh = max(0.5, self.current_dist_thresh)
            if dist_error < active_thresh: reached = True

        if reached:
            rospy.loginfo(f"✅ Step Reached (Odom). Moving to next...")
            self.sent_goal_xyz = None 
            self.current_step_idx += 1
            self.process_next_action()

    def run(self):
        rate = rospy.Rate(60)
        log_printed = False
        last_goal_pub_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            if self.mission_complete:
                if not log_printed:
                    rospy.loginfo("🏆 Mission Accomplished! Idling...")
                    log_printed = True
                stop = TwistStamped()
                stop.header.stamp = rospy.Time.now(); stop.header.frame_id = "base_link"
                self.cmd_pub.publish(stop.twist)
                rate.sleep()
                continue

            # 1. SCANNING (STOPPED)
            if self.state == "SEARCH_SCAN":
                stop = Twist(); self.cmd_pub.publish(stop)

            # 2. ROTATING (BLIND MOVE)
            elif self.state == "SEARCH_MOVE":
                yaw_err = self.search_target_yaw - self.robot_yaw
                if yaw_err > math.pi: yaw_err -= 2*math.pi
                if yaw_err < -math.pi: yaw_err += 2*math.pi
                
                if abs(yaw_err) < 0.1: 
                    rospy.loginfo("🛑 30-Deg Step Complete. Scanning...")
                    self.state = "SEARCH_SCAN" 
                    self.frames_scanned_in_step = 0
                    self.cmd_pub.publish(Twist())
                else:
                    cmd = Twist()
                    cmd.angular.z = 0.5 * np.sign(yaw_err) 
                    self.cmd_pub.publish(cmd)
            
            rate.sleep()

    def transform_point(self, pt_sensor):
        # Convert Sensor Frame -> Map Frame
        try:
            trans = self.tf_buffer.lookup_transform("map", pt_sensor.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
            pt_map = tf2_geometry_msgs.do_transform_point(pt_sensor, trans)
            return pt_map
        except Exception as e:
            # rospy.logwarn(f"TF Error: {e}")
            return None

    def yolo_loop(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            try:
                if (rospy.Time.now() - self.node_start_time).to_sec() < 3.0:
                    rate.sleep(); continue
                
                # FIX: Check if we have a target ID before running
                if self.current_target_id is None:
                    rate.sleep(); continue

                if self.state not in ["SEARCH_SCAN", "WAIT_FOR_REACH"]:
                    rate.sleep(); continue
                
                # THROTTLE FOR TRACKING
                if self.state == "WAIT_FOR_REACH":
                    if (time.time() - self.last_track_time) < 0.5: 
                        rate.sleep(); continue
                    self.last_track_time = time.time()

                if self.latest_rgb is None or self.latest_depth is None:
                    rate.sleep(); continue

                if self.state == "SEARCH_SCAN":
                    self.frames_scanned_in_step += 1
                    if self.frames_scanned_in_step > FRAMES_TO_SCAN:
                        rospy.loginfo(f"🤷‍♂️ Nothing here. Rotating {SCAN_ANGLE_STEP_DEG} deg...")
                        step_rad = math.radians(SCAN_ANGLE_STEP_DEG)
                        self.search_target_yaw = self.robot_yaw + step_rad
                        self.state = "SEARCH_MOVE"
                        rate.sleep(); continue
                
                target_id = int(self.current_target_id)
                results = self.yolo_model(self.latest_rgb, classes=[target_id], conf=0.4, verbose=False)[0]
                
                if results.boxes is None or len(results.boxes) == 0:
                    self.detection_counter = 0
                    rate.sleep(); continue

                self.frames_scanned_in_step = 0 
                found_conf = float(results.boxes[0].conf[0])
                
                self.detection_counter += 1
                required_count = 3 if self.state == "SEARCH_SCAN" else 1
                if self.detection_counter < required_count: 
                    rate.sleep(); continue 

                # DEPTH ESTIMATION
                box = results.boxes[0].xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = box
                h_img, w_img = self.latest_depth.shape
                depth_copy = self.latest_depth.copy().astype(np.float32) * 0.001 
                x1 = np.clip(x1, 0, w_img); x2 = np.clip(x2, 0, w_img)
                y1 = np.clip(y1, 0, h_img); y2 = np.clip(y2, 0, h_img)
                depth_roi = depth_copy[y1:y2, x1:x2]
                valid_mask = np.isfinite(depth_roi) & (depth_roi > 0.5)
                points = depth_roi[valid_mask]

                if len(points) < 20: 
                    rate.sleep(); continue

                z_robust = np.median(points)
                if z_robust < 0.5: rate.sleep(); continue

                # --- CHECK ARRIVAL (VISUAL SERVOING) ---
                # This is the ONLY place that should trigger 'Step Reached' for FIND actions
                if z_robust < self.current_dist_thresh:
                    rospy.loginfo(f"✅ Visual Target Reached! (Dist: {z_robust:.2f}m)")
                    # Stop
                    self.cmd_pub.publish(Twist())
                    self.sent_goal_xyz = None
                    self.target_visible = False
                    self.current_step_idx += 1
                    self.process_next_action()
                    rate.sleep(); continue
                
                # --- CALCULATE RELATIVE GOAL ---
                scale_factor = (z_robust - self.current_dist_thresh) / z_robust
                cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
                x_opt = (cx - self.K[0, 2]) * z_robust / self.K[0, 0]
                y_opt = (cy - self.K[1, 2]) * z_robust / self.K[1, 1]
                
                x_cam = z_robust * scale_factor
                y_cam = -x_opt * scale_factor
                z_cam = -y_opt * scale_factor

                pt_cam = PointStamped()
                pt_cam.header.stamp = rospy.Time.now()
                pt_cam.header.frame_id = "sensor"
                pt_cam.point.x = float(x_cam)
                pt_cam.point.y = float(y_cam)
                pt_cam.point.z = float(z_cam)

                # TRANSFORM TO MAP (For Visualization & Planner Consistency)
                pt_map = self.transform_point(pt_cam)
                
                if pt_map:
                    if self.state == "SEARCH_SCAN":
                        rospy.loginfo(f"🎯 TARGET LOCKED! Starting Move. Dist: {z_robust:.2f}m")
                        self.state = "WAIT_FOR_REACH"
                        self.target_visible = True
                    elif self.state == "WAIT_FOR_REACH":
                         rospy.loginfo(f"🔄 TRACKING: Updating... Dist: {z_robust:.2f}m")

                    self.wpt_pub.publish(pt_map) 
                    self.publish_visual_marker(pt_map.point.x, pt_map.point.y, 0.0, "map")
                    self.sent_goal_xyz = (pt_map.point.x, pt_map.point.y, pt_map.point.z)
                else:
                    # Fallback if TF fails: Send relative goal (some planners accept this)
                    self.wpt_pub.publish(pt_cam)

                rate.sleep()
            except Exception as e:
                rospy.logerr(f"YOLO Error: {e}")
                rate.sleep()

    def camera_callback(self, d_msg, r_msg):
        if self.mission_complete: return
        self.latest_depth = self.bridge.imgmsg_to_cv2(d_msg, "passthrough")
        self.latest_rgb = self.bridge.imgmsg_to_cv2(r_msg, "rgb8")
        self.publish_point_cloud(self.latest_depth, d_msg.header, self.latest_rgb)

    def publish_point_cloud(self, depth, header, rgb):
        corrected_time = header.stamp - rospy.Duration(SYSTEM_LATENCY)
        
        h, w = depth.shape
        step = 20
        rows = np.arange(0, h, step); cols = np.arange(0, w, step)
        u, v = np.meshgrid(rows, cols, indexing='ij')
        
        depth = depth.astype(np.float32) * 0.001 
        z = depth[u, v]
        
        x_cam = (v - self.K[0, 2]) * z / self.K[0, 0]
        y_cam = (u - self.K[1, 2]) * z / self.K[1, 1]
        
        x_f = z
        y_f = -x_cam
        z_f = -y_cam
        
        mask = (z_f > -0.5) & (z_f < 0.1)
        x_f = x_f[mask]; y_f = y_f[mask]; z_f = z_f[mask]
        mask1 = (x_f < 10) & (x_f > 0.5)
        x_f = x_f[mask1]; y_f = y_f[mask1]; z_f = z_f[mask1]

        try:
            ppoints = redNet(self.latest_depth, rgb)
        except Exception as e:
            ppoints = None
        
        points = np.stack((x_f, y_f, z_f), axis=-1)

        if ppoints is not None and len(ppoints.shape) > 1 and ppoints.shape[0] > 0:
            points = np.vstack((points, ppoints))
        
        if points.shape[0] == 0: return

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud = cloud.voxel_down_sample(voxel_size=0.05)
        points = np.asarray(cloud.points)
        
        if points.shape[0] == 0: return

        dummy_intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points, dummy_intensity))

        out_header = std_msgs.msg.Header()
        out_header.stamp = corrected_time
        out_header.frame_id = 'sensor'
        
        fields = [PointField('x', 0, 7, 1), PointField('y', 4, 7, 1), PointField('z', 8, 7, 1), PointField('intensity', 12, 7, 1)]
        self.pc_pub.publish(pc2.create_cloud(out_header, fields, points))

# --- REDNET FUNCTION ---
def redNet(depth_image, color_image):
    if not rednet_loaded:
        return np.empty((0, 3), dtype=np.float32)  # Skip silently if model didn't load
    try:
        depth_image = depth_image.copy()
        color_image = color_image.copy()
        depth_m = depth_image.astype(np.float32) * 0.001
        
        masked_color = color_image.copy()
        masked_depth = depth_image.copy()
        masked_color[: masked_color.shape[0] // 2, :] = 0
        masked_depth[: masked_depth.shape[0] // 2, :] = 0

        color_image_pil = PILImage.fromarray(masked_color)

        if masked_depth.dtype != np.uint8:
            max_val = masked_depth.max() if masked_depth.max() > 0 else 1.0
            depth_8bit = cv2.convertScaleAbs(masked_depth, alpha=(255.0 / max_val))
        else:
            depth_8bit = masked_depth

        depth_image_pil = PILImage.fromarray(depth_8bit).convert("L")

        rgb_tensor = transform_rgb(color_image_pil).unsqueeze(0).to(device)
        depth_tensor = transform_depth(depth_image_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(rgb_tensor, depth_tensor)

        seg = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
        mask = (seg == 1)
        ys, xs = np.where(mask)
        depths = depth_m[ys, xs]

        valid = (depths > 0.3) & (depths < 10.0)
        xs, ys, depths = xs[valid], ys[valid], depths[valid]

        if depths.size == 0: return None

        fx, fy = camera_intrinsicsrgb[0, 0], camera_intrinsicsrgb[1, 1]
        cx, cy = camera_intrinsicsrgb[0, 2], camera_intrinsicsrgb[1, 2]

        X = (xs - cx) * depths / fx
        Y = (ys - cy) * depths / fy
        Z = depths
        y, z, x = -X, -Y, Z
        
        points = np.stack((x, y, z), axis=-1).astype(np.float32)
        points = points[np.isfinite(points).all(axis=1)]

        if points.shape[0] == 0: return np.empty((0, 3), dtype=np.float32)

        return points

    except Exception as e:
        rospy.logerr(f"Inference error: {e}")
        return np.empty((0, 3), dtype=np.float32)

if __name__ == '__main__':
    node = HuskyMissionNode()
    node.run()