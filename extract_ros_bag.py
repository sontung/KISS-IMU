import os
import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# ==========================================
# CONFIGURATION: Change these to match your bag
# ==========================================
BAG_DIR = "/home/vr/work/datasets/VINUNI/second_run"  # Path to the folder containing metadata.yaml
OUTPUT_ROOT = "/home/vr/work/datasets/VINUNI/second_run/MyCustomData"
SEQUENCE_NAME = "Bag_Sequence_01"

IMU_TOPIC = "/livox/imu"
LIDAR_TOPIC = "/livox/lidar"
POSE_TOPIC = None # Optional: Set to None if you don't have GT poses

# Create directories
seq_root = os.path.join(OUTPUT_ROOT, SEQUENCE_NAME)
points_dir = os.path.join(seq_root, "points", "data")
os.makedirs(points_dir, exist_ok=True)

print(f"[*] Reading bag from: {BAG_DIR}")
print(f"[*] Target destination: {seq_root}")

imu_records = []
lidar_timestamps = []
gt_pose_records = []
scan_idx = 0

# Target lidar dtype matching 'diter_os' requirements
lidar_dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8'), ('intensity', '<f8')])

with Reader(BAG_DIR) as reader:
    # Check what connections/topics actually exist in this bag
    connections = [c for c in reader.connections]
    for connection, timestamp, rawdata in reader.messages():
        topic = connection.topic

        # 1. PARSE IMU DATA
        if topic == IMU_TOPIC:
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # Combine header seconds + nanoseconds for monotonic float64 time
            msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

            imu_records.append([
                msg_time,
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

        # 2. PARSE LIDAR POINT CLOUDS (sensor_msgs/msg/PointCloud2)
        elif topic == LIDAR_TOPIC:

            # self.timestamps.append(float(timestamp)/1e9)
            # msg = self.bag.deserialize(rawdata, connection.msgtype)

            msg = deserialize_cdr(rawdata, connection.msgtype)
            msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            lidar_timestamps.append(float(timestamp)/1e9)

            # Extract raw byte buffer from the ROS PointCloud2 message
            # Assumes standard x, y, z float layout (adjust fields if yours has different packing)
            # For most systems, structured unpacking via numpy frombuffer works fast:
            fmt_type = np.float32 if msg.fields[0].datatype == 7 else np.float64

            # Read out step lengths
            point_step = msg.point_step
            num_points = msg.width * msg.height

            # Simple conversion trick assuming standard xyz field offsets
            raw_array = np.frombuffer(msg.data, dtype=np.uint8).reshape(num_points, point_step)

            # Extract floats dynamically
            x_vals = raw_array[:, 0:4].view(np.float32).astype(np.float64)
            y_vals = raw_array[:, 4:8].view(np.float32).astype(np.float64)
            z_vals = raw_array[:, 8:12].view(np.float32).astype(np.float64)

            # Default intensity to 1.0 if not available or unaligned
            intensity = np.ones_like(x_vals)
            if len(msg.fields) > 3:
                try:
                    intensity = raw_array[:, 12:16].view(np.float32).astype(np.float64)
                except:
                    pass

            # Build structured binary payload matching layout requirements
            structured_scan = np.zeros(num_points, dtype=lidar_dtype)
            structured_scan['x'] = x_vals.flatten()
            structured_scan['y'] = y_vals.flatten()
            structured_scan['z'] = z_vals.flatten()
            structured_scan['intensity'] = intensity.flatten()

            # Save out to structured bin layout
            bin_filename = os.path.join(points_dir, f"{scan_idx:06d}.bin")
            structured_scan.tofile(bin_filename)
            scan_idx += 1

        # 3. PARSE GROUND TRUTH TRAJECTORY POSES (geometry_msgs/msg/PoseStamped or Odometry)
        elif POSE_TOPIC and topic == POSE_TOPIC:
            msg = deserialize_cdr(rawdata, connection.msgtype)
            # Extract pose object regardless of whether it's an Odometry or PoseStamped message wrapper
            pose_obj = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose

            # Save 7-column layout format [x, y, z, qx, qy, qz, qw]
            gt_pose_records.append([
                pose_obj.position.x,
                pose_obj.position.y,
                pose_obj.position.z,
                pose_obj.orientation.x,
                pose_obj.orientation.y,
                pose_obj.orientation.z,
                pose_obj.orientation.w
            ])

# ==========================================
# EXPORT PROCESSORS
# ==========================================
print("[*] Writing indexing descriptors to disk...")

# Write imu.csv
if imu_records:
    imu_np = np.array(imu_records)
    # Sort by timestamps to guarantee monotonic layout
    imu_np = imu_np[np.argsort(imu_np[:, 0])]
    np.savetxt(os.path.join(seq_root, "imu.csv"), imu_np, delimiter=",", fmt="%.9f")
    print(f"[✓] Extracted {len(imu_np)} IMU entries.")

# Write timestamps.txt
if lidar_timestamps:
    np.savetxt(os.path.join(seq_root, "points", "timestamps.txt"), sorted(lidar_timestamps), fmt="%.9f")
    print(f"[✓] Extracted {len(lidar_timestamps)} Lidar index anchors.")

# Write gt_pose.csv
if gt_pose_records:
    np.savetxt(os.path.join(seq_root, "gt_pose.csv"), np.array(gt_pose_records), delimiter=",", fmt="%.9f")
    print(f"[✓] Extracted {len(gt_pose_records)} Pose anchors.")
else:
    # If no ground truth exists, generate a dummy zero-trajectory matching Lidar array dimensions
    print("[!] No Pose topic found, writing stationary dummy sequence framework.")
    dummy_poses = np.zeros((max(1, scan_idx), 7))
    dummy_poses[:, 6] = 1.0  # Unit Quaternion identity
    np.savetxt(os.path.join(seq_root, "gt_pose.csv"), dummy_poses, delimiter=",", fmt="%.9f")

print("\n[COMPLETE] Your rosbag data has been reformatted to the target structural layout.")