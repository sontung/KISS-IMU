import os
import numpy as np
import pandas as pd
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from kiss_icp.tools.point_cloud2 import read_point_cloud


# ==========================================
# CONFIGURATION: Change these to match your bag
# ==========================================
BAG_DIR = "/home/vr/work/datasets/VINUNI/first_run"  # Path to the folder containing metadata.yaml
OUTPUT_ROOT = "/home/vr/work/datasets/VINUNI/first_run/MyCustomData"

# BAG_DIR = "/home/vr/work/datasets/VINUNI/second_run"  # Path to the folder containing metadata.yaml
# OUTPUT_ROOT = "/home/vr/work/datasets/VINUNI/second_run/MyCustomData"

SEQUENCE_NAME = ""

IMU_TOPIC = "/livox/imu"
LIDAR_TOPIC = "/livox/lidar"
POSE_TOPIC = None  # Optional: Set to None if you don't have GT poses

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

# TARGET LIDAR DTYPE: Updated to include the timestamp field (<f8 = float64)
lidar_dtype = np.dtype([
    ('x', '<f8'),
    ('y', '<f8'),
    ('z', '<f8'),
    ('intensity', '<f8'),
    ('timestamp', '<f8')
])

with Reader(BAG_DIR) as reader:
    connections = [c for c in reader.connections]
    for connection, timestamp, rawdata in reader.messages():
        topic = connection.topic

        # 1. PARSE IMU DATA
        if topic == IMU_TOPIC:
            msg = deserialize_cdr(rawdata, connection.msgtype)
            msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)

            imu_records.append([
                msg_time,
                msg.linear_acceleration.x*9.81,
                msg.linear_acceleration.y*9.81,
                msg.linear_acceleration.z*9.81,
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])

        # 2. PARSE LIDAR POINT CLOUDS (sensor_msgs/msg/PointCloud2)
        elif topic == LIDAR_TOPIC:
            msg = deserialize_cdr(rawdata, connection.msgtype)
            msg_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            lidar_timestamps.append(float(timestamp) / 1e9)

            # Extract raw byte buffer from the ROS PointCloud2 message
            fmt_type = np.float32 if msg.fields[0].datatype == 7 else np.float64

            # msg = self.bag.deserialize(rawdata, connection.msgtype)
            points_np, point_ts = read_point_cloud(msg)

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

            # Broadcast the frame-level msg_time to every point in this scan
            structured_scan['timestamp'] = point_ts

            # Save out to structured bin layout
            bin_filename = os.path.join(points_dir, f"{scan_idx:06d}.bin")
            structured_scan.tofile(bin_filename)
            scan_idx += 1

        # 3. PARSE GROUND TRUTH TRAJECTORY POSES
        elif POSE_TOPIC and topic == POSE_TOPIC:
            msg = deserialize_cdr(rawdata, connection.msgtype)

            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                pose_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            else:
                pose_time = float(timestamp) / 1e9

            pose_obj = msg.pose.pose if hasattr(msg.pose, 'pose') else msg.pose

            gt_pose_records.append([
                pose_time,
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
    imu_np = imu_np[np.argsort(imu_np[:, 0])]
    np.savetxt(os.path.join(seq_root, "imu.csv"), imu_np, delimiter=",", fmt="%.9f")
    print(f"[✓] Extracted {len(imu_np)} IMU entries.")

# Write timestamps.txt and lidar_timestamps.csv
if lidar_timestamps:
    sorted_timestamps = sorted(lidar_timestamps)
    np.savetxt(os.path.join(seq_root, "points", "timestamps.txt"), sorted_timestamps, fmt="%.9f")

    lidar_df = pd.DataFrame({
        'frame_idx': [f"{i:06d}" for i in range(len(sorted_timestamps))],
        'timestamp': sorted_timestamps
    })
    lidar_df.to_csv(os.path.join(seq_root, "lidar_timestamps.csv"), index=False)
    print(f"[✓] Extracted {len(lidar_timestamps)} Lidar index anchors.")

# Write gt_pose.csv
if gt_pose_records:
    gt_pose_np = np.array(gt_pose_records)
    gt_pose_np = gt_pose_np[np.argsort(gt_pose_np[:, 0])]
    np.savetxt(os.path.join(seq_root, "gt_pose.csv"), gt_pose_np, delimiter=",", fmt="%.9f")
    print(f"[✓] Extracted {len(gt_pose_np)} Pose anchors with timestamps.")
else:
    print("[!] No Pose topic found, writing stationary dummy sequence framework.")
    dummy_poses = np.zeros((max(1, scan_idx), 8))

    if lidar_timestamps:
        dummy_poses[:, 0] = sorted(lidar_timestamps)[:max(1, scan_idx)]

    dummy_poses[:, 7] = 1.0
    np.savetxt(os.path.join(seq_root, "gt_pose.csv"), dummy_poses, delimiter=",", fmt="%.9f")

print("\n[COMPLETE] Your rosbag data has been reformatted to the target structural layout.")