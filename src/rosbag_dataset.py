import os
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pypose as pp
import natsort
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def collate_fn(batch):
    """
    Collates data packets of varying array dimensions into structural packed batches.
    Pads variable-length IMU readings to the maximum length present in this batch.
    """
    output_batch = {}

    # 1. Identify maximum IMU length padding target inside this specific batch segment
    # Force timestamps, features, and dts to align to the same maximal sequence limit
    max_imu_len = max([item["valid_length"].item() for item in batch])

    collated_imu_ts = []
    collated_accels = []
    collated_gyros = []
    collated_imu_dts = []  # Renamed to match model expectation

    # Track non-variable metadata keys separately
    # ADDED: 'lidar_ts0' and 'lidar_ts1' tracking keys here
    tensor_keys = [
        "gt_pose0",
        "gt_pose1",
        "gt_velocity",
        "valid_length",
        "scan0_ts",
        "scan1_ts",
    ]
    for k in tensor_keys:
        output_batch[k] = []

    # We keep scan lists uncollated in raw python groups since point counts vary across locations
    output_batch["scan0"] = [item["scan0"] for item in batch]
    output_batch["scan1"] = [item["scan1"] for item in batch]

    for item in batch:
        curr_imu_len = item["valid_length"].item()
        pad_imu_len = max_imu_len - curr_imu_len

        # Calculate dt pad requirement using the same universal max_imu_len anchor
        pad_dts_len = max_imu_len - len(item["imu_dts"])

        # Pad IMU Timestamps [N] -> [Max_N]
        padded_ts = torch.nn.functional.pad(
            item["imu_ts"], (0, pad_imu_len), mode="constant", value=0.0
        )
        collated_imu_ts.append(padded_ts)

        # Pad Accelerations [N, 3] -> [Max_N, 3]
        padded_acc = torch.nn.functional.pad(
            item["accels"], (0, 0, 0, pad_imu_len), mode="constant", value=0.0
        )
        collated_accels.append(padded_acc)

        # Pad Gyroscopes [N, 3] -> [Max_N, 3]
        padded_gyr = torch.nn.functional.pad(
            item["gyros"], (0, 0, 0, pad_imu_len), mode="constant", value=0.0
        )
        collated_gyros.append(padded_gyr)

        # Pad Incremental Delta-T spans [N] -> [Max_N]
        # Crucial fix: Default padding value changed to 0.0 or typical dt offset (e.g., 0.01 for 100Hz)
        padded_dts = torch.nn.functional.pad(
            item["imu_dts"], (0, pad_dts_len), mode="constant", value=0.01
        )
        collated_imu_dts.append(padded_dts)

        for k in tensor_keys:
            output_batch[k].append(item[k])

    # Stack sequential vectors into final batched dimension layers
    output_batch["imu_ts"] = torch.stack(collated_imu_ts, dim=0)
    output_batch["accels"] = torch.stack(collated_accels, dim=0)
    output_batch["gyros"] = torch.stack(collated_gyros, dim=0)

    # CRITICAL FIX: Named key transformed from 'dts' to 'imu_dts' to satisfy your forward method
    output_batch["imu_dts"] = torch.stack(collated_imu_dts, dim=0)

    output_batch["gt_pose0"] = torch.stack(output_batch["gt_pose0"], dim=0)
    output_batch["gt_pose1"] = torch.stack(output_batch["gt_pose1"], dim=0)
    output_batch["gt_velocity"] = torch.stack(output_batch["gt_velocity"], dim=0)
    output_batch["valid_length"] = torch.stack(output_batch["valid_length"], dim=0)

    # ADDED: Stack scalar LiDAR timestamp tensors over batch dimensions
    output_batch["scan0_ts"] = torch.stack(output_batch["scan0_ts"], dim=0)
    output_batch["scan1_ts"] = torch.stack(output_batch["scan1_ts"], dim=0)

    return output_batch


def matrix_4x4_to_vector_7d(T):
    """
    Converts a 4x4 homogeneous transformation matrix into a 7D vector.

    Format of output: [tx, ty, tz, qx, qy, qz, qw]
    - Translation: tx, ty, tz
    - Quaternion (SciPy / ROS convention): qx, qy, qz, qw
    """
    # Ensure input is a numpy array
    T = np.array(T)

    # 1. Extract the 3D translation vector (top right 3x1)
    translation = T[0:3, 3]

    # 2. Extract the 3D rotation matrix (top left 3x3)
    rotation_matrix = T[0:3, 0:3]

    # 3. Convert rotation matrix to quaternion
    # SciPy natively outputs in (x, y, z, w) format
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    # 4. Concatenate translation and quaternion into a 7D vector
    vector_7d = np.concatenate([translation, quaternion])

    return torch.from_numpy(vector_7d)


class RosbagSeqDataset(Dataset):
    def __init__(self, data_root, data_seq, start, end,
                 data_dir: Path, gt_pose_path=None,
                 topic=None, data_type="diter_os", window_size=5):
        """
        Args:
            data_root (str): Path to the dataset root folder.
            data_seq (str): The specific bag file or directory containing splits (e.g., 'sequence_01.bag').
            topic (str): The ROS topic name for the PointCloud2 messages.
            data_type (str): 'diter_os' or 'kitti' mapping conventions.
            window_size (int): Number of consecutive lidar frames packaged per training step.
        """
        try:
            from rosbags.highlevel import AnyReader
        except ModuleNotFoundError:
            print('rosbags library not installed, run "pip install -U rosbags"')
            sys.exit(1)

        from kiss_icp.tools.point_cloud2 import read_point_cloud
        self.read_point_cloud = read_point_cloud

        self.data_root = data_root
        self.data_seq = data_seq
        self.seq_path = Path(os.path.join(data_root, data_seq))
        self.data_type = data_type
        self.window_size = window_size

        self.T_I_G = np.zeros(3)
        self.R_I_L = np.identity(3)

        # =========================================================================
        # 1. INTEGRATED: Rosbag Initialization and Metadata Index Caching
        # =========================================================================
        if data_dir.is_file():
            self.sequence_id = os.path.basename(data_dir).split(".")[0]
            self.bag = AnyReader([data_dir])
        else:
            bagfiles = [Path(path) for path in glob.glob(os.path.join(data_dir, "*.bag"))]
            if len(bagfiles) > 0:
                self.sequence_id = os.path.basename(bagfiles[0]).split(".")[0]
                self.bag = AnyReader(bagfiles)
            else:
                self.sequence_id = os.path.basename(data_dir).split(".")[0]
                self.bag = AnyReader([data_dir])

        if len(self.bag.paths) > 1:
            print("Reading multiple .bag files in directory:")
            print("\n".join(natsort.natsorted([path.name for path in self.bag.paths])))

        self.bag.open()
        self.topic = self.check_topic(topic)
        self.n_scans = self.bag.topics[self.topic].msgcount
        self.data_dir = data_dir

        # Build connections mapping
        self.connections = [x for x in self.bag.connections if x.topic == self.topic]

        # CRITICAL FOR RANDOM ACCESS: Cache all message reference points & timestamps up front
        print(f"Indexing rosbag for topic {self.topic}... This may take a moment.")
        self.msg_references = []
        self.lidar_timestamps = []
        self.start = start
        self.end = end

        # Scan through the message framework to build an absolute index index map
        for connection, timestamp, rawdata in self.bag.messages(connections=self.connections):
            self.msg_references.append((connection, rawdata))
            self.lidar_timestamps.append(self.to_sec(timestamp))

        self.lidar_timestamps = np.array(self.lidar_timestamps, dtype=np.float64)[start:end]
        print(f"Successfully indexed {len(self.msg_references)} point cloud frames.")

        # =========================================================================
        # 2. RETAINED: Monotonic IMU Streams & Auxiliary Anchors
        # =========================================================================
        # Assumes imu.csv and gt_pose.csv are located next to the bag files in your sequence folder
        imu_file = os.path.join(self.seq_path.parent if self.seq_path.is_file() else self.seq_path, "imu.csv")
        self.imu_data = pd.read_csv(imu_file, header=None).values

        self.imu_ts = self.imu_data[:, 0]
        self.accels = self.imu_data[:, 1:4]
        self.gyros = self.imu_data[:, 4:7]
        self.gravity = torch.tensor([0.0, 0.0, -9.80665])

        pose_file = os.path.join(self.seq_path.parent if self.seq_path.is_file() else self.seq_path, "gt_pose.csv")
        self.gt_poses = pd.read_csv(pose_file, header=None).values
        if gt_pose_path is not None:
            self.gt_poses = torch.from_numpy(np.load(gt_pose_path))[start:end]

        self.init = {
            "pos": torch.zeros(3),
            "rot": torch.tensor([0, 0, 0, 1]),
            "vel": torch.zeros(3),
        }

        timestamps = self.imu_data[:, 0]
        dts = np.diff(timestamps)
        avg_dt = np.mean(dts) if len(dts) > 0 else 0.01
        self.imu_dts = np.insert(dts, 0, avg_dt)

    def __del__(self):
        if hasattr(self, "bag"):
            self.bag.close()

    @staticmethod
    def to_sec(nsec: int):
        return float(nsec) / 1e9

    def load_scan_from_bag(self, msg_idx):
        """Deserializes a rosbag frame reference index directly into a PyTorch point cloud."""
        connection, rawdata = self.msg_references[msg_idx]
        msg = self.bag.deserialize(rawdata, connection.msgtype)

        # Extract points using the kiss_icp helper utility function layout
        # Returns a numpy array or structured format depending on your package backend
        points_np, ts = self.read_point_cloud(msg)

        return torch.from_numpy(np.hstack([points_np, ts[:, None]]))

    def __len__(self):
        return len(self.lidar_timestamps)-1

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = idx + 1

        # Retrieve absolute timestamps out of our cached rosbag timing arrays
        t_start = self.lidar_timestamps[start_idx]
        t_end = self.lidar_timestamps[end_idx]

        # Extract slices of IMU data landing natively between t_start and t_end
        imu_mask = (self.imu_data[:, 0] >= t_start) & (self.imu_data[:, 0] <= t_end)
        imu_indices = np.arange(self.imu_data.shape[0])[imu_mask]
        imu_start_idx = imu_indices[0]
        imu_end_idx = imu_indices[-1]
        assert imu_start_idx < imu_end_idx
        window_imu = self.imu_data[imu_mask]

        if len(window_imu) == 0:
            window_imu = np.zeros((10, 7))
            window_imu[:, 0] = np.linspace(t_start, t_end, 10)
            tqdm.write(f"NO imu data found in {t_start} to {t_end}")
            return {}

        imu_ts = torch.from_numpy(window_imu[:, 0].astype(np.float32))
        accels = torch.from_numpy(window_imu[:, 1:4].astype(np.float32))
        gyros = torch.from_numpy(window_imu[:, 4:7].astype(np.float32))

        dt_segment = self.imu_dts[imu_start_idx:imu_end_idx]

        gt_pose0 = self.gt_poses[start_idx]
        gt_pose1 = self.gt_poses[end_idx]

        dt_window = t_end - t_start
        dp_window = gt_pose1[:3, 3]-gt_pose0[:3, 3]
        gt_velocity = (dp_window / (dt_window + 1e-8))
        gt_pose0 = matrix_4x4_to_vector_7d(gt_pose0)
        gt_pose1 = matrix_4x4_to_vector_7d(gt_pose1)

        # INTEGRATED LOADING: Fetch point cloud matrices directly via index signatures
        scan0 = self.load_scan_from_bag(start_idx)
        scan1 = self.load_scan_from_bag(end_idx)

        return {
            "scan0": scan0,
            "scan1": scan1,
            "imu_ts": imu_ts,
            "accels": accels,
            "gyros": gyros,
            # "dts": torch.tensor(dt_segment, dtype=torch.float32),
            # "dts": t_end-t_start,
            "valid_length": torch.tensor(len(imu_ts), dtype=torch.long),
            "gt_pose0": gt_pose0,
            "gt_pose1": gt_pose1,
            "gt_velocity": gt_velocity,
            "imu_dts": torch.tensor(dt_segment, dtype=torch.float32),
            "scan0_ts": torch.tensor(t_start, dtype=torch.float32),
            "scan1_ts": torch.tensor(t_end, dtype=torch.float32),
        }

    def check_topic(self, topic: str) -> str:
        point_cloud_topics = [
            t[0] for t in self.bag.topics.items() if t[1].msgtype == "sensor_msgs/msg/PointCloud2"
        ]

        def print_available_topics_and_exit():
            print(50 * "-")
            for t in point_cloud_topics:
                print(f"--topic {t}")
            print(50 * "-")
            sys.exit(1)

        if topic and topic in point_cloud_topics:
            return topic
        if topic and topic not in point_cloud_topics:
            print(f'[ERROR] Dataset does not contain msg topic "{topic}".')
            print_available_topics_and_exit()
        if len(point_cloud_topics) > 1:
            print("Multiple sensor_msgs/msg/PointCloud2 topics available.")
            print_available_topics_and_exit()
        if len(point_cloud_topics) == 0:
            print("[ERROR] Your dataset does not contain any PointCloud2 topics")
            sys.exit(1)
        return point_cloud_topics[0]