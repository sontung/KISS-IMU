import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pypose as pp


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
        pad_dts_len = max_imu_len - len(item["dts"])

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
            item["dts"], (0, pad_dts_len), mode="constant", value=0.01
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


class SeqDataset(Dataset):
    def __init__(self, data_root, data_seq, data_type="diter_os", window_size=5):
        """
        Args:
            data_root (str): Path to the dataset root folder (e.g., '/storage1/For_IMUNet/DiTer_os')
            data_seq (str): The specific sequence folder (e.g., 'Forest_new')
            data_type (str): 'diter_os' or 'kitti' mapping point structures and index mappings.
            window_size (int): Number of consecutive lidar frames packaged per training step.
        """
        self.data_root = data_root
        self.data_seq = data_seq
        self.seq_path = os.path.join(data_root, data_seq)
        self.data_type = data_type
        self.window_size = window_size

        self.T_I_G = np.zeros(3)
        self.R_I_L = np.identity(3)

        # 1. Resolve Point Cloud files and Timestamps
        self.bin_files = sorted(
            glob.glob(os.path.join(self.seq_path, "points", "data", "*.bin"))
        )
        ts_file = os.path.join(self.seq_path, "points", "timestamps.txt")
        self.lidar_timestamps = np.loadtxt(ts_file, dtype=np.float64)

        # Make sure file-counts match timestamp lines
        min_len = min(len(self.bin_files), len(self.lidar_timestamps))
        self.bin_files = self.bin_files[:min_len]
        self.lidar_timestamps = self.lidar_timestamps[:min_len]

        # 2. Read Monotonic IMU Streams
        # Column Map: [0: time, 1: ax, 2: ay, 3: az, 4: gx, 5: gy, 6: gz]
        imu_file = os.path.join(self.seq_path, "imu.csv")
        self.imu_data = pd.read_csv(imu_file, header=None).values

        # 1. Separate the structural column components
        self.imu_ts = self.imu_data[:, 0]  # First column (N,)
        self.accels = self.imu_data[:, 1:4]  # Next 3 columns (N, 3)
        self.gyros = self.imu_data[:, 4:7]  # Last 3 columns (N, 3)
        self.gravity = None

        # 3. Read Ground Truth Trajectory Poses
        pose_file = os.path.join(self.seq_path, "gt_pose.csv")
        self.gt_poses = pd.read_csv(
            pose_file, header=None
        ).values  # Shape: (Total_Frames, 7)

        # Set base initial state used by the training logic initialization macros:
        # e.g., init_state = dataset.init
        self.init = {
            "pos": torch.zeros(3),
            "rot": torch.tensor([0, 0, 0, 1]),
            "vel": torch.zeros(3),  # Can be initialized to zero or estimated
        }

        timestamps = self.imu_data[:, 0]

        # Calculate time difference between consecutive samples
        # Pad the first element with the average step so it maintains length N
        dts = np.diff(timestamps)
        avg_dt = np.mean(dts) if len(dts) > 0 else 0.01  # default to 100Hz if empty
        self.imu_dts = np.insert(dts, 0, avg_dt)

    def load_scan(self, path):
        """Loads and formats a packed little-endian binary scan based on dataset properties."""
        if self.data_type == "diter_os":
            lidar_dtype = np.dtype(
                [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("intensity", "<f8")]
            )
            scan = np.fromfile(path, dtype=lidar_dtype)
            points = np.stack([scan["x"], scan["y"], scan["z"]], axis=-1).astype(
                np.float32
            )
        else:  # Fallback Default 'kitti' (float32 flat array layout)
            scan = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            points = scan[:, :3]
        return torch.from_numpy(points)

    def __len__(self):
        # Calculate maximum possible sequence windows we can slip across the time framework
        if len(self.bin_files) < self.window_size:
            return 0
        return len(self.bin_files) - self.window_size

    def __getitem__(self, idx):
        # Window bounds
        start_idx = idx
        end_idx = idx + self.window_size

        # Lidar Frame Temporal Anchors
        t_start = self.lidar_timestamps[start_idx]
        t_end = self.lidar_timestamps[end_idx]

        # Extract slices of IMU data landing natively between t_start and t_end boundary indices
        imu_mask = (self.imu_data[:, 0] >= t_start) & (self.imu_data[:, 0] <= t_end)
        window_imu = self.imu_data[imu_mask]

        # Safe fallback constraint if no raw imu metrics match the interval
        if len(window_imu) == 0:
            window_imu = np.zeros((10, 7))
            window_imu[:, 0] = np.linspace(t_start, t_end, 10)

        # Segment out times, raw accelerations, and rotational velocities
        imu_ts = torch.from_numpy(window_imu[:, 0].astype(np.float32))
        accels = torch.from_numpy(window_imu[:, 1:4].astype(np.float32))
        gyros = torch.from_numpy(window_imu[:, 4:7].astype(np.float32))

        # Dts computation within tracking arrays
        dts = imu_ts[1:] - imu_ts[:-1]
        dt_segment = self.imu_dts[start_idx:end_idx]  # Shape: (valid_len,)
        if len(dts) == 0:
            dts = torch.tensor([0.01], dtype=torch.float32)

        # Ground Truth Window states [x, y, z, qx, qy, qz, qw]
        gt_pose0 = torch.from_numpy(self.gt_poses[start_idx].astype(np.float32))
        gt_pose1 = torch.from_numpy(self.gt_poses[start_idx:end_idx].astype(np.float32))

        # Approximate raw linear velocity boundary vector assuming translation diff over dt
        dt_window = t_end - t_start
        dp_window = self.gt_poses[end_idx, :3] - self.gt_poses[start_idx, :3]
        gt_velocity = torch.from_numpy(
            (dp_window / (dt_window + 1e-8)).astype(np.float32)
        )

        # Read LiDAR boundaries requested directly by lo_model optimization step
        scan0 = self.load_scan(self.bin_files[start_idx])
        scan1 = self.load_scan(self.bin_files[end_idx - 1])

        return {
            "scan0": scan0,
            "scan1": scan1,
            "imu_ts": imu_ts,
            "accels": accels,
            "gyros": gyros,
            "dts": dts,
            "valid_length": torch.tensor(len(imu_ts), dtype=torch.long),
            "gt_pose0": gt_pose0,
            "gt_pose1": gt_pose1,
            "gt_velocity": gt_velocity,
            "imu_dts": torch.tensor(dt_segment, dtype=torch.float32),
            # ADDED: Package the start and end LiDAR anchor timestamps as scalar float32 tensors
            "scan0_ts": torch.tensor(t_start, dtype=torch.float32),
            "scan1_ts": torch.tensor(
                self.lidar_timestamps[end_idx - 1], dtype=torch.float32
            ),
        }
