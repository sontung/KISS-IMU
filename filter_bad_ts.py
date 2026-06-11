import numpy as np
import matplotlib.pyplot as plt


def isolate_active_window(dir_="/home/vr/work/kiss-icp/results/2026-06-08_14-51-30/first_run_poses.npy",
                          window_size=10,
                          threshold=0.1):
    # 1. Load poses and calculate raw frame-to-frame distances
    poses = np.load(dir_)
    translations = poses[:, :3, 3]
    distances = np.linalg.norm(np.diff(translations, axis=0), axis=1)  # Length: N-1

    # 2. Compute a moving average to smooth out setup/teardown jitter
    moving_mean = np.convolve(distances, np.ones(window_size) / window_size, mode='same')

    # 3. Find all indices where the smoothed movement is above the threshold
    active_indices = np.where(moving_mean > threshold)[0]

    if active_indices.size == 0:
        print("No active movement window found with the current threshold.")
        return None, None

    # 4. Extract the exact boundaries of the active window
    start_idx = active_indices[0]
    end_idx = active_indices[-1]

    print(f"Setup Phase Ends / Movement Starts at frame: {start_idx}")
    print(f"Movement Ends / Teardown Phase Begins at frame: {end_idx}")
    print(f"Total active frames: {end_idx - start_idx}")

    # --- Optional: Plotting the isolated window ---
    plt.figure(figsize=(11, 5))
    plt.plot(distances, color='gray', alpha=0.4, label='Raw Displacement')
    plt.plot(moving_mean, color='blue', lw=2, label='Smoothed Movement (Rolling Mean)')

    # Visualize the threshold and boundaries
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold}m)')
    plt.axvline(x=start_idx, color='green', linestyle=':', lw=2, label=f'Start Frame ({start_idx})')
    plt.axvline(x=end_idx, color='orange', linestyle=':', lw=2, label=f'End Frame ({end_idx})')

    # Highlight the isolated "actual movement" zone
    plt.axvspan(start_idx, end_idx, color='green', alpha=0.1, label='Active Movement Window')

    plt.title('Isolating Actual Movement Window (Removing Setup & Teardown)', fontsize=12)
    plt.xlabel('Frame Index', fontsize=10)
    plt.ylabel('Distance Moved per Frame (meters)', fontsize=10)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('isolated_movement_window.png', dpi=300)
    plt.show()

    # 5. Return the indices so you can crop your poses array if needed
    # cropped_poses = poses[start_idx:end_idx]
    return start_idx, end_idx


if __name__ == '__main__':
    # start, end = isolate_active_window()
    #1461 10127
    #571 8946
    # print(start, end)
    start, end = isolate_active_window("/home/vr/work/kiss-icp/results/2026-06-08_15-01-07/second_run_poses.npy")
    print(start, end)