# run_sim.py

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from localization_sim.config import DT, Q_diag, R_diag
from localization_sim.ekf import EKFLocalization
from localization_sim.simulate import simulate_truth_and_sensors
from localization_sim.utils_vis import plot_trajectory_with_ekf
from localization_sim.metrics import rmse_position, rmse_yaw, rmse_speed


def plot_error_time_series(t_arr, pos_err, yaw_err, v_err, output_dir=None):
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure()
    plt.grid(True)
    plt.plot(t_arr, pos_err)
    plt.xlabel('Time [s]')
    plt.ylabel('Position Error [m]')
    plt.title('Position Error Time Series')
    plt.tight_layout()
    output_path = output_dir / 'position_error.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved position error plot to {output_path}")
    plt.close()

    plt.figure()
    plt.grid(True)
    plt.plot(t_arr, yaw_err)
    plt.xlabel('Time [s]')
    plt.ylabel('Yaw Error [rad]')
    plt.title('Yaw Error Time Series')
    plt.tight_layout()
    output_path = output_dir / 'yaw_error.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved yaw error plot to {output_path}")
    plt.close()

    plt.figure()
    plt.grid(True)
    plt.plot(t_arr, v_err)
    plt.xlabel('Time [s]')
    plt.ylabel('Speed Error [m/s]')
    plt.title('Speed Error Time Series')
    plt.tight_layout()
    output_path = output_dir / 'speed_error.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved speed error plot to {output_path}")
    plt.close()

def main():
    # Simulate ground truth + sensors
    t_arr, x_true_hist, imu_meas_hist, gps_meas_hist = simulate_truth_and_sensors()

    # EKF instance
    ekf = EKFLocalization(Q_diag, R_diag)

    # Buffers
    N = len(t_arr)
    x_est_hist = np.zeros((N, 4))
    P_hist = np.zeros((N, 4, 4))

    # For aligning GPS by time
    gps_index = 0

    for k, t in enumerate(t_arr):
        u_meas = imu_meas_hist[k]

        # Prediction every time step using IMU
        ekf.predict(u_meas, DT)

        # If there is a GPS measurement at (approximately) this time, use correction
        while gps_index < len(gps_meas_hist) and abs(gps_meas_hist[gps_index][0] - t) < 0.5 * DT:
            _, z = gps_meas_hist[gps_index]
            ekf.update(z)
            gps_index += 1

        x_est_hist[k] = ekf.x
        P_hist[k] = ekf.P

    # --- RMSE Metrics --- 
    pos_rmse, pos_err = rmse_position(x_true_hist, x_est_hist)
    yaw_rmse, yaw_err = rmse_yaw(x_true_hist, x_est_hist)
    v_rmse, v_err = rmse_speed(x_true_hist, x_est_hist)

    print("=== EKF Metrics ===")
    print(f"Position RMSE  (||e_xy||): {pos_rmse:.3f} m")
    print(f"Yaw RMSE        (e_yaw): {yaw_rmse:.4f} rad ({yaw_rmse * 180 / np.pi:.2f} deg)")
    print(f"Speed RMSE      (e_v): {v_rmse:.3f} m/s")

    # steady-state RMSE over last 50% samples
    start = len(t_arr) // 2
    pos_rmse_ss = np.sqrt(np.mean(pos_err[start:] ** 2))
    yaw_rmse_ss = np.sqrt(np.mean(yaw_err[start:] ** 2))
    v_rmse_ss   = np.sqrt(np.mean(v_err[start:] ** 2))

    print("---- Steady-state (last 50%) ----")
    print(f"Position RMSE SS: {pos_rmse_ss:.3f} m")
    print(f"Yaw RMSE SS:      {yaw_rmse_ss:.4f} rad ({yaw_rmse_ss * 180.0/np.pi:.2f} deg)")
    print(f"Speed RMSE SS:    {v_rmse_ss:.3f} m/s")

    # Set output directory
    output_dir = Path(__file__).parent.parent.parent / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    plot_trajectory_with_ekf(t_arr, x_true_hist, x_est_hist, P_hist, gps_meas_hist, output_dir)

    plot_error_time_series(t_arr, pos_err, yaw_err, v_err, output_dir)

if __name__ == "__main__":
    main()
