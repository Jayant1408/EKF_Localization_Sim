# run_sim.py

import numpy as np

from config import DT, Q_diag, R_diag
from ekf import EKFLocalization
from simulate import simulate_truth_and_sensors
from utils_vis import plot_trajectory_with_ekf


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

    # Plot results
    plot_trajectory_with_ekf(t_arr, x_true_hist, x_est_hist, P_hist, gps_meas_hist)


if __name__ == "__main__":
    main()
