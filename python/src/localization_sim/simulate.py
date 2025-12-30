import numpy as np

from . import config
from .models import motion_model, generate_reference_control
from .config import (
    DT,
    T_END,
    GPS_DT,
    GPS_NOISE_STD,
    ACC_NOISE_STD,
    GYRO_NOISE_STD,
    GPS_DROPOUT_PROB,
    RANDOM_SEED,
)

def simulate_truth_and_sensors():

    """
    Simulate the true trajectory + noisy IMU + GPS.
    Returns:
      t_arr          : (N,)
      x_true_hist    : (N, 4)  true state
      imu_meas_hist  : (N, 2)  [a_meas, omega_meas]
      gps_meas_hist  : list of (t, z) where z is (2,) GPS measurement
    """
    np.random.seed(RANDOM_SEED)

    N = int(T_END / DT) + 1
    t_arr = np.linspace(0.0, T_END, N)

    x_true = np.array([0.0,0.0,0.0,0.0]) #px, py, yaw, v

    x_true_hist = np.zeros((N,4))
    imu_meas_hist = np.zeros((N, 2))

    gps_meas_hist = [] 

    next_gps_time = 0.0

    for k, t in enumerate(t_arr):
        
        # True inputs
        a_true, omega_true = generate_reference_control(t)

        # Integrate true dynamics (no noise)
        u_true = np.array([a_true, omega_true])
        x_true = motion_model(x_true, u_true, DT)
        x_true_hist[k] = x_true

         # IMU measurements = true + noise
        a_meas = a_true + np.random.randn() * ACC_NOISE_STD
        omega_meas = omega_true + np.random.randn() * GYRO_NOISE_STD
        imu_meas_hist[k] = np.array([a_meas, omega_meas])

        # GPS Measurement at slower rate
        if t >= next_gps_time - 1e-9:
            # with dropout prob
            if np.random.rand() > GPS_DROPOUT_PROB:
                px_true, py_true, yaw_true, v_true = x_true
                z = np.array([
                    px_true + np.random.randn() * GPS_NOISE_STD,
                    py_true + np.random.randn() * GPS_NOISE_STD,
                ])
                gps_meas_hist.append((t, z))
            next_gps_time += GPS_DT

    return t_arr, x_true_hist, imu_meas_hist, gps_meas_hist


    




