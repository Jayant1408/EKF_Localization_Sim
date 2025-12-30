import numpy as np

# ==============================
# Global Simulation Config
# ==============================

DT = 0.01             # [s] simulation / IMU update period
T_END = 40.0          # [s] total simulation time

GPS_DT = 0.2          # [s] GPS update period (slower than IMU)

# Sensor noise
GPS_NOISE_STD = 1.5   # [m] GPS position noise
ACC_NOISE_STD = 0.3   # [m/s^2] accel noise (IMU)
GYRO_NOISE_STD = 0.05 # [rad/s] yaw-rate noise (IMU)

# EKF process and measurement noise (tune these)
Q_diag = np.array([0.05, 0.05, 0.01, 0.2])  # x, y, yaw, v
R_diag = np.array([GPS_NOISE_STD**2, GPS_NOISE_STD**2])

# Sensor dropout
GPS_DROPOUT_PROB = 0.1  # 10% of GPS updates randomly dropped

# Random seed for reproducibility
RANDOM_SEED = 0
