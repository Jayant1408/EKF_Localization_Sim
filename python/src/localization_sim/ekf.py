import numpy as np

# from models import *
from .models import (
    motion_model,
    jacobian_F,
    meaasurement_model,
    jacobian_H,
    wrap_angle
)

class EKFLocalization:
    """
    EKF for state x = [px,py,yaw,v]
    Uses IMU (a, omega) in prediction and GPS (px,py) in update.  
    """

    def __init__(self, Q_diag: np.ndarray, R_diag: np.ndarray):
        self.Q = np.diag(Q_diag)
        self.R = np.diag(R_diag)

        # Initialize with some uncertainity 
        self.x = np.zeros(4) #[px,py,yaw,v]
        self.P = np.eye(4) * 1.0

    def predict(self, u: np.ndarray, dt: float) -> None:
        """
        Prediction step: x_k+1|k, P_k+1|k
        """
        # NonLiner prediction 
        self.x = motion_model(self.x, u, dt)
        F = jacobian_F(self.x, u, dt)

        # Covariance propogation: P = FPF^T + Q
        self.P = F @ self.P @ F.T + self.Q
    
    def update(self, z: np.ndarray) -> None:
        """
        Correction step using GPS measurement z = [px,py].
        """
        H = jacobian_H(self.x)
        z_pred = meaasurement_model(self.x)
        y = z - z_pred # innovation

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2]) # wrap yaw

        # Covariance update (simple form)
        I = np.eye(len(self.x))
        self.P = (I - K @ H) @ self.P
        