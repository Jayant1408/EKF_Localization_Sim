import numpy as np

def wrap_angle(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

# ==============================
# Motion & Measurement Models
# State x = [px, py, yaw, v]
# IMU measurements: a (longitudinal accel), omega (yaw rate)
# GPS measurements: z = [px, py] (position)
# ==============================

def motion_model(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    x: (4,) [px,py,yaw,v]
    u: (2,) [a,omega]
    """
    px, py, yaw, v = x
    a, omega = u

    # Simple kinematic model
    yaw_next = yaw + omega * dt
    v_next = v + a * dt
    px_next = px + v * np.cos(yaw) * dt
    py_next = py + v * np.sin(yaw) * dt


    return np.array([px_next, py_next, wrap_angle(yaw_next), v_next])

def jacobian_F(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """
    Jacobian of motion model w.r.t state x
    """
    px,py,yaw,v = x
    a,omega = u # unused but kept for symmetry

    r"""
    \(J=\left[\begin{matrix}\frac{\partial f_{1}}{\partial x_{1}}&\cdots &\frac{\partial f_{1}}{\partial x_{n}}\\ \vdots &\ddots &\vdots \\ \frac{\partial f_{n}}{\partial x_{1}}&\cdots &\frac{\partial f_{n}}{\partial x_{n}}\end{matrix}\right]\) 
    """
    F = np.eye(4)
    F[0,2] = -v * np.sin(yaw) * dt
    F[0,3] = np.cos(yaw) * dt
    F[1,2] = v * np.cos(yaw) * dt
    F[1,3] = np.sin(yaw) * dt
    #yaw, v rows remain identity
    return F

def meaasurement_model(x: np.ndarray) -> np.ndarray:
    """
    GPS: measures position only
    z = [px, py]
    """
    px,py,yaw, v = x
    return np.array([px,py])

def jacobian_H(x: np.ndarray) -> np.ndarray:
    """
    Jacobian of measurement model w.r.t state
    For z = [px,py]
    H = [1 0 0 0
         0 1 0 0]
    """

    H = np.zeros((2,4))
    H[0,0] = 1.0
    H[1,1] = 1.0
    return H

# ==============================
# Reference input generation
# ==============================

def generate_reference_control(t: float) -> tuple[float, float]:
    """
    Define a smooth motion profile via IMU inputs:
    - Accelerate, then cruise, then decelerate
    - Slowly varying yaw_rate to make a curve / arc
    Returns: a_true, omega_true
    """

    # Longitudinal accel profile
    if t < 5.0:
        a = 0.8  # accelrate
    elif t < 25.0:
        a = 0.0 # cruise
    elif t < 30.0:
        a = -0.8 # brake
    else:
        a = 0.0 # stop

     # Yaw-rate profile: gentle turn for a while
    if 10.0 < t < 25.0:
        omega = 0.15  # rad/s
    else:
        omega = 0.0

    return a, omega





