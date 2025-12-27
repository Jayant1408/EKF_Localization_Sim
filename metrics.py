import numpy as np

def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """
    Wrap angles to [-pi, pi]. Works with scalars or numpy arrays.
    
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def rmse(a : np.ndarray, b : np.ndarray) -> float:
    """
    Root Mean Squared Error between two arrays with same shape.
    """

    err = a - b
    return float(np.sqrt(np.mean(e * e)))

def rmse_position(true_states: np.ndarray, estimated_states: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Position RMSE for [px, py].
    Returns: 
        rmse_total: scalar RMSE of Euclidean position error.
        e_norm : (N, ) position error norm over time
    """

    e_xy = estimated_states[:, :2] - true_states[:, :2]
    e_norm = np.linalg.norm(e_xy, axis = 1)
    rmse_total = float(np.sqrt(np.mean(e_norm * e_norm)))
    return rmse_total, e_norm


def rmse_yaw(true_states: np.ndarray, estimated_states: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Yaw RMSE for yaw angle (wrapped).
    Returns:
        rmse_total : scalar
        e_yaw : (N, ) yaw error over time (wrapped)
    """

    e = wrap_angle(estimated_states[:, 2] - true_states[:, 2])
    rmse_total = float(np.sqrt(np.mean(e * e)))

    return rmse_total, e

def rmse_speed(true_states: np.ndarray, estimated_states: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Speed RMSE for v.
    Returns:
        rmse_total : scalar
        e_v = : (N, ) speed error over time
    """

    e = estimated_states[:, 3] - true_states[:, 3]
    rmse_total = float(np.sqrt(np.mean(e * e)))
    return rmse_total, e