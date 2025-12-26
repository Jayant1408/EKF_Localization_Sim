import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_with_ekf(t_arr, x_true_hist, x_est_hist, P_hist, gps_meas_hist):
    plt.figure()
    plt.axis('equal')
    plt.grid(True)

    # True trajectory
    plt.plot(x_true_hist[:,0], x_true_hist[:,1], label='True',linewidth=2)

    # Estimated trajectory
    plt.plot(x_est_hist[:,0], x_est_hist[:, 1], label = "EKF estimate", linestyle='--')
    
    # GPS Measurments
    if gps_meas_hist:
        gps_arr = np.array([z for (_,z) in gps_meas_hist])
        plt.scatter(gps_arr[:, 0], gps_arr[:, 1], s = 10, alpha = 0.5, label = "GPS")

    # Plot covariance ellipses every N steps
    N = len(t_arr)
    step = max(1, N // 20) # ~20 ellipses max
    for k in range(0, N, step):
        Pk = P_hist[k]
        plot_cov_ellipse(x_est_hist[k, 0:2], Pk[0:2, 0:2])

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("EKF Localization: True vs Estimated Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_cov_ellipse(mean_xy, cov_xy, n_points = 40, conf = 0.95):
    """
    Draw a covariance ellipse for 2D Gaussian with mean and 2*2 covariance.
    conf: confidence level (0.95 ~ 2.4477 sigma in 2D)
    """
    # Chi-square value for 2 DOF and desired confidence
    # Approx 5.991 for 95% confidence -> sqrt(5.991) ~ 2.4477

    if conf == 0.95:
        s = np.sqrt(5.991)
    else:
        # fallback: 2-sigma
        s = 2.0

    # Eigen-decomposition
    vals, vecs = np.linalg.eigh(cov_xy)
    
    # Sort eigenvalues (largest first)
    order = vals.argsort()[: : -1]
    vals = vals[order]
    vecs = vecs[:, order]

    # Ellipse axes lengths
    axes = s * np.sqrt(vals)

    # ELlipse points
    theta = np.linspace(0,2 * np.pi, n_points)
    circle = np.stack([np.cos(theta), np.sin(theta)],axis = 0)

    # Transform: scale + rotate

    ellipse = (vecs @ np.diag(axes)) @ circle
    ellipse = ellipse.T + mean_xy #(n_points, 2)

    plt.plot(ellipse[:, 0], ellipse[:, 1], linewidth = 0.8, alpha = 0.7
    )


