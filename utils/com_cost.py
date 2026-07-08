import numpy as np

def com_cost(com_log):
    com_log = np.array(com_log)
    diffs = np.diff(com_log, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))

def com_95ellipse_area_xz(com_log):
    com_log = np.asarray(com_log, dtype=float)

    x = com_log[:, 0]
    z = com_log[:, 2]

    dx = x - np.mean(x)
    dz = z - np.mean(z)

    n = len(x)
    if n < 2:
        return 0.0

    var_x = np.sum(dx * dx) / (n - 1)
    var_z = np.sum(dz * dz) / (n - 1)
    cov_xz = np.sum(dx * dz) / (n - 1)

    det_cov = var_x * var_z - cov_xz * cov_xz
    det_cov = max(det_cov, 0.0)

    chi2_95 = 5.991

    area = np.pi * chi2_95 * np.sqrt(det_cov)

    return float(area)

import matplotlib.pyplot as plt

def plot_com_95ellipse(com_log, title="COM 95% Confidence Ellipse"):
    com_log = np.asarray(com_log, dtype=float)

    # x-z plane (sagittal plane)
    com_xz = com_log[:, [0, 2]]

    center = np.mean(com_xz, axis=0)
    cov = np.cov(com_xz.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # 大きい固有値順
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eigvals = np.maximum(eigvals, 0.0)

    chi2_95 = 5.991

    # 半軸長
    a = np.sqrt(chi2_95 * eigvals[0])
    b = np.sqrt(chi2_95 * eigvals[1])

    theta = np.linspace(0, 2 * np.pi, 300)

    ellipse = np.vstack((
        a * np.cos(theta),
        b * np.sin(theta),
    ))

    ellipse = eigvecs @ ellipse

    ellipse_x = ellipse[0] + center[0]
    ellipse_z = ellipse[1] + center[1]

    area = np.pi * a * b

    plt.figure(figsize=(6, 6))

    # COM軌跡
    plt.plot(
        com_xz[:, 0],
        com_xz[:, 1],
        "o-",
        markersize=2,
        linewidth=1,
        label="COM trajectory",
    )

    # 95%信頼楕円
    plt.plot(
        ellipse_x,
        ellipse_z,
        linewidth=2,
        label="95% confidence ellipse",
    )

    # 重心平均
    plt.scatter(
        center[0],
        center[1],
        marker="x",
        s=80,
        label="Mean COM",
    )

    plt.axis("equal")
    plt.xlabel("COM x [m]")
    plt.ylabel("COM z [m]")
    plt.title(f"{title}\nArea = {area:.6e} m²")
    plt.grid(True)
    plt.legend()
    plt.show()

    return area