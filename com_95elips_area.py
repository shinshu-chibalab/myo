import numpy as np
import matplotlib.pyplot as plt


def plot_com_95ellipse(com_log, title="COM 95% Confidence Ellipse"):
    com_log = np.asarray(com_log, dtype=float)
    com_xy = com_log[:, :2]

    center = np.mean(com_xy, axis=0)
    cov = np.cov(com_xy.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    # 大きい固有値順に並べる
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # 2次元95%信頼楕円のカイ二乗値
    chi2_95 = 5.991

    # 楕円の半径
    width = np.sqrt(chi2_95 * eigvals[0])
    height = np.sqrt(chi2_95 * eigvals[1])

    # 楕円の角度
    theta = np.linspace(0, 2*np.pi, 300)
    ellipse = np.array([
        width * np.cos(theta),
        height * np.sin(theta),
    ])

    ellipse_rotated = eigvecs @ ellipse
    ellipse_x = ellipse_rotated[0] + center[0]
    ellipse_y = ellipse_rotated[1] + center[1]

    area = np.pi * width * height

    plt.figure(figsize=(6, 6))
    plt.plot(com_xy[:, 0], com_xy[:, 1], marker="o", markersize=2, linewidth=1, label="COM trajectory")
    plt.plot(ellipse_x, ellipse_y, linewidth=2, label="95% confidence ellipse")
    plt.scatter(center[0], center[1], marker="x", s=80, label="Mean COM")

    plt.axis("equal")
    plt.xlabel("COM x [m]")
    plt.ylabel("COM y [m]")
    plt.title(f"{title}\nArea = {area:.6e} m²")
    plt.legend()
    plt.grid(True)
    plt.show()

    return area

com_log1 = np.array([
    [0.00, 0.00, 1.00],
    [0.00, 0.00, 1.00],
    [0.00, 0.00, 1.00],
    [0.00, 0.00, 1.00],
    [0.00, 0.00, 1.00],
])

com_log2 = np.array([
    [-0.02, 0.00, 1.00],
    [-0.01, 0.00, 1.00],
    [ 0.00, 0.00, 1.00],
    [ 0.01, 0.00, 1.00],
    [ 0.02, 0.00, 1.00],
])

com_log3 = np.array([
    [0.00, -0.02, 1.00],
    [0.00, -0.01, 1.00],
    [0.00,  0.00, 1.00],
    [0.00,  0.01, 1.00],
    [0.00,  0.02, 1.00],
])

theta4 = np.linspace(0, 2*np.pi, 100)
r4 = 0.01
com_log4 = np.column_stack([
    r4*np.cos(theta4),
    r4*np.sin(theta4),
    np.ones_like(theta4)
])

theta5 = np.linspace(0, 2*np.pi, 100)
r5 = 0.03
com_log5 = np.column_stack([
    r5*np.cos(theta5),
    r5*np.sin(theta5),
    np.ones_like(theta5)
])

theta6 = np.linspace(0, 2*np.pi, 100)
r6_x = 0.03
r6_y = 0.01
com_log6 = np.column_stack([
    r6_x*np.cos(theta6),
    r6_y*np.sin(theta6),
    np.ones_like(theta6)
])

np.random.seed(0)
com_log7 = np.column_stack([
    np.random.normal(0.00, 0.005, 1000),
    np.random.normal(0.00, 0.003, 1000),
    np.ones(1000),
])

np.random.seed(1)
com_log8 = np.column_stack([
    np.random.normal(0.00, 0.010, 1000),
    np.random.normal(0.00, 0.006, 1000),
    np.ones(1000),
])

np.random.seed(2)
com_log9 = np.column_stack([
    np.random.normal(0.00, 0.005, 1000),
    np.random.normal(0.00, 0.005, 1000),
    np.ones(1000),
])
com_log9[500] = [0.15, 0.15, 1.0]

np.random.seed(3)
x = np.cumsum(np.random.normal(0, 0.0003, 1000))
y = np.cumsum(np.random.normal(0, 0.0002, 1000))
com_log10 = np.column_stack([
    x,
    y,
    np.ones(1000)
])


area1 = plot_com_95ellipse(com_log1, title="COM Log 1")
area2 = plot_com_95ellipse(com_log2, title="COM Log 2")
area3 = plot_com_95ellipse(com_log3, title="COM Log 3")
area4 = plot_com_95ellipse(com_log4, title="COM Log 4")
area5 = plot_com_95ellipse(com_log5, title="COM Log 5")
area6 = plot_com_95ellipse(com_log6, title="COM Log 6")
area7 = plot_com_95ellipse(com_log7, title="COM Log 7")
area8 = plot_com_95ellipse(com_log8, title="COM Log 8")
area9 = plot_com_95ellipse(com_log9, title="COM Log 9")
area10 = plot_com_95ellipse(com_log10, title="COM Log 10")

print("95% ellipse area:", area1)
print("95% ellipse area:", area2)
print("95% ellipse area:", area3)
print("95% ellipse area:", area4)
print("95% ellipse area:", area5)
print("95% ellipse area:", area6)
print("95% ellipse area:", area7)
print("95% ellipse area:", area8)
print("95% ellipse area:", area9)
print("95% ellipse area:", area10)