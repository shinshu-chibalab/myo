import numpy as np

def standing_controller(Kp, l0, l, Kd, v0, v, Kf, f0, f, ff):

    u = ff + Kp * np.maximum(l - l0, 0) + Kd * np.maximum(v - v0, 0) + Kf * (f - f0)

    return np.clip(u, 0.0, 1.0, out=u)
