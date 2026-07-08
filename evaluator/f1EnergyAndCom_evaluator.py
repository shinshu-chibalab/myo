import numpy as np
from utils.com_cost import com_95ellipse_area_xz

def standing_evaluator(logs):
    total_Edot = logs["total_Edot"]
    sim_steps = logs["sim_steps"]
    com_log = logs["com_log"]
    fall_cost = logs["fall_cost"]

    f1 = 6e2 * com_95ellipse_area_xz(com_log) + (fall_cost)

    return np.array([f1], dtype=float)