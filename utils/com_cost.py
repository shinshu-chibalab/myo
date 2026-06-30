import numpy as np

def com_cost(com_log):
    com_log = np.array(com_log)
    diffs = np.diff(com_log, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))
