import numpy as np

def create_x0_bounds(muscles, symmetry=False):

    muscle_names = list(muscles.keys())

    if symmetry:
        muscle_names = [m for m in muscle_names if m.endswith("_r")]

    # 先頭の筋に書かれているパラメータ順を使う
    first_muscle = muscle_names[0]
    params = list(muscles[first_muscle].keys())

    x0 = []
    lower = []
    upper = []

    for param in params:
        for muscle in muscle_names:
            info = muscles[muscle][param]

            x0.append(info["x0"])

            low, high = info["range"]
            lower.append(low)
            upper.append(high)

    return (
        np.array(x0, dtype=float),
        np.array(lower, dtype=float),
        np.array(upper, dtype=float),
    )

def expand_params(x, muscles, symmetry=False):
    x = np.asarray(x, dtype=float)

    all_muscle_names = list(muscles.keys())

    if symmetry:
        use_muscle_names = [m for m in all_muscle_names if m.endswith("_r")]
    else:
        use_muscle_names = all_muscle_names

    n_use_muscles = len(use_muscle_names)
    param_keys = list(muscles[use_muscle_names[0]].keys())

    values = {}

    for i, param_key in enumerate(param_keys):
        start = i * n_use_muscles
        end = (i + 1) * n_use_muscles

        use_values = x[start:end]
        param_full = {}

        for muscle, value in zip(use_muscle_names, use_values):
            param_full[muscle] = value

            if symmetry and muscle.endswith("_r"):
                left_muscle = muscle[:-2] + "_l"

                if left_muscle in all_muscle_names:
                    param_full[left_muscle] = value

        values[param_key] = np.array(
            [param_full[m] for m in all_muscle_names],
            dtype=float
        )

    return values