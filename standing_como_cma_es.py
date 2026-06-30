import numpy as np
from optim.como_cma_es import COMO_CMA_ES
from render.render import render_video
from controller.Standing_Controller import standing_controller
from evaluator.f1Energy_f2Com_evaluator import standing_evaluator
# from x0_gait10dof18musc import muscles18_v1
from x0_gait10dof24musc import muscles24_v0


if __name__ == "__main__":

    # model_path = "myo_sim/gait10dof18musc/gait10dof18musc_cvt6.xml"
    model_path = "myo_sim/gait10dof24musc/gait10dof24musc_cvt2.xml"
    # model_name = "gait10dof18musc_standing_v2(como_cma_es)"
    model_name = "gait10dof24musc_standing_v1(como_cma_es)"
    # muscles = muscles18_v1
    muscles = muscles24_v0

    sim_steps = 1000
    sigma0 = 0.05
    popsize = 896
    maxiter = 3000
    delay_time = 0.20
    noise_std = 0.00
    n_jobs = 56
    symmetry = True
    n_kernels = 8
    reference_point=(3e3, 3e3)

    controller = standing_controller
    evaluator = standing_evaluator

    optimizer = COMO_CMA_ES(
        model_path=model_path,
        muscles=muscles,
        controller=controller,
        evaluator=evaluator,
        sim_steps=sim_steps,
        model_name=model_name,
    )

    result = optimizer.optimize(
        sigma0=sigma0,
        maxiter=maxiter,
        popsize=popsize,
        delay_time=delay_time,
        noise_std=noise_std,
        n_jobs=n_jobs,
        symmetry=symmetry,
        n_kernels=n_kernels,
        reference_point=reference_point,
    )

    best_params = result["params"]
    best_fitness = result["fitness"]

    print(
        f"model_path: {model_path}\n"
        f"model_name: {model_name}\n"
        f"sim_steps:  {sim_steps}\n"
        f"popsize:    {popsize}\n"
        f"maxiter:    {maxiter}\n"
        f"delay_time: {delay_time}\n"
        f"noise_std:  {noise_std}\n"
        f"n_jobs:     {n_jobs}\n"
        f"symmetry:   {symmetry}\n"
    )

    cameras = ["front", "diagonal", "side", "oblique"]

    for i, (params, fitness) in enumerate(zip(best_params, best_fitness), start=1):

        fit_text = "_".join(
            f"f{j+1}_{value}"
            for j, value in enumerate(fitness)
        )

        opt_name = f"{fit_text}_res{i}"

        for camera in cameras:
            render_video(
                model_path=model_path,
                x=params,
                muscles=muscles,
                controller=controller,
                sim_steps=sim_steps,
                model_name=model_name,
                delay_time=delay_time,
                noise_std=noise_std,
                camera=camera,
                opt_name=opt_name,
            )

        print(f"\n[Solution {i}]")

        for j, value in enumerate(fitness, start=1):
            print(f"f{j}: {value}")

        print("l0:", params.get("l0"))
        print("Kp:", params.get("Kp"))
        print("Kd:", params.get("Kd"))
        print("ff:", params.get("ff"))