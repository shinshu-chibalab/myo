import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
from cma import CMAEvolutionStrategy
import skvideo.io
import os

class PDGainOptimizer:
    def __init__(self, model_path, muscles, sim_steps=100):
        self.model = MjModel.from_xml_path(model_path)
        self.data = MjData(self.model)
        self.muscles = muscles
        self.tendon_ids = [self.model.tendon(f"{m}_tendon").id for m in muscles]
        self.actuator_ids = [self.model.actuator(m).id for m in muscles]
        # 初期筋長を目標に
        mj_step(self.model, self.data)
        self.target_len = self.data.ten_length[self.tendon_ids].copy()
        self.sim_steps = sim_steps

    def run_simulation(self, params, render=False):
        """params: [Kp(16), Kd(16)]"""
        num_muscles = len(self.muscles)
        Kp = np.array(params[:num_muscles])
        Kd = np.array(params[num_muscles:])

        data = MjData(self.model)
        data.qpos[:] = self.model.key_qpos[0]
        data.qvel[:] = 0
        data.qacc[:] = 0

        renderer = Renderer(self.model, height=400, width=400) if render else None
        frames = []
        cost = 0.0

        for _ in range(self.sim_steps):
            l = data.ten_length[self.tendon_ids]
            v = data.ten_velocity[self.tendon_ids]
            diff_len = l - self.target_len

            # PD 制御
            u = Kp * diff_len + Kd * v
            u = np.clip(u, -1.0, 1.0)

            ctrl = np.zeros(self.model.nu)
            for act_id, ui in zip(self.actuator_ids, u):
                if ui > 0:
                    ctrl[act_id] = ui

            data.ctrl[:] = ctrl
            mj_step(self.model, data)

            cost += np.sum(diff_len**2)

            if render:
                renderer.update_scene(data)
                frames.append(renderer.render())

        if render:
            os.makedirs("videos", exist_ok=True)
            skvideo.io.vwrite("videos/stand_optimized.mp4",
                              np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
            renderer.close()

        return cost / self.sim_steps

    def optimize(self, x0=None, sigma0=0.5, maxiter=30, popsize=30):
        num_muscles = len(self.muscles)
        if x0 is None:
            # 初期ゲイン（Kp=1.0, Kd=0.05）
            x0 = [1.0] * num_muscles + [0.05] * num_muscles

        opts = {"maxiter": maxiter, "popsize": popsize}
        es = CMAEvolutionStrategy(x0, sigma0, opts)

        while not es.stop():
            solutions = es.ask()
            costs = [self.run_simulation(s) for s in solutions]
            es.tell(solutions, costs)
            es.disp()

        self.best_params = es.result.xbest
        return self.best_params

if __name__ == "__main__":
    model_path = "myo_sim/gait10dof18musc/gait10dof18musc_cvt3.xml"
    muscles = ["hamstrings_r","glut_max_r","iliopsoas_r",
               "rect_fem_r","vasti_r","gastroc_r","soleus_r","tib_ant_r",
               "hamstrings_l","glut_max_l","iliopsoas_l",
               "rect_fem_l","vasti_l","gastroc_l","soleus_l","tib_ant_l"]

    optimizer = PDGainOptimizer(model_path, muscles, sim_steps=100)

    # CMA-ES による最適化（50世代）
    best_params = optimizer.optimize(maxiter=1000)
    print("最適化されたパラメータ:", best_params)

    # 最適化されたゲインでレンダリング付きシミュレーション
    optimizer.run_simulation(best_params, render=True)

    num_muscles = len(muscles)
    Kp_opt = best_params[:num_muscles]
    Kd_opt = best_params[num_muscles:]
    print("最適化された Kp:", Kp_opt)
    print("最適化された Kd:", Kd_opt)
