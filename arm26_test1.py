from pd_gain_optimizer import PDGainOptimizer

if __name__ == "__main__":
    # --- モデル情報 ---
    model_path = "myo_sim/arm26/arm26_cvt3.xml"
    muscles = ["BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed"]
    target_len = [0.37, 0.29, 0.12,
                  0.31, 0.20, 0.19]

    optimizer = PDGainOptimizer(model_path, muscles, target_len, model_name="arm26")

    # --- CMA-ES最適化 ---
    best_params = optimizer.optimize(maxiter=1000)  # デモ用に10世代
    print("最適パラメータ:", best_params)

    # --- 最適化されたゲインでレンダリング付きシミュレーション ---
    optimizer.run_simulation(best_params, render=True, plot=True)

    # --- 結果 ---
    num_muscles = len(muscles)
    Kp_opt = best_params[:num_muscles]
    Kd_opt = best_params[num_muscles:]
    print("最適化されたKp:", Kp_opt)
    print("最適化されたKd:", Kd_opt)
