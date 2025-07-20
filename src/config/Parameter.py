import os


class Parameter:
    image_path = os.path.abspath("../image")

    # ============================env相关参数===============================
    init_grid = [[0, 0, 0, 0, 0, -1, 0, 0],  # -1表示障碍，0为非障碍
                    [0, 0, 0, -1, 0, 0, 0, 0],
                    [0, 0, 0, -1, -1, 0, 0, 0],
                    [0, 0, 0, -1, -1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0]]
    init_start = (1, 1)
    init_end = (4, 7)
    init_target_reward = 1
    init_boundary_reward = -1
    init_forbidden_reward = -1
    init_random_start = False

    # ============================DP相关参数===============================
    discount_factor = 0.9

    # ============================画图相关参数===============================
    grid_color_dict = {0: "white", -1: "yellow"}
