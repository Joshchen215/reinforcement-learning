import os


class Parameter:
    image_path = os.path.abspath("../image")

    # ============================env相关参数===============================
    # init_grid = [[0, 0, 0, 0, 0, -1, 0, 0],  # -1表示障碍，0为非障碍
    #                 [0, 0, 0, -1, 0, 0, 0, 0],
    #                 [0, 0, 0, -1, -1, 0, 0, 0],
    #                 [0, 0, 0, -1, -1, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0],
    #                 [0, 0, 0, 0, 0, 0, 0, 0]]
    # init_start = (1, 1)
    # init_end = (1, 7)
    init_grid = [[0, 0, 0, 0, 0],  # -1表示障碍，0为非障碍
                 [0, -1, -1, 0, 0],
                 [0, 0, -1, 0, 0],
                 [0, -1, 0, -1, 0],
                 [0, -1, 0, 0, 0]]
    init_start = (0, 0)
    init_end = (3, 2)
    init_target_reward = 1
    init_boundary_reward = -1
    init_forbidden_reward = -10
    init_random_start = False

    # ============================DP相关参数===============================
    discount_factor = 0
    eps = 0.00001
    max_iter = 10

    # ============================画图相关参数===============================
    grid_color_dict = {0: "white", -1: "yellow"}
    color_forbid = (0.9290, 0.6940, 0.125)
    color_target = (0.3010, 0.7450, 0.9330)
    color_policy = (0.4660, 0.6740, 0.1880)
    color_trajectory = (0, 1, 0)
    color_agent = (0, 0, 1)