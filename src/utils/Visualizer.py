import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.config.Parameter import Parameter


class Visualizer:
    @staticmethod
    def plot_env(env, color_dict=Parameter.grid_color_dict):
        """
        可视化网格世界布局
        """
        plt.figure(figsize=(5, 5), dpi=200)
        plt.imshow(env.grid <= -1, cmap='binary', interpolation="nearest")
        ax = plt.gca()
        ax.grid(False)  # 关闭默认网格
        plt.xticks([])
        plt.yticks([])
        # 标记起点S
        plt.text(
            env.start[1], env.start[0],
            "start", ha='center', va='center')

        # 标记目标状态
        plt.text(env.end[1], env.end[0], "end",ha='center', va='center')

        # 绘制网格线
        height, weight = env.grid.shape
        for y in range(height - 1):
            plt.plot([-0.5, weight - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(weight - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, height - 0.5], '-k', lw=2)
        plt.savefig(Parameter.image_path+r'\grid_visualization.png')

    @staticmethod
    def plot_state_values(env, state_values):
        """可视化状态值函数"""
        plt.figure(figsize=(5, 5), dpi=200)
        # 用不同灰度表示墙壁和目标状态
        plt.imshow((env.gird <= -1) + (env.gird > 0) * 0.5, cmap='Greys', vmin=0, vmax=1)
        ax = plt.gca()
        ax.grid(False)
        plt.xticks([])
        plt.yticks([])

        # 在非终止状态位置显示状态值
        for (int_obs, state_value) in enumerate(state_values):
            y, x = env.int_to_state(int_obs)
            if (y, x) in env._non_term_states:
                plt.text(x + 1, y + 1, state_value, ha='center', va='center')

        # 绘制网格线
        height, weight = env.grid.shape
        for y in range(height - 1):
            plt.plot([-0.5, weight - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(weight - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, height - 0.5], '-k', lw=2)
        plt.savefig(Parameter.image_path + r'\policy_visualization.png')