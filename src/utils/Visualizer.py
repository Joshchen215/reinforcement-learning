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
        plt.text(env.end[1], env.end[0], "end", ha='center', va='center')

        # 绘制网格线
        height, weight = env.grid.shape
        for y in range(height - 1):
            plt.plot([-0.5, weight - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(weight - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, height - 0.5], '-k', lw=2)
        # plt.savefig(Parameter.image_path + r'\grid_visualization.png')
        plt.tight_layout()
        plt.show()

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

    @staticmethod
    def plot_policy(env, policy, plot_title=None):
        """
        可视化策略
        Args:
            policy: (总状态数,)数组，每个状态的动作(0-3)
            plot_title: 图表标题
        """
        # 动作符号映射
        action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
        plt.figure(figsize=(5, 5), dpi=200)
        plt.imshow((env.grid <= -1) + (env.grid > 0) * 0.5, cmap='Greys', vmin=0, vmax=1)
        ax = plt.gca()
        ax.grid(False)
        plt.xticks([])
        plt.yticks([])

        if plot_title:
            plt.title(plot_title)

        # 在非终止状态位置显示动作方向
        for (int_obs, action) in enumerate(policy):
            y, x = env.int_to_state(int_obs)
            if (y, x) in env._non_term_states:
                action_arrow = action_names[action]
                plt.text(x + 1, y + 1, action_arrow, ha='center', va='center')

    @staticmethod
    def render(env, animation_interval=0.2):
        agent_star, traj_obj = Visualizer.set_canvas(env)
        # agent_circle.center = (agent_state[0], agent_state[1])
        agent_star.set_data([env.current[0]], [env.current[1]])
        traj_x, traj_y = zip(*env.start)
        traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)

    @staticmethod
    def set_canvas(env):
        plt.ion()
        canvas, ax = plt.subplots()
        ax.set_xlim(-0.5, len(env) - 0.5)
        ax.set_ylim(-0.5, len(env[0]) - 0.5)
        ax.xaxis.set_ticks(np.arange(-0.5, len(env), 1))
        ax.yaxis.set_ticks(np.arange(-0.5, len(env[0]), 1))
        ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.xaxis.set_ticks_position('top')

        idx_labels_x = [i for i in range(len(env))]
        idx_labels_y = [i for i in range(len(env[0]))]
        for lb in idx_labels_x:
            ax.text(lb, -0.75, str(lb + 1), size=10, ha='center', va='center', color='black')
        for lb in idx_labels_y:
            ax.text(-0.75, lb, str(lb + 1), size=10, ha='center', va='center', color='black')
        ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                       labeltop=False)

        target_rect = patches.Rectangle((env.end[0] - 0.5, env.end[1] - 0.5), 1, 1,
                                        linewidth=1, edgecolor=Parameter.color_target, facecolor=Parameter.color_target)
        ax.add_patch(target_rect)

        for (row, col), value in np.ndenumerate(env.grid):
            rect = patches.Rectangle((row - 0.5, col - 0.5), 1, 1, linewidth=1,
                                     edgecolor=Parameter.color_forbid, facecolor=Parameter.color_forbid)
            ax.add_patch(rect)

        agent_star, = ax.plot([], [], marker='*', color=Parameter.color_agent, markersize=20, linewidth=0.5)
        traj_obj, = ax.plot([], [], color=Parameter.color_trajectory, linewidth=0.5)

        return agent_star, traj_obj