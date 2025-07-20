import numpy as np
import matplotlib.pyplot as plt
from src.core.Enum.Action import Action


class GridWorld:
    """网格世界环境类"""
    def __init__(self, grid, start, terminal_rewards=10, reward_wall=-5, random_start=False):
        """
        Args:
            grid: 2D 列表或数组，非负值为正常格，-1 为障碍
            start: 起始位置 (row, col)
            terminal_rewards: dict，终止状态到奖励的映射 {(r, c): reward}
            reward_wall: 碰撞墙体惩罚
            random_start: 是否随机起点
        """
        self.grid = np.array(grid)
        self.start = start
        self.terminal_rewards = terminal_rewards
        self.reward_wall = reward_wall
        self.random_start = random_start

        # 带边界的网格，边界视为墙
        self._grid_padded = np.pad(self.grid, pad_width=1, mode='constant', constant_values=-1)
        self._state_shape = self.grid.shape
        self.reset()

    def reset(self):
        """重置到起始状态"""
        if self.random_start:
            non_term_states = [(r, c) for r in range(self.grid.shape[0])
                               for c in range(self.grid.shape[1]) if self.grid[r, c] >= 0]
            chosen = non_term_states[np.random.randint(len(non_term_states))]
            self._state = (chosen[0] + 1, chosen[1] + 1)
        else:
            self._state = (self.start[0] + 1, self.start[1] + 1)
        return self.current_state()

    def current_state(self):
        """返回当前状态编号"""
        r, c = self._state
        rows, cols = self._state_shape
        return (r - 1) * cols + (c - 1)

    def step(self, action: Action):
        """执行动作，返回 (next_state, reward, done)"""
        r, c = self._state
        # 计算新坐标
        if action == Action.UP:
            nr, nc = r - 1, c
        elif action == Action.RIGHT:
            nr, nc = r, c + 1
        elif action == Action.DOWN:
            nr, nc = r + 1, c
        elif action == Action.LEFT:
            nr, nc = r, c - 1
        else:
            raise ValueError(f"未知动作: {action}")

        cell = self._grid_padded[nr, nc]
        # 撞墙
        if cell == -1:
            reward = self.reward_wall
            nr, nc = r, c
            done = False
        # 普通空格
        elif (nr-1, nc-1) not in self.terminal_rewards:
            reward = 0.0
            done = False
        # 终止状态
        else:
            reward = self.terminal_rewards[(nr-1, nc-1)]
            done = True
        self._state = (nr, nc)
        next_state = self.current_state()
        return next_state, reward, done

    def plot_grid(self, plot_title=None):
        """可视化网格世界布局"""
        plt.figure(figsize=(5, 5), dpi=200)
        plt.imshow(self._grid_padded <= -1, cmap='binary', interpolation="nearest")
        ax = plt.gca()
        ax.grid(False)  # 关闭默认网格
        plt.xticks([])
        plt.yticks([])

        if plot_title:
            plt.title(plot_title)

        # 标记起点S
        plt.text(
            self._start_state[1] + 1, self._start_state[0] + 1,
            r"$\mathbf{S}$", ha='center', va='center')

        # 标记目标状态
        for goal_state in self._goal_states:
            plt.text(
                goal_state[1] + 1, goal_state[0] + 1,
                "{:d}".format(self._grid[goal_state[0], goal_state[1]]), ha='center', va='center')

        # 绘制网格线
        h, w = self._grid_padded.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)

    def plot_state_values(self, state_values, value_format="{:.1f}", plot_title=None):
        """
        可视化状态值函数
        Args:
            state_values: (总状态数,)数组，每个状态的值
            value_format: 数值格式
            plot_title: 图表标题
        """
        plt.figure(figsize=(5, 5), dpi=200)
        # 用不同灰度表示墙壁和目标状态
        plt.imshow((self._grid_padded <= -1) + (self._grid_padded > 0) * 0.5, cmap='Greys', vmin=0, vmax=1)
        ax = plt.gca()
        ax.grid(False)
        plt.xticks([])
        plt.yticks([])

        if plot_title:
            plt.title(plot_title)

        # 在非终止状态位置显示状态值
        for (int_obs, state_value) in enumerate(state_values):
            y, x = self.int_to_state(int_obs)
            if (y, x) in self._non_term_states:
                plt.text(x + 1, y + 1, value_format.format(state_value), ha='center', va='center')

        # 绘制网格线
        h, w = self._grid_padded.shape
        for y in range(h - 1):
            plt.plot([-0.5, w - 0.5], [y + 0.5, y + 0.5], '-k', lw=2)
        for x in range(w - 1):
            plt.plot([x + 0.5, x + 0.5], [-0.5, h - 0.5], '-k', lw=2)

    def plot_policy(self, policy, plot_title=None):
        """
        可视化策略
        Args:
            policy: (总状态数,)数组，每个状态的动作(0-3)
            plot_title: 图表标题
        """
        # 动作符号映射
        action_names = [r"$\uparrow$", r"$\rightarrow$", r"$\downarrow$", r"$\leftarrow$"]
        plt.figure(figsize=(5, 5), dpi=200)
        plt.imshow((self._grid_padded <= -1) + (self._grid_padded > 0) * 0.5, cmap='Greys', vmin=0, vmax=1)
        ax = plt.gca()
        ax.grid(False)
        plt.xticks([])
        plt.yticks([])

        if plot_title:
            plt.title(plot_title)

        # 在非终止状态位置显示动作方向
        for (int_obs, action) in enumerate(policy):
            y, x = self.int_to_state(int_obs)
            if (y, x) in self._non_term_states:
                action_arrow = action_names[action]
                plt.text(x + 1, y + 1, action_arrow, ha='center', va='center')
