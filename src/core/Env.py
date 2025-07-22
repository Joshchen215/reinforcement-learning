import numpy as np
import matplotlib.pyplot as plt
from src.core.Enum.Action import Action


class GridWorld:
    """网格世界环境类"""
    def __init__(self, grid, start, end, target_reward, boundary_reward, forbidden_reward, random_start):
        """

        :param grid: 2D 列表或数组，非负值为正常格，-1 为障碍
        :param start: 起始位置 (row, col)
        :param end: 终点位置
        :param target_reward: 到达终点的奖励
        :param boundary_reward: 碰撞墙体惩罚
        :param forbidden_reward: 进入禁行区域的惩罚
        :param random_start: 是否随机起点
        """
        self.grid = np.array(grid)
        self.start = np.array(start)
        self.end = np.array(end)
        self.target_reward = target_reward
        self.boundary_reward = boundary_reward
        self.forbidden_reward = forbidden_reward
        self.random_start = random_start
        self.current = self.start
        self.reset()

    def reset(self):
        """重置开始状态"""
        self.current = self.start

    def state2State_ind(self, state):
        """返回当前状态编号"""
        row, col = state
        return row * self.grid.shape[1] + col

    def state_ind2state(self, state):
        """根据状态编号放回状态对应的网格坐标"""
        row = state // self.grid.shape[1]
        col = state % self.grid.shape[1]
        return np.array((row, col))

    def step_determine(self, state_id, action: Action):
        """
        P(s'|s,a)只有一种情况的概率为1，其他情况概率为0
        根据执行动作，返回 (next_state, reward)
        """
        row, col = self.state_ind2state(state_id)
        # 计算新坐标
        if action == Action.UP:
            new_row, new_col = row - 1, col
        elif action == Action.RIGHT:
            new_row, new_col = row, col + 1
        elif action == Action.DOWN:
            new_row, new_col = row + 1, col
        elif action == Action.LEFT:
            new_row, new_col = row, col - 1
        elif action == Action.STAY:
            new_row, new_col = row, col
        else:
            raise ValueError(f"未知动作: {action}")

        # 判断新坐标是否合法
        # 撞墙
        if new_row < 0 or new_row >= self.grid.shape[0] or new_col < 0 or new_col >= self.grid.shape[1]:
            reward = self.boundary_reward
            new_row, new_col = row, col
        # 普通空格
        elif new_row != self.end[0] and new_col != self.end[1]:
            reward = 0.0
        # 终止状态
        else:
            reward = self.target_reward

        return self.state2State_ind(np.array((new_row, new_col))), reward
