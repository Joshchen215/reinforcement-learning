from typing import List

import numpy as np
import matplotlib.pyplot as plt
from src.core.Enum.Action import Action


class GridWorld:
    """网格世界环境类"""

    def __init__(self, grid, start, end, target_reward, boundary_reward, forbidden_reward):
        """

        :param grid: 2D 列表或数组，非负值为正常格，-1 为障碍
        :param start: 起始位置 (row, col)
        :param end: 终点位置
        :param target_reward: 到达终点的奖励
        :param boundary_reward: 碰撞墙体惩罚
        :param forbidden_reward: 进入禁行区域的惩罚
        """
        self.grid = np.array(grid)
        self.start = np.array(start)
        self.end = np.array(end)
        self.target_reward = target_reward
        self.boundary_reward = boundary_reward
        self.forbidden_reward = forbidden_reward
        self.current = self.start  # 当前位置
        self.policy = None  # 策略 P(a|s)
        self.transition_prob = None  # 转移概率 P(s' | s,a)
        self.rewards = None  # 奖励 r_(s,a,s')
        self.state_values = np.zeros(self.grid.size, dtype=float)  # 状态值
        self.state_action_value = np.zeros(self.grid.size, dtype=float)
        self._initial()

    def _reset(self):
        """重置开始状态"""
        self.current = self.start

    def _initial(self):
        """初始化"""
        self._reset()
        # 初始化转移概率 transition_prob[s][a][s'] = p(s'| s,a)
        self.transition_prob = np.zeros((self.grid.size, len(Action.all_actions()), self.grid.size), dtype=float)
        # 初始化奖励 rewards[s][a][s'] = r_sas' 该案例中，r是determine的，因此可以直接用rewards[s][a][s']存储奖励，而不是存储奖励的概率。
        # 如果s'对应的r是stochastic的，则应该表示为reward_prob[s][a][s'][r] = p(s',r|s,a)
        self.rewards = np.zeros((self.grid.size, len(Action.all_actions()), self.grid.size), dtype=float)
        for state_id in range(self.grid.size):
            for ind, action in enumerate(Action.all_actions()):
                next_state, reward = self.step_determine(state_id, action)
                self.transition_prob[state_id, ind, next_state] = 1
                self.rewards[state_id, ind, next_state] = reward
        # 初始化策略
        self.policy = np.zeros((self.grid.size, len(Action.all_actions())), dtype=int)
        for state_id in range(self.grid.size):
            # 随机选择一个动作的概率为1
            # action_ind = np.random.choice(len(Action.all_actions()))
            # self.policy[state_id, action_ind] = 1
            self.policy[state_id, 1] = 1

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
        # 禁行区域
        elif self.grid[new_row, new_col] == -1:
            reward = self.forbidden_reward
            # new_row, new_col = row, col
        # 终止状态
        elif new_row == self.end[0] and new_col == self.end[1]:
            reward = self.target_reward
        # 普通空格
        else:
            reward = 0

        return self.state2State_ind(np.array((new_row, new_col))), reward
