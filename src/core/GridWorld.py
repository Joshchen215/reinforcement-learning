import matplotlib.pyplot as plt
import numpy as np


class GridWorld:
    """网格世界环境，支持值迭代和策略迭代算法"""

    def __init__(self, reward_wall=-5):
        """
        初始化网格世界环境
        Args:
            reward_wall: 撞墙奖励值，默认为-5
        """
        # 定义7x8网格矩阵：
        # >0: 目标状态（终点）
        # -1: 墙壁/障碍物
        # 0: 非终止状态
        self._grid = np.array(
            [[0, 0, 0, 0, 0, -1, 0, 0],
             [0, 0, 0, -1, 0, 0, 0, 5],  # 右下角有奖励5的目标状态
             [0, 0, 0, -1, -1, 0, 0, 0],
             [0, 0, 0, -1, -1, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]
             ])
        # 为网格添加边界墙（用-1填充边界）
        self._grid_padded = np.pad(self._grid, pad_width=1, mode='constant', constant_values=-1)
        self._reward_wall = reward_wall  # 撞墙惩罚值

        # 初始状态设置
        self._start_state = (1, 1)  # 网格内坐标(1,1)为起点
        self._random_start = False  # 是否随机起点

        # 识别目标状态和非终止状态坐标
        idx_goal_state_y, idx_goal_state_x = np.nonzero(self._grid > 0)
        self._goal_states = [(idx_goal_state_y[i], idx_goal_state_x[i]) for i in range(len(idx_goal_state_x))]

        idx_non_term_y, idx_non_term_x = np.nonzero(self._grid == 0)
        self._non_term_states = [(idx_non_term_y[i], idx_non_term_x[i]) for i in range(len(idx_non_term_x))]

        # 当前状态在带边界网格中的坐标（初始为起点）
        self._state_padded = (self._start_state[0] + 1, self._start_state[1] + 1)

    def get_state_num(self):
        """返回网格总状态数（包含障碍物）"""
        return np.prod(np.shape(self._grid))

    def get_state_grid(self):
        """生成状态编号网格（用于状态转换）"""
        # 创建状态编号矩阵，障碍物位置标记为-1
        state_grid = np.multiply(np.reshape(np.arange(self.get_state_num()), self._grid.shape), self._grid >= 0) - (
                self._grid == -1)
        # 添加边界
        padded_state_grid = np.pad(state_grid, pad_width=1, mode='constant', constant_values=-1)
        return state_grid, padded_state_grid

    def get_current_state(self):
        """获取当前状态的整数编号(0到总状态数-1)"""
        y, x = self._state_padded
        return (y - 1) * self._grid.shape[1] + (x - 1)

    def int_to_state(self, int_obs):
        """将整数状态编号转换为网格坐标(y,x)"""
        x = int_obs % self._grid.shape[1]  # 计算列坐标
        y = int_obs // self._grid.shape[1]  # 计算行坐标
        return y, x

    def reset(self):
        """重置环境到初始状态"""
        if self._random_start:
            # 随机选择非终止状态作为起点
            idx_start = np.random.randint(len(self._non_term_states))
            start_state = self._non_term_states[idx_start]
            self._state_padded = (start_state[0] + 1, start_state[1] + 1)
        else:
            # 使用预设起点
            self._state_padded = (self._start_state[0] + 1, self._start_state[1] + 1)

    def step(self, action):
        """
        执行动作并返回结果
        Args:
            action: 整数(0=上,1=右,2=下,3=左)
        Returns:
            reward: 奖励值
            terminated: 是否终止(1=终止,0=继续)
            next_state: 新状态编号
        """
        y, x = self._state_padded  # 当前带边界坐标

        # 计算新位置
        if action == 0:  # 上
            new_state_padded = (y - 1, x)
        elif action == 1:  # 右
            new_state_padded = (y, x + 1)
        elif action == 2:  # 下
            new_state_padded = (y + 1, x)
        elif action == 3:  # 左
            new_state_padded = (y, x - 1)
        else:
            raise ValueError("无效动作: {} (应为0-3)".format(action))

        new_y, new_x = new_state_padded

        # 判断新位置类型
        if self._grid_padded[new_y, new_x] == -1:  # 撞墙
            reward = self._reward_wall
            new_state_padded = (y, x)  # 保持原位
        elif self._grid_padded[new_y, new_x] == 0:  # 普通格子
            reward = 0.
        else:  # 到达目标
            reward = self._grid_padded[new_y, new_x]
            self.reset()  # 重置环境
            terminated = 1
            return reward, terminated, self.get_current_state()

        terminated = 0
        self._state_padded = new_state_padded
        return reward, terminated, self.get_current_state()

    def transition(self, action):
        """
        计算状态转移矩阵和奖励
        Args:
            action: 要执行的动作(0-3)
        Returns:
            reward: 奖励向量(总状态数,)
            probability: 转移概率矩阵(总状态数, 总状态数)
        """
        # 根据动作确定锚点位置
        if action == 0:  # 上
            anchor_state_padded = (0, 1)
        elif action == 1:  # 右
            anchor_state_padded = (1, 2)
        elif action == 2:  # 下
            anchor_state_padded = (2, 1)
        elif action == 3:  # 左
            anchor_state_padded = (1, 0)
        else:
            raise ValueError("无效动作: {} (应为0-3)".format(action))

        state_num = self.get_state_num()
        h, w = self._grid.shape
        y_a, x_a = anchor_state_padded

        # 计算奖励矩阵（仅非目标状态有奖励）
        reward = np.multiply(self._grid_padded[y_a:y_a + h, x_a:x_a + w], self._grid == 0)

        # 计算状态转移
        state_grid, state_grid_padded = self.get_state_grid()
        next_state = state_grid_padded[y_a:y_a + h, x_a:x_a + w]
        next_state = np.multiply(state_grid, next_state == -1) + np.multiply(next_state, next_state > -1)

        # 处理特殊状态（墙和目标）
        next_state[self._grid == -1] = -1
        next_state[self._grid > 0] = state_grid[self._grid > 0]

        # 展平为向量
        next_state_vec = next_state.flatten()
        state_vec = state_grid.flatten()

        # 构建转移概率矩阵（确定性环境）
        probability = np.zeros((state_num, state_num))
        valid_states = state_vec > -1
        probability[state_vec[valid_states], next_state_vec[valid_states]] = 1

        return reward.flatten(), probability

    def value_iteration(self, gamma, eps=1e-5, max_iter=2000):
        """
        值迭代算法[1,5](@ref)
        Args:
            gamma: 折扣因子(0-1)
            eps: 收敛阈值
            max_iter: 最大迭代次数
        Returns:
            optim_value: 最优状态值函数
            optim_policy: 最优策略
        """
        # 初始化值函数
        v = np.zeros((self.get_state_num(),))

        for _ in range(max_iter):
            # 1. 策略更新：计算Q值
            q = np.zeros((self.get_state_num(), 4))  # q(s,a)
            for action in range(4):
                reward_vec, tran_prob = self.transition(action)
                # Q(s,a) = R(s,a) + γ * Σ[P(s'|s,a)*V(s')]
                q[:, action] = reward_vec + gamma * np.matmul(tran_prob, v)

            # 2. 值更新：取最大Q值作为新V值
            v_tmp = np.max(q, axis=1)

            # 检查收敛
            if np.linalg.norm(v_tmp - v) < eps:
                break
            else:
                v = v_tmp

        # 计算最优策略
        optim_value = v
        optim_policy = np.argmax(q, axis=1)

        return optim_value, optim_policy

    def policy_iteration(self, gamma=0.9, max_it=1000, tol=1e-5):
        """
        策略迭代算法[1,5](@ref)
        Args:
            gamma: 折扣因子(0-1)
            max_it: 最大迭代次数
            tol: 收敛阈值
        Returns:
            optimal_v: 最优状态值函数
            optimal_policy: 最优策略
        """
        # 初始化随机策略
        stochastic_mat = np.random.rand(self.get_state_num(), 4)
        pi = stochastic_mat / stochastic_mat.sum(axis=1)[:, None]  # π(a|s)
        policy = np.argmax(pi, axis=1)

        for _ in range(max_it):
            # 1. 策略评估：计算当前策略的状态值
            v = np.zeros((self.get_state_num(),))
            for _ in range(max_it):
                value_temp = np.zeros((self.get_state_num(),))
                for action in range(4):
                    reward_vec, tran_prob = self.transition(action)
                    # V(s) = Σ[π(a|s) * (R(s,a) + γ * Σ[P(s'|s,a)*V(s')])]
                    value_temp += pi[:, action] * (reward_vec + gamma * np.matmul(tran_prob, v))

                # 检查收敛
                if np.linalg.norm(value_temp - v) < tol:
                    break
                else:
                    v = value_temp
            v_final = v

            # 2. 策略改进：计算新策略
            q = np.zeros((self.get_state_num(), 4))
            for action in range(4):
                reward_vec, tran_prob = self.transition(action)
                q[:, action] = reward_vec + gamma * np.matmul(tran_prob, v_final)
            now_policy = np.argmax(q, axis=1)

            # 检查策略是否稳定
            if np.array_equal(policy, now_policy):
                optimal_policy = policy
                optimal_v = v_final
                break
            else:
                policy = now_policy
                # 更新为贪婪策略
                pi = np.zeros((self.get_state_num(), 4))
                pi[np.arange(self.get_state_num()), policy] = 1

        return optimal_v, optimal_policy