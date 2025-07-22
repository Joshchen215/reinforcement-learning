import numpy as np

from src.core.Enum.Action import Action
from src.core.Env import GridWorld


class ValueIteration:
    @staticmethod
    def optimize(discount_factor, eps, max_iter, env: GridWorld):
        """
        值迭代算法
        :param discount_factor: 折扣因子
        :param eps: 收敛阈值
        :param max_iter: 最大迭代次数
        :param env: gridWorld环境
        :return:
        """
        # 初始化状态值
        v = np.zeros(env.grid.size)
        q = np.zeros((env.grid.size, 5))  # q(s,a)
        for _ in range(max_iter):
            # 1. 策略更新：计算Q值
            # 遍历状态
            for state_id in range(env.grid.size):
                # 遍历动作
                for ind, action in enumerate(Action.all_actions()):
                    # 计算Q值
                    # Q(s,a) = Σp(r|s,a)r + γ * Σ[P(s'|s,a)*V(s')]
                    # 在该案例中，可以简化为 Q(s,a) = r + γ * V(s'_{s,a})
                    next_state, reward = env.step_determine(state_id,action)
                    q[state_id, ind] = reward + discount_factor * v[next_state]

            # 2. 策略更新：Π_{k+1}(a|s) = argmax_a Q_k(s,a)
            # 更新V值
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
