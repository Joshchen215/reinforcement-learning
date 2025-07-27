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
        for _ in range(max_iter):
            v = env.state_values
            # 1. 策略更新：计算Q值
            # q_Π(s,a) = Σ_(s',r) p(s',r|s,a)r + γ * Σ_s' [P(s'|s,a)*V(s')]
            q = np.sum(env.rewards, axis=2) + discount_factor * (env.transition_prob @ env.state_values)

            # 2. 策略更新：Π_{k+1}(a|s) = argmax_a Q_k(s,a)
            new_policy = np.argmax(q, axis=1)
            env.policy = np.zeros((env.grid.size, len(Action.all_actions())), dtype=float)
            env.policy[np.arange(env.grid.size), new_policy] = 1
            # 更新V值
            env.state_values = np.max(q, axis=1)

            # 检查收敛
            if np.linalg.norm(env.state_values - v) < eps:
                break
