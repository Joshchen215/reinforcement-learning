import numpy as np

from src.config.Parameter import Parameter
from src.core.Enum.Action import Action
from src.core.Env import GridWorld


class MCBasic:
    @staticmethod
    def optimize(discount_factor, eps, max_iter, env: GridWorld):
        """
        Monte Carlo Basic算法
        :param discount_factor: 折扣因子
        :param eps: 收敛阈值
        :param max_iter: 最大迭代次数
        :param env: gridWorld环境
        :return:
        """
        for _ in range(max_iter):
            q = np.zeros((env.grid.size, len(Action.all_actions())), dtype=float)
            for state_id in range(env.grid.size):
                for ind, action in enumerate(Action.all_actions()):
                    """
                    生成足够多的从s,a开始的回合
                    该案例是确定性的，多次运行将得到相同的回合，因此只生成一个回合即可。(回合长度设置为15)
                    q_Π(s,a) = E[G_t|s,a]
                    G_t = R_t+1 + γ * R_t+2 + γ^2 * R_t+3 + ...
                    """
                    g = 0
                    current_state = state_id
                    next_state, reward = env.step_determine(current_state, action)
                    g += reward
                    for i in range(1, Parameter.episode_length):
                        current_state = next_state
                        next_action_id = np.argmax(env.policy[current_state])
                        next_state, reward = env.step_determine(current_state, Action.all_actions()[next_action_id])
                        g += reward * (discount_factor ** i)
                    # 策略评价(计算Q值)
                    q[state_id, ind] = g

                # 2. 策略更新：Π_{k+1}(a|s) = argmax_a Q_k(s,a)
                new_policy = np.argmax(q, axis=1)
                env.policy = np.zeros((env.grid.size, len(Action.all_actions())), dtype=float)
                env.policy[np.arange(env.grid.size), new_policy] = 1
            # 检查收敛
            # if np.linalg.norm(env.state_values - v) < eps:
            #     break
