import numpy as np

from src.config.Parameter import Parameter
from src.core.Enum.Action import Action
from src.core.Env import GridWorld


class CalStateValue:
    @staticmethod
    def calSV(env: GridWorld, max_iter=Parameter.max_iter_SV, gamma=Parameter.discount_factor_SV):
        current_v = env.state_values.copy()  # 当前状态值
        next_v = None  # 新状态值
        """
        矩阵向量形式：
        v_k+1 = r_Π + γ * P_Π · v_k
        展开形式：
        ┌ v_k+1(s1)┐ = ┌ Σ_a Π(a|,s1) Σ_r P(r|s1,a)r ┐     ┌ p_Π(s1|s1) p_Π(s2|s1) ... p_Π(sn|s1)┐ ┌ v_k(s1) ┐
        | v_k+1(s2)| = | Σ_a Π(a|,s2) Σ_r P(r|s2,a)r |     | p_Π(s1|s2) p_Π(s2|s2) ... p_Π(sn|s2)| | v_k(s2) |
        | v_k+1(s3)| = | Σ_a Π(a|,s3) Σ_r P(r|s3,a)r | + γ | p_Π(s1|s3) p_Π(s2|s3) ... p_Π(sn|s3)| | v_k(s3) |
        |  ......  | = |            ......           |     |          ...          ...     ...   | |   ...   |
        └ v_k+1(sn)┘ = └ Σ_a Π(a|,sn) Σ_r P(r|sn,a)r ┘     └ p_Π(s1|sn) p_Π(s2|sn) ... p_Π(sn|sn)┘ └ v_k(sn) ┘
        """
        # 计算r for循环实现
        r = np.zeros(env.grid.size)
        # for state_id in range(env.grid.size):
        #     for ind, action in enumerate(Action.all_actions()):
        #         print(f"policy[{state_id}, {ind}]={env.policy[state_id, ind]}")
        #         for state_id_next in range(env.grid.size):
        #             print(f"rewards[{state_id},{ind},{state_id_next}]={env.rewards[state_id, ind, state_id_next]}")
        #             r[state_id] += env.policy[state_id, ind] * env.rewards[state_id, ind, state_id_next]
        # 计算r numpy向量化实现
        r = np.sum(env.policy * np.sum(env.rewards, axis=2), axis=1)

        # 计算P状态转移概率矩阵 for循环实现
        # p(s'|s) = Σ_a Π(a|s) P(s'|s,a)
        # P = np.zeros((env.grid.size, env.grid.size))
        # for state_id in range(env.grid.size):
        #     for ind, action in enumerate(Action.all_actions()):
        #         for state_id_next in range(env.grid.size):
        #             P[state_id, state_id_next] += env.policy[state_id, ind] * env.transition_prob[state_id, ind, state_id_next]
        # 计算P状态转移概率矩阵 numpy向量化实现
        P = np.sum(env.policy[:, :, None] * env.transition_prob, axis=1)

        # 开始迭代计算v
        for _ in range(max_iter):
            # 计算v_k+1
            next_v = r + gamma * P @ current_v
            # 更新状态值
            current_v = next_v
        env.state_values = next_v
