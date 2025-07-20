import numpy as np

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
        v = np.zeros((env.get_state_num(),))
        pass
