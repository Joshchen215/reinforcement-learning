from enum import Enum, auto


class Action(Enum):
    """定义网格世界中的五个基本动作"""
    UP = (-1, 0)
    RIGHT = (0, 1)
    DOWN = (1, 0)
    LEFT = (0, -1)
    # STAY = (0, 0)

    @staticmethod
    def all_actions():
        """返回所有动作列表"""
        return list(Action)
