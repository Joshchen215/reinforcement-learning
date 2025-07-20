from enum import Enum, auto


class Action(Enum):
    """定义网格世界中的四个基本动作"""
    UP = auto()
    RIGHT = auto()
    DOWN = auto()
    LEFT = auto()

    @staticmethod
    def all_actions():
        """返回所有动作列表"""
        return list(Action)
