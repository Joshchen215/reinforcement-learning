from src.utils.Visualizer import Visualizer
from src.config.Parameter import Parameter
from src.core.Env import GridWorld

if __name__ == '__main__':
    env = GridWorld(grid=Parameter.init_grid, start=Parameter.init_start, end=Parameter.init_end,
                    target_reward=Parameter.init_target_reward, boundary_reward=Parameter.init_boundary_reward,
                    forbidden_reward=Parameter.init_forbidden_reward, random_start=Parameter.init_random_start)

    Visualizer.plot_env(env)
