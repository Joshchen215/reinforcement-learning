from src.algorithm.DP.CalStateValue import CalStateValue
from src.algorithm.DP.PolicyIteration import PolicyIteration
from src.algorithm.DP.TruncatedPolicyIteration import TruncatedPolicyIteration
from src.algorithm.DP.ValueIteration import ValueIteration
from src.algorithm.MC.MCBasic import MCBasic
from src.utils.Visualizer import Visualizer
from src.config.Parameter import Parameter
from src.core.Env import GridWorld

if __name__ == '__main__':
    env = GridWorld(grid=Parameter.init_grid, start=Parameter.init_start, end=Parameter.init_end,
                    target_reward=Parameter.init_target_reward, boundary_reward=Parameter.init_boundary_reward,
                    forbidden_reward=Parameter.init_forbidden_reward)
    # Visualizer.plot_env(env)

    # ValueIteration.optimize(Parameter.discount_factor, Parameter.eps, Parameter.max_iter, env)
    # PolicyIteration.optimize(Parameter.discount_factor, Parameter.eps, Parameter.max_iter, env)
    # TruncatedPolicyIteration.optimize(Parameter.discount_factor, Parameter.eps, Parameter.max_iter, max_iter_SV=5, env=env)  # max_iter_SV表示截断的次数
    MCBasic.optimize(Parameter.discount_factor, Parameter.eps, Parameter.max_iter, env)
    Visualizer.plot_state_values(env)
    Visualizer.plot_policy(env)
    
