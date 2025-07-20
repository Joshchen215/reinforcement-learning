from src.core.GridWorld import GridWorld

if __name__ == '__main__':
    env = GridWorld()
    env.reset()

    env.plot_grid("GridWorld")
