import gym

# env = gym.make("CartPole-v0")
# env = gym.make("MountainCar-v0")
env = gym.make("FrozenLake-v0")

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

    pass
