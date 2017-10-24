import gym
import numpy as np

env = gym.make("CartPole-v0")

env.reset()
random_episodes = 0
reward_sum = 0

while random_episodes < 100:
    env.render()
    observation, reward, done, _ = env.step(np.random.randint(0, 1))
    print(observation)
    reward_sum += reward
    if done:
        random_episodes += 1
        print("reward for this episode was: {}".format(reward_sum))
        reward_sum = 0
        env.reset()
    pass
