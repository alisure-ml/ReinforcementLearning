import gym

# env = gym.make("CartPole-v0")
# env = gym.make("MountainCar-v0")
# env = gym.make("FrozenLake-v0")
# env = gym.make("AirRaid-v0")
# env = gym.make("Alien-v0")
# env = gym.make("Amidar-v0")
# env = gym.make("Ant-v1")
# env = gym.make("Assault-v0")
# env = gym.make("Asterix-v0")
# env = gym.make("Atlantis-v0")
# env = gym.make("BankHeist-v0")
# env = gym.make("BattleZone-v0")
# env = gym.make("BeamRider-v0")
env = gym.make("Berzerk-v0")




env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

    pass
