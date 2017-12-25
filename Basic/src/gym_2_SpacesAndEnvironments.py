import gym

# Spaces
env = gym.make('CartPole-v0')
print(env.action_space)  # Discrete(2) (left:0, right:1)
print(env.observation_space)  # Box(4,)
print(env.observation_space.high)
print(env.observation_space.low)

# Environments
envids = [spec.id for spec in gym.envs.registry.all()]
for envid in sorted(envids):
    print(envid)
