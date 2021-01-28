import gym
env = gym.make('FetchPickAndPlace-v1')

env.reset()

print(*env.observation_space['observation'].shape)
print(env.action_space.low[0])
# for _ in range(100):
#     # env.render()
#     rndm_action = env.action_space.sample()
#     print(rndm_action)
#     env.step(rndm_action) # take a random action
env.close()
