import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym

from DDPG import DDPG
from utils import plotLearning

MAX_EPISODES = 100
MAX_TIMESTEPS = 50
NUM_UPDATES = 20
RENDER = True
TRAIN_VANILLA = False
TRAIN_HER = False
TEST = True

if __name__ == "__main__":
	if TRAIN_VANILLA:
		# Train vanilla DDPG
		env = gym.make('FetchPickAndPlace-v1')
		env.reset()
		
		input_shape = env.observation_space['observation'].shape[0]
		num_actions = env.action_space.shape[0]
		agent = DDPG(input_shape, num_actions, env.action_space)
		
		print(env.reset())
		
		score_history = []
		for i in range(MAX_EPISODES):
			env_dict = env.reset()
			observation = env_dict['observation']
			achieved_goal = env_dict['achieved_goal']
			desired_goal  = env_dict['desired_goal']
			done = False
			score = 0
			
			while not done:
				action = agent.select_action(observation)
				new_env_dict, reward, done, info = env.step(action)
				
				new_state = new_env_dict['observation']
				
				agent.save_transition(observation, action, reward, new_state, int(done))
				
				agent.learn()
				score += reward
				observation = new_state
				
				if RENDER:
					env.render()
					
				score_history.append(score)	

			print('episode ', i, 'score %.2f' % score,
				'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
		
	
		env.close()
	
		filename = 'FetchPickAndPlace1.png'
		plotLearning(score_history, filename, window=100)

	elif TRAIN_HER:
		# TRAIN DDPG+HER
		env_her = gym.make('FetchPickAndPlace-v1')
		env_her.reset()

		input_shape = env_her.observation_space['observation'].shape[0]
		num_actions = env_her.action_space.shape[0]
		agent_her = DDPG(input_shape, num_actions, env_her.action_space)

		print(env_her.reset())
  
		print(env_her.observation_space['desired_goal'].low[0])
	
		score_history_her = []
		for i in range(MAX_EPISODES):
			env_dict = env_her.reset()
			observation = env_dict['observation']
			achieved_goal = env_dict['achieved_goal']
			desired_goal  = env_dict['desired_goal']
			done = False
			score = 0

			while not done:
				action = agent_her.select_action(observation)
				new_env_dict, reward, done, info = env_her.step(action)

				new_state = new_env_dict['observation']
				
				agent_her.save_transition(observation, action, reward, new_state, int(done))

				# If we want, we can substitute a goal here and re-compute
				# the reward. For instance, we can just pretend that the desired
				# goal was what we achieved all along.
    
				# Final strategy
				if done:
					substitute_goal = new_env_dict['achieved_goal'].copy()
				
					substitute_reward = env_her.compute_reward(new_env_dict['achieved_goal'], substitute_goal, info)
					# print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))

					# extra_goals = sample_goals(i, trajectory)
					# for extra_goal in extra_goals:
					#     extra_reward = compute_reward(observation, action, extra_goal)
					agent_her.save_transition(observation, action, substitute_reward, new_state, int(done))
						

				agent_her.learn()
				score += reward
				observation = new_state

				if RENDER:
					env_her.render()
			
			
			score_history_her.append(score)

			print('episode ', i, 'score %.2f' % score,
						'trailing 100 games avg %.3f' % np.mean(score_history_her[-100:]))
			

		env_her.close()
			
		filename_her = 'FetchPickAndPlace1HER.png'
		plotLearning(score_history_her, filename_her, window=100)

		agent_her.save_models()
	
	elif TEST:
		# TEST
		env_test = gym.make('FetchPickAndPlace-v1')
		env_test.reset()

		input_shape = env_test.observation_space['observation'].shape[0]
		num_actions = env_test.action_space.shape[0]
		agent_test = DDPG(input_shape, num_actions, env_test.action_space)
		agent_test.load_models()
		agent_test.nn_eval()
  
		print(env_test.reset())
	
		score_history_test = []
		for _ in range(MAX_EPISODES):
			env_dict = env_test.reset()
			observation = env_dict['observation']
			achieved_goal = env_dict['achieved_goal']
			desired_goal  = env_dict['desired_goal']
			done = False
			score = 0

			while not done:
				action = agent_test.select_action(observation)
				new_env_dict, reward, done, info = env_test.step(action)
				new_state = new_env_dict['observation']
    			
				score += reward
				observation = new_state
    
				if RENDER:
					env_test.render()

			score_history_test.append(score)
   
		mean = np.mean(score_history_test)
		variance = np.var(score_history_test)
		print("Score (on 100 episodes): {} +/- {}".format(mean, variance))