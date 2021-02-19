import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym

from DDPG import DDPG
from utils.utils import plotLearning

ENV_NAME = 'FetchPickAndPlace-v1'
MAX_EPOCHES = 50
N_CYCLES = 50
MAX_TIMESTEPS = 50
NUM_UPDATES = 20
RENDER = True
TRAIN_VANILLA = False
TRAIN_HER = True
TEST = False

if __name__ == "__main__":
	if TRAIN_VANILLA:
		# Train vanilla DDPG
		env = gym.make(ENV_NAME)
		env.seed(123)
		env.reset()
		
		if ENV_NAME == 'FetchPickAndPlace-v1':
			input_shape = env.observation_space['observation'].shape[0]
		else:
			input_shape = env.observation_space.shape[0]

		num_actions = env.action_space.shape[0]
		agent = DDPG(input_shape, num_actions, env.action_space, HER = False)
		
		print(env.reset())
		
		score_history = []
		for i in range(MAX_EPOCHES):
			env_dict = env.reset()

			if ENV_NAME == 'FetchPickAndPlace-v1':
				observation = env_dict['observation']
				achieved_goal = env_dict['achieved_goal']
				desired_goal  = env_dict['desired_goal']
			else:
				observation = env_dict

			done = False
			score = 0
			
			while not done:
				with torch.no_grad():
					action = agent.select_action(observation)
				new_env_dict, reward, done, info = env.step(action)
				
				if ENV_NAME == 'FetchPickAndPlace-v1':
					new_state = new_env_dict['observation']
				else:
					new_state = new_env_dict

				agent.save_transition(observation, action, reward, new_state, int(done))
				
				agent.learn()
				score += reward
				observation = new_state
				
				if RENDER and i > 50:
					env.render()
					
			score_history.append(score)	

			print('episode ', i, 'score %.2f' % score,
				'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
		
	
		env.close()
	
		filename = './output/FetchPickAndPlace1.png'
		plotLearning(score_history, filename, window=100)

	elif TRAIN_HER:
		# TRAIN DDPG+HER
		env_her = gym.make('FetchPickAndPlace-v1')
		env_her.reset()

		input_shape = env_her.observation_space['observation'].shape[0]
		num_actions = env_her.action_space.shape[0]
		agent_her = DDPG(input_shape, num_actions, env_her.action_space)

		print(env_her.reset())
  
		print(env_her.observation_space['desired_goal'])
	
		score_history_her = []
		for i in range(MAX_EPOCHES):
			for _ in range(N_CYCLES):
				env_dict = env_her.reset()
				state = env_dict['observation']
				achieved_goal = env_dict['achieved_goal']
				desired_goal  = env_dict['desired_goal']
				done = False
				score = 0

				while not done:

					if RENDER:
						env_her.render()
					with torch.no_grad():
						action = agent_her.select_action(state, desired_goal)
					new_env_dict, reward, done, info = env_her.step(action)

					new_state = new_env_dict['observation']
					new_achieved_goal = new_env_dict['achieved_goal']
					
					agent_her.save_transition(state, action, reward, new_state, int(done), desired_goal, achieved_goal)

					# If we want, we can substitute a goal here and re-compute
					# the reward. For instance, we can just pretend that the desired
					# goal was what we achieved all along.
		
					# # Final strategy
					# if done:
					# 	substitute_goal = new_env_dict['achieved_goal'].copy()
					
					# 	substitute_reward = env_her.compute_reward(new_env_dict['achieved_goal'], substitute_goal, info)
					# 	# print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))

					# 	# extra_goals = sample_goals(i, trajectory)
					# 	# for extra_goal in extra_goals:
					# 	#     extra_reward = compute_reward(state, action, extra_goal)
					# 	agent_her.save_transition(state, action, substitute_reward, new_state, int(done), substitute_goal)
					# 	# reward = substitute_reward
							

					agent_her.learn()
					score += reward

					# Upload observation
					state = new_state
					achieved_goal = new_achieved_goal
					# desired_goal = new_env_dict['desired_goal']

				
				score_history_her.append(score)
				# print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), i, success_rate))

				print('episode ', i, 'score %.2f' % score,
							'trailing 100 games avg %.3f' % np.mean(score_history_her[-100:]))
				

		env_her.close()
			
		filename_her = './output/FetchPickAndPlace1HER.png'
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
		for _ in range(MAX_EPOCHES):
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