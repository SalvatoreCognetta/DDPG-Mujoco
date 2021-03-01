import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime
import time
from copy import deepcopy
from mpi4py import MPI

import gym

from DDPG import DDPG
from utils import plotLearning

"""
the MPI part code is taken from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)
"""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV_NAME = 'FetchPickAndPlace-v1'
MAX_EPOCHES = 200
N_CYCLES = 50
MAX_EPISODES = 16
NUM_UPDATES = 40
SEED = 123
RENDER = False
TRAIN_VANILLA = False
TRAIN_HER = True
TEST = True

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'

if __name__ == "__main__":
	if TRAIN_VANILLA:
		# Train vanilla DDPG
		env = gym.make(ENV_NAME)
		env.seed(123)
		env.reset()

		if ENV_NAME == 'FetchPickAndPlace-v1':
			input_shape = env.observation_space['observation'].shape[0]
			num_goals = env.observation_space.spaces['desired_goal'].shape[0]
		else:
			input_shape = env.observation_space.shape[0]
			num_goals = 1

		num_actions = env.action_space.shape[0]
		agent = DDPG(input_shape, num_actions, env.action_space, num_goals, HER = False, isOUNoise=True)
				
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
	
		filename = './output/'+ENV_NAME+'.png'
		plotLearning(score_history, filename, window=100)
		agent.save_models()

	elif TRAIN_HER:
		# TRAIN DDPG+HER
		env_her = gym.make('FetchPickAndPlace-v1')
		env_her.seed(SEED + MPI.COMM_WORLD.Get_rank())
		random.seed(SEED + MPI.COMM_WORLD.Get_rank())
		np.random.seed(SEED + MPI.COMM_WORLD.Get_rank())
		torch.manual_seed(SEED + MPI.COMM_WORLD.Get_rank())
		if DEVICE == 'cuda':
			torch.cuda.manual_seed(SEED + MPI.COMM_WORLD.Get_rank())
   
		env_her.reset()

		input_shape = env_her.observation_space['observation'].shape[0]
		num_actions = env_her.action_space.shape[0]
		num_goals = env_her.observation_space.spaces['desired_goal'].shape[0]
		agent_her = DDPG(input_shape, num_actions, env_her.action_space, num_goals, HER = True, isOUNoise=False)

	
		score_history_her = []
		for i in range(MAX_EPOCHES):
			start_time = time.time()
			epoch_actor_loss = 0
			epoch_critic_loss = 0
			for _ in range(N_CYCLES):
				mini_batch = []
				cycle_actor_loss = 0
				cycle_critic_loss = 0

				for episode in range(MAX_EPISODES):
					episode_dict = {
						"state": [],
						"action": [],
						"reward": [],
						"done": [],
						"new_state": [],
						"desired_goal": [],
						"achieved_goal": [],
						"new_achieved_goal": []}

					env_dict = env_her.reset()

					state = env_dict['observation']
					achieved_goal = env_dict['achieved_goal']
					desired_goal  = env_dict['desired_goal']

					done = False
					score = 0

					# Reset the environment if the cube is on the goal at start
					while np.linalg.norm(achieved_goal - desired_goal) <= 0.05:
						env_dict = env_her.reset()
						state = env_dict["observation"]
						achieved_goal = env_dict["achieved_goal"]
						desired_goal = env_dict["desired_goal"]

					while not done:
						if RENDER:
							env_her.render()

						# Choose an action
						with torch.no_grad():
							action = agent_her.select_action(state, desired_goal)
						# Perform the action
						new_env_dict, reward, done, info = env_her.step(action)

						new_state = new_env_dict['observation']
						new_achieved_goal = new_env_dict['achieved_goal']

						episode_dict['state'].append(state.copy())
						episode_dict['action'].append(action.copy())
						episode_dict['reward'].append(reward.copy())
						episode_dict['done'].append(done)
						episode_dict['achieved_goal'].append(achieved_goal.copy())
						episode_dict['desired_goal'].append(desired_goal.copy())
						
						# agent_her.save_transition(state, action, reward, new_state, int(done), desired_goal, achieved_goal)

						# # If we want, we can substitute a goal here and re-compute
						# # the reward. For instance, we can just pretend that the desired
						# # goal was what we achieved all along.
			
						# # Final strategy
						# if done:
						# 	substitute_goal = new_env_dict['achieved_goal'].copy()
						
						# 	substitute_reward = env_her.compute_reward(new_env_dict['achieved_goal'], substitute_goal, info)
						# 	print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))

						# 	# extra_goals = sample_goals(i, trajectory)
						# 	# for extra_goal in extra_goals:
						# 	#     extra_reward = compute_reward(state, action, extra_goal)
						# 	agent_her.save_transition(state, action, substitute_reward, new_state, int(done), substitute_goal, achieved_goal)
						# 	# reward = substitute_reward
								

						# agent_her.learn()
						score += reward

						# Upload observation
						state = new_state
						achieved_goal = new_achieved_goal
						desired_goal = new_env_dict['desired_goal']

						if done:
							# Final strategy
							substitute_goal = new_env_dict['achieved_goal'].copy()
							substitute_reward = env_her.compute_reward(new_env_dict['achieved_goal'], substitute_goal, info)
							episode_dict['state'].append(state.copy())
							episode_dict['action'].append(action.copy())
							episode_dict['reward'].append(substitute_reward.copy())
							episode_dict['done'].append(done)
							episode_dict['achieved_goal'].append(substitute_goal.copy())
							episode_dict['desired_goal'].append(substitute_goal.copy())

							episode_dict['achieved_goal'].append(achieved_goal.copy())
							episode_dict['desired_goal'].append(desired_goal.copy())
							episode_dict['new_state'] = episode_dict['state'][1:]
							episode_dict['new_achieved_goal'] = episode_dict['achieved_goal'][1:]

					mini_batch.append(deepcopy(episode_dict))
					
				agent_her.save_transitions(mini_batch)
				for n in range(NUM_UPDATES):
					actor_loss, critic_loss = agent_her.learn()
					cycle_actor_loss += actor_loss
					cycle_critic_loss += critic_loss

				epoch_actor_loss += cycle_actor_loss/NUM_UPDATES
				epoch_critic_loss += cycle_critic_loss/NUM_UPDATES
				
			score_history_her.append(score)
			
			
			if MPI.COMM_WORLD.Get_rank() == 0:
				print(f"[{datetime.now()}] |Epoch:{i}| "
					f"Duration:{time.time() - start_time:.3f}| "
					f"Reward:{score:.3f}|"
					f"trailing 100 games avg:{np.mean(score_history_her[-100:]):.3f}")				

		env_her.close()

		
		if MPI.COMM_WORLD.Get_rank() == 0:	
			filename_her = './output/'+ENV_NAME+'_HER.png'
			plotLearning(score_history_her, filename_her, window=100)
			agent_her.save_models()
	
	elif TEST:
		# TEST
		env_test = gym.make(ENV_NAME)

		env_test.seed(123)
		env_test.reset()

		if ENV_NAME == 'FetchPickAndPlace-v1':
			input_shape = env_test.observation_space['observation'].shape[0]
			num_goals = env_test.observation_space.spaces['desired_goal'].shape[0]
		else:
			input_shape = env_test.observation_space.shape[0]
			num_goals = 1

		num_actions = env_test.action_space.shape[0]

		agent_test = DDPG(input_shape, num_actions, env_test.action_space, num_goals, HER=False, isOUNoise=True)
		agent_test.load_models()
		agent_test.nn_eval()
  
	
		score_history_test = []
		for i in range(MAX_EPOCHES):
			env_dict = env_test.reset()
			
			if ENV_NAME == 'FetchPickAndPlace-v1':
				observation = env_dict['observation']
				achieved_goal = env_dict['achieved_goal']
				desired_goal  = env_dict['desired_goal']
			else:
				observation = env_dict
			
			done = False
			score = 0

			while not done:
				action = agent_test.select_action(observation)
				new_env_dict, reward, done, info = env_test.step(action)

				if ENV_NAME == 'FetchPickAndPlace-v1':
					new_state = new_env_dict['observation']
				else:
					new_state = new_env_dict
				
				score += reward
				observation = new_state
	
				env_test.render()

			score_history_test.append(score)
			print('episode ', i, 'score %.2f' % score,
				'trailing 100 games avg %.3f' % np.mean(score_history_test[-100:]))
