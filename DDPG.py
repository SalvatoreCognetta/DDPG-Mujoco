import numpy as np
import torch
import torch.nn as nn

from ActorCritic import Actor, Critic
from ReplayBuffer import ReplayBuffer
from OUNoise import OUNoise
from Normalizer import Normalizer

# Check if the GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DDPG(object):
	def __init__(self, input_shape, num_actions, action_space,lr_actor=1e-4, lr_critic=1e-3, l2_critic=1e-2, 
			   gamma=0.99, tau=1e-3, hidden_shape=([400,300]), batch_size=64, rbuffer_size=10**6, HER = True, isOUNoise=False, noiseEps=0.2, rndEps=0.3):
		self.gamma = gamma
		self.tau = tau
		self.input_shape = input_shape
		self.batch_size = batch_size
		self.action_space = action_space
		self.num_actions = num_actions
		self.isOUNoise = isOUNoise
		self.noise_eps = noiseEps
		self.random_eps = rndEps	

		# Intialize critic network Q(s,a|theta^Q) and target network Q'
		self.critic = Critic(self.input_shape, hidden_shape, self.num_actions, lr_critic, l2_critic, is_her=HER)
		self.critic_target = Critic(self.input_shape, hidden_shape, self.num_actions, lr_critic, l2_critic, is_her=HER)
		
		# Intialize actor network mu(s|theta^mu) and target network mu'
		self.actor = Actor(self.input_shape, hidden_shape, self.num_actions, lr_actor, is_her=HER)
		self.actor_target = Actor(self.input_shape, hidden_shape, self.num_actions, lr_actor, is_her=HER)

		# Make sure targets have the same weight
		self.update_target_params(self.critic, self.critic_target, 1)
		self.update_target_params(self.actor, self.actor_target, 1)

		# Initialize replay buffer
		self.HER = HER
		self.replay_buffer = ReplayBuffer(rbuffer_size, input_shape, self.num_actions, is_her = HER)

		self.noise = OUNoise(mu=np.zeros(self.num_actions))

		# create the normalizer
		# self.state_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
		# self.goal_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)


	def select_action(self, state, goal=None):
		# evaluate the model (turn off batch norm)
		self.actor.eval()

		if self.HER:
			state = torch.tensor([np.concatenate([state, goal])], dtype=torch.float).to(DEVICE)
			state = state.view(-1)
		else:
			state = torch.tensor(state, dtype=torch.float).to(DEVICE)

		mu = self.actor(state).to(DEVICE)

		if self.isOUNoise == True:
			mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(DEVICE)
			# Clip the output according to the action space of the env
			mu_prime = mu_prime.clamp(self.action_space.low[0], self.action_space.high[0])
			mu_prime.cpu().detach().numpy() # We cannot pass a tensor to openai gym
		else:
			"""
			The authors of the original DDPG paper recommended time-correlated OU noise, but more recent results suggest that uncorrelated, 
			mean-zero Gaussian noise works perfectly well.
			"""
			# add the gaussian
			mu_prime = mu + self.noise_eps * self.action_space.high[0] * np.random.randn(self.num_actions)
			mu_prime = np.clip(mu_prime, self.action_space.low[0], self.action_space.high[0])
			# random actions...
			random_actions = np.random.uniform(low=self.action_space.low[0], high=self.action_space.high[0], \
												size=self.num_actions)
			# choose if use the random actions
			mu_prime += np.random.binomial(1, self.random_eps, 1)[0] * (random_actions - mu_prime)

		self.actor.train()

		return mu_prime 


	def save_transition(self, state, action, reward, new_state, done, goal=None, achieved_goal=None):
		if self.HER:
			self.replay_buffer.store_transition(state, action, reward, new_state, done, goal, achieved_goal)
		else:
			self.replay_buffer.store_transition(state, action, reward, new_state, done)


	def learn(self):
		if self.replay_buffer.mem_cntr < self.batch_size: 
			return
		
		# Sample random minibatch of N=batch_size transitions (s_i, a_i, r_i, s_(i+1)) from the replay buffer
		if self.HER:
			state, action, reward, new_state, done, goal, achieved_goal = self.replay_buffer.sample_buffer(self.batch_size)
		else:
			state, action, reward, new_state, done = self.replay_buffer.sample_buffer(self.batch_size)

		# state = self.state_norm.normalize(state)
		# goal = self.goal_norm.normalize(goal)
		# new_state = self.state_norm.normalize(new_state)
        

		state = torch.tensor(state, dtype=torch.float).to(DEVICE)
		new_state = torch.tensor(new_state, dtype=torch.float).to(DEVICE)
		if self.HER:
			# print("new_state: ",new_state.shape)
			goal = torch.tensor(goal, dtype=torch.float).to(DEVICE)
			state = torch.cat((state, goal), 1)
			new_state = torch.cat((new_state, goal), 1)
			# print("new_state cat: ",new_state.shape)

		action = torch.tensor(action, dtype=torch.float).to(DEVICE)
		reward = torch.tensor(reward, dtype=torch.float).to(DEVICE)
		done = torch.tensor(done, dtype=torch.float).to(DEVICE)

		self.nn_eval()

		# Compute mu'(s_(i+1)|theta^mu')
		mu_prime = self.actor_target(new_state)
		# Compute Q'(s_(i+1),mu'(s_(i+1)|theta^mu')|theta^Q')
		Q_prime = self.critic_target(new_state, mu_prime)

		# Set y_i = r_i + gamma*Q'(s_(i+1),mu'(s_(i+1)|theta^mu')|theta^Q')
		y = []
		for i in range(self.batch_size):
			y.append(reward[i] + self.gamma*Q_prime[i] * done[i]) # We don't keep new rewarding after terminal state
		target = torch.tensor(y).to(DEVICE)
		target = target.view(self.batch_size, 1) # Reshape the tensor

		# Update the critic by minimizing the loss
		self.critic.train()
		Q = self.critic(state, action) # Compute Q(s_i,a_i|theta^Q)
		value_loss = nn.functional.mse_loss(target, Q) # L
		self.critic.optimizer.zero_grad() # zero_grad clears old gradients from the last step
		value_loss.backward() # computes the derivative of the loss wrt the parameters using backpropagation
		self.critic.optimizer.step() # causes the optimizer to take a step based on the gradients of the parameters

		# Update the actor policy using the sampled policy gradient
		self.critic.eval()
		mu = self.actor(state)
		self.actor.train()
		policy_loss = -self.critic(state, mu)
		policy_loss = torch.mean(policy_loss)
		self.actor.optimizer.zero_grad()
		policy_loss.backward()
		self.actor.optimizer.step()

		self.update_target_params(self.critic, self.critic_target)
		self.update_target_params(self.actor, self.actor_target)

	def update_target_params(self, source, target, tau=None):
		if tau == None:
			# Perform soft update, otherwise hard update
			tau = self.tau
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data*tau + (1-tau)*target_param.data)

	def nn_eval(self):
		self.critic.eval()
		self.critic_target.eval()
		self.actor.eval()
		self.actor_target.eval()
		
	def save_models(self):
		self.actor.save_model()
		self.actor_target.save_model()
		self.critic.save_model()
		self.critic_target.save_model()

	def load_models(self):
		self.actor.load_model()
		self.actor_target.load_model()
		self.critic.load_model()
		self.critic_target.load_model()
