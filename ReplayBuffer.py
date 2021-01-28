import numpy as np

class ReplayBuffer(object):
	def __init__(self, max_size, input_shape, num_actions, is_her = True):
		self.mem_size = max_size
		self.is_her = is_her
		self.mem_cntr = 0
		
		self.state_mem = np.zeros((self.mem_size, input_shape), dtype=np.float32) 
		self.new_state_mem = np.zeros((self.mem_size, input_shape), dtype=np.float32)
		self.action_mem = np.zeros((self.mem_size, num_actions), dtype=np.float32)
		self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
		self.terminal_mem = np.zeros(self.mem_size, dtype=bool)
		if is_her:
			self.goal_mem = np.zeros((self.mem_size, 3), dtype=np.float32)

	def store_transition(self, state, action, reward, state_, done, goal=None):
		index = self.mem_cntr % self.mem_size #first available position
		
		self.state_mem[index] = state
		self.new_state_mem[index] = state_
		self.reward_mem[index] = reward
		self.action_mem[index] = action
		self.terminal_mem[index] = 1 - int(done)
		if self.is_her:
			self.goal_mem[index] = goal
		
		self.mem_cntr += 1

	def sample_buffer(self, batch_size):
		max_mem = min(self.mem_cntr, self.mem_size)
		batch = np.random.choice(max_mem, batch_size)

		states = self.state_mem[batch]
		new_states = self.new_state_mem[batch]
		actions = self.action_mem[batch]
		rewards = self.reward_mem[batch]
		terminal = self.terminal_mem[batch]
		if self.is_her:
			goals = self.goal_mem[batch]

		if self.is_her:
			return states, actions, rewards, new_states, terminal, goals
		else:
			return states, actions, rewards, new_states, terminal
