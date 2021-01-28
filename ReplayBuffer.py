import numpy as np

class ReplayBuffer(object):
  def __init__(self, max_size, input_shape, num_actions):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.state_mem = np.zeros((self.mem_size, input_shape)) 
    self.new_state_mem = np.zeros((self.mem_size, input_shape))
    self.action_mem = np.zeros((self.mem_size, num_actions))
    self.reward_mem = np.zeros(self.mem_size)
    self.terminal_mem = np.zeros(self.mem_size)

  def store_transition(self, state, action, reward, state_, done):
    index = self.mem_cntr % self.mem_size #first available position
    self.state_mem[index] = state
    self.new_state_mem[index] = state_
    self.reward_mem[index] = reward
    self.action_mem[index] = action
    self.terminal_mem[index] = 1 - int(done)
    self.mem_cntr += 1

  def sample_buffer(self, batch_size):
    max_mem = min(self.mem_cntr, self.mem_size)
    batch = np.random.choice(max_mem, batch_size)

    states = self.state_mem[batch]
    new_states = self.new_state_mem[batch]
    actions = self.action_mem[batch]
    rewards = self.reward_mem[batch]
    terminal = self.terminal_mem[batch]

    return states, actions, rewards, new_states, terminal