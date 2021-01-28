import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Check if the GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# Used for initialize final layer weights and biases of actor and critic
WEIGHT_FINAL_LAYER = 3e-3
BIAS_FINAL_LAYER   = 3e-4

class Actor(nn.Module):
  def __init__(self, input_shape, hidden_shape, num_actions, lr, filepath='./actor'):
    super(Actor, self).__init__()
    self.input_shape = input_shape
    self.hidden_shape = hidden_shape
    self.num_actions = num_actions
    self.filepath = filepath

    # Layer 1
    self.fc1 = nn.Linear(self.input_shape, hidden_shape[0])
    self.bn1 = nn.LayerNorm(hidden_shape[0])
    # Layer 2
    self.fc2 = nn.Linear(hidden_shape[0], hidden_shape[1])
    self.bn2 = nn.LayerNorm(hidden_shape[1])
    # Output Layer - mu
    self.mu = nn.Linear(hidden_shape[1], num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr)
    self.to(DEVICE)

  def init_weight_bias(self):
    # Weights and biases initialization
    fan_in_weight_1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
    fan_in_bias_1 = 1/np.sqrt(self.fc1.bias.data.size()[0])
    nn.init.uniform_(self.fc1.weight, -fan_in_weight_1, fan_in_weight_1)
    nn.init.uniform_(self.fc1.bias, -fan_in_bias_1, fan_in_bias_1)

    fan_in_weight_2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
    fan_in_bias_2 = 1/np.sqrt(self.fc2.bias.data.size()[0])
    nn.init.uniform_(self.fc2.weight, -fan_in_weight_2, fan_in_weight_2)
    nn.init.uniform_(self.fc2.bias, -fan_in_bias_2, fan_in_bias_2)

    nn.init.uniform_(self.mu.weight, -WEIGHT_FINAL_LAYER, WEIGHT_FINAL_LAYER)
    nn.init.uniform_(self.mu.bias, -BIAS_FINAL_LAYER, BIAS_FINAL_LAYER)

  def forward(self, state):
    x = self.fc1(state)
    x = self.bn1(x)
    x = nn.functional.relu(x)

    x = self.fc2(x)
    x = self.bn2(x)
    x = nn.functional.relu(x)

    # Output
    mu = torch.tanh(self.mu(x))
    return mu

  def save_model(self):
    torch.save(self.state_dict(), self.filepath)

  def load_model(self):
    self.load_state_dict(torch.load(self.filepath))

class Critic(nn.Module):
  def __init__(self, input_shape, hidden_shape, num_actions, lr, l2, filepath='./critic'):
    super(Critic, self).__init__()
    self.input_shape = input_shape
    self.hidden_shape = hidden_shape
    self.num_actions = num_actions
    self.filepath = filepath

    # Layer 1
    self.fc1 = nn.Linear(self.input_shape, self.hidden_shape[0])
    self.bn1 = nn.LayerNorm(self.hidden_shape[0])
    # Layer 2
    self.fc2 = nn.Linear(self.hidden_shape[0] + num_actions, self.hidden_shape[1])
    self.bn2 = nn.LayerNorm(self.hidden_shape[1])
    # Output layer - Q
    self.Q = nn.Linear(self.hidden_shape[1], 1)

    self.optimizer = optim.Adam(self.parameters(), lr, weight_decay=l2)
    self.to(DEVICE)

  def init_weight_bias(self):
    # Weights and biases initialization
    fan_in_weight_1 = 1/np.sqrt(self.fc1.weight.data.size()[0])
    fan_in_bias_1 = 1/np.sqrt(self.fc1.bias.data.size()[0])
    nn.init.uniform_(self.fc1.weight, -fan_in_weight_1, fan_in_weight_1)
    nn.init.uniform_(self.fc1.bias, -fan_in_bias_1, fan_in_bias_1)

    fan_in_weight_2 = 1/np.sqrt(self.fc2.weight.data.size()[0])
    fan_in_bias_2 = 1/np.sqrt(self.fc2.bias.data.size()[0])
    nn.init.uniform_(self.fc2.weight, -fan_in_weight_2, fan_in_weight_2)
    nn.init.uniform_(self.fc2.bias, -fan_in_bias_2, fan_in_bias_2)

    nn.init.uniform_(self.Q.weight, -WEIGHT_FINAL_LAYER, WEIGHT_FINAL_LAYER)
    nn.init.uniform_(self.Q.bias, -BIAS_FINAL_LAYER, BIAS_FINAL_LAYER)

  def forward(self, state, actions):
    x = self.fc1(state)
    x = self.bn1(x)
    x = nn.functional.relu(x)

    x = torch.cat((x, actions), 1)
    x = self.fc2(x)
    x = self.bn2(x)
    x = nn.functional.relu(x)

    # Output
    output = self.Q(x)
    return output

  def save_model(self):
    torch.save(self.state_dict(), self.filepath)

  def load_model(self):
    self.load_state_dict(torch.load(self.filepath))
