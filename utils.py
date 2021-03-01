import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import torch

def plotLearning(scores, filename, x=None, window=5):   
	N = len(scores)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
	if x is None:
		x = [i for i in range(N)]
	plt.ylabel('Score')       
	plt.xlabel('Game')                     
	plt.plot(x, running_avg)
	plt.savefig(filename)

# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
	"""
	include two kinds: grads and params

	"""
	attr = 'data' if mode == 'params' else 'grad'
	return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])

def _set_flat_params_or_grads(network, flat_params, mode='params'):
	"""
	include two kinds: grads and params

	"""
	attr = 'data' if mode == 'params' else 'grad'
	# the pointer
	pointer = 0
	for param in network.parameters():
		getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
		pointer += param.data.numel()
