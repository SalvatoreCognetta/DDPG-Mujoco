# Check if the GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
# Used for initialize final layer weights and biases of actor and critic
WEIGHT_FINAL_LAYER = 3e-3
BIAS_FINAL_LAYER   = 3e-4