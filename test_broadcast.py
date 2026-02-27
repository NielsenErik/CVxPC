import torch

N_children = 2
N_pixels = 1639
stack = torch.randn(N_children, N_pixels)
weights = torch.randn(N_children)

w_shape = [-1] + [1] * (stack.dim() - 1)
log_w = torch.log_softmax(weights, dim=0).view(*w_shape)

print(log_w.shape)
print((stack + log_w).shape)
