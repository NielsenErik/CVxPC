import torch

in_C = 3
out_C = 32
B = 16
H = 128
W = 128

x = torch.randn(B, in_C, H, W)
w = torch.randn(out_C, in_C)

log_w = torch.log_softmax(w, dim=1)
out = torch.logsumexp(log_w.view(1, out_C, in_C, 1, 1) + x.unsqueeze(1), dim=2)
print(out.shape)
