import torch
from src.probabilistic_circuits.deep_feature_pcnet import DeepFeaturePCNet, RandomDeepFeaturePCEnsemble
import torchvision.models as models
import torch.nn as nn

print("Building feature extractor...")
resnet = models.resnet18(pretrained=True)
# To keep spatial dimensions, we use layers up to layer2
extractor = nn.Sequential(*list(resnet.children())[:6])

print("Mocking Batch...")
# Batch of 4 images, 3 channels, 128x128 resolution
x = torch.randn(4, 3, 128, 128)

print("Forward Extract...")
z = extractor(x)
print("Latent Feature Shape:", z.shape) # Expecting (4, 128, 16, 16)

print("Initializing DeepFeaturePCNet...")
model = DeepFeaturePCNet(feature_extractor=extractor)
model.init_network(x)

print("Testing PCNet Forward...")
log_prob = model(x)
print("Output Log-Prob Shape:", log_prob.shape)

if log_prob.shape == (4, 1, z.shape[2], z.shape[3]):
    print("SUCCESS: Dimensions match!")
else:
    print("FAILED: Dimensions do not match!")
    print(f"Expected: (4, 1, {z.shape[2]}, {z.shape[3]})")
    
print("\\nTesting Ensemble Forward...")
ensemble = RandomDeepFeaturePCEnsemble(feature_extractor=extractor, n_models=3)
ensemble.init_network(x)
ens_log_prob = ensemble(x)
print("Ensemble Output Log-Prob Shape:", ens_log_prob.shape)
if ens_log_prob.shape == (4, 1, z.shape[2], z.shape[3]):
    print("SUCCESS: Ensemble Dimensions match!")
else:
    print("FAILED: Ensemble Dimensions do not match!")
