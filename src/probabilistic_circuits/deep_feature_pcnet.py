import torch
import torch.nn as nn
# from probabilistic_circuits.probabilistic_circuits import SpatialPCNet

class DeepFeaturePCNet(nn.Module):
    """
    Latent Density Estimation: Uses a frozen deep feature extractor (e.g., GAN Encoder) 
    to extract robust spatial features (z), and learns their distribution using a SpatialPCNet.
    """
    def __init__(self, feature_extractor, spatial_mixtures=10, spatial_depth=3, channels_per_layer=32):
        super().__init__()
        self.feature_extractor = feature_extractor
        
        # Freeze the feature extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
            
        self.spatial_mixtures = spatial_mixtures
        self.spatial_depth = spatial_depth
        self.channels_per_layer = channels_per_layer
        
        # Will be initialized in init_network once we know the latent channel dimension
        self.spatial_pc = None

    def init_network(self, x):
        pass

    def forward(self, x):
        pass

class RandomDeepFeaturePCEnsemble(nn.Module):
    """
    Ensemble of DeepFeaturePCNets, sharing a single feature extractor.
    """
    def __init__(self, feature_extractor, n_models=10, device="cpu", **model_params):
        super().__init__()
        self.n_models = n_models
        self.target_device = device
        self.model_params = model_params
        
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        
        self.models = nn.ModuleList()

    def init_network(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(self.target_device)
        
        n_mixtures = self.model_params.get("n_mixtures", 10)
        depth = self.model_params.get("max_depth", 3)
        channels_per_layer = self.model_params.get("channels_per_layer", 32)
        
        for i in range(self.n_models):
            # Pass the same feature extractor reference, but it's frozen anyway
            model = DeepFeaturePCNet(
                feature_extractor=self.feature_extractor,
                spatial_mixtures=n_mixtures,
                spatial_depth=depth,
                channels_per_layer=channels_per_layer
            ).to(self.target_device)
            
            model.init_network(X)
            self.models.append(model)
            
        self.to(self.target_device)

    def forward(self, x):
        # We only need to extract features ONCE for the whole ensemble!
        with torch.no_grad():
            self.feature_extractor.eval()
            z = self.feature_extractor(x)
            
        # 1. Get log probabilities from each model's SpatialPCNet acting on z
        log_probs_list = [m.spatial_pc(z) for m in self.models]
        
        # 2. Stack: (N_Models, B, 1, H', W')
        stack = torch.stack(log_probs_list, dim=0)
        
        import numpy as np
        # 3. Ensemble Averaging via Log-Sum-Exp
        return torch.logsumexp(stack, dim=0) - np.log(self.n_models)
