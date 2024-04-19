import torch
import torch.nn as nn
import torch.nn.functional as F
from basicBlocks import CrossAttentionBlock

class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This is an abstract class.")

class BaseFusion(Fusion):
    """
    Fusion Modules that project both modalities into fusion dim at once
    """
    def __init__(self, dim1: int, dim2: int, fuse_dim: int, projector: nn.Module = nn.Linear) -> None:
        super().__init__()

        if projector == nn.Linear:
            self.projection1 = projector(dim1, fuse_dim)
            self.projection2 = projector(dim2, fuse_dim)
        elif projector in [nn.AdaptiveAvgPool1d, nn.AdaptiveMaxPool2d]:
            self.projection1 = projector(fuse_dim)
            self.projection2 = projector(fuse_dim)
        else:
            raise NotImplementedError("Currently only support projection to be: nn.Linear, nn.AdaptiveAvgPool1d or nn.AdaptiveMaxPool2d.")

class SummationFusion(BaseFusion):

    def __init__(self, dim1: int, dim2: int, fuse_dim: int, projector: nn.Module = nn.Linear) -> None:
        super().__init__(dim1, dim2, fuse_dim, projector)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        X1 = self.projection1(X1)
        X2 = self.projection2(X2)
        return X1 + X2

class ConcatFusion(BaseFusion):
    def __init__(self, dim1: int, dim2: int, fuse_dim: int, projector: nn.Module = nn.Linear) -> None:
        super().__init__(dim1, dim2, fuse_dim, projector)

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        X1 = self.projection1(X1)
        X2 = self.projection2(X2)
        return torch.concat((X1, X2), dim=-1)

class GatedFusion(BaseFusion):
    def __init__(self, dim1: int, dim2: int, fuse_dim: int, projector: nn.Module = nn.Linear) -> None:
        super().__init__(dim1, dim2, fuse_dim, projector)
        # Define gates for video and audio features
        self.gate1 = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Linear(fuse_dim, fuse_dim),
            nn.Sigmoid()
        )

    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> torch.Tensor:
        # Project features to a common dimension
        X1_projected = self.projection1(X1)
        X2_projected = self.projection2(X2)

        # Apply gating mechanism
        X1_weighted = X1_projected * self.gate1(X2_projected)
        X2_weighted = X2_projected * self.gate2(X1_projected)

        # Combine the features
        combined_features = X1_weighted + X2_weighted
        return combined_features

class LowRankFusion(Fusion):
    def __init__(self, dim1: int, dim2: int, rank: int, num_classes: int):
        super(LowRankFusion, self).__init__()
        self.rank = rank
        self.video_proj = nn.ParameterList([nn.Parameter(torch.randn(dim1, rank)) for _ in range(rank)])
        self.audio_proj = nn.ParameterList([nn.Parameter(torch.randn(rank, dim2)) for _ in range(rank)])

        self.fc = nn.Sequential(
            nn.Linear(rank, num_classes),
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        # Initialize fusion tensor
        fusion = X1.new_empty(X1.size(0), self.rank)
        # Perform low-rank fusion
        for r in range(self.rank):
            x1_proj = X1 @ self.video_proj[r]
            x2_proj = self.audio_proj[r] @ X2.t()
            fusion += x1_proj * x2_proj.t()
        # Pass through final classifier
        out = self.fc(fusion)
        return out
    
class TensorFusion(BaseFusion):
    def __init__(self, dim1: int, dim2: int, rank: int, num_classes: int, projector: nn.Module = nn.Linear):
        super().__init__(dim1, dim2, rank, projector)
        self.rank = rank
        self.num_classes = num_classes

        # Fusion layer: Since the outer product can be very large, we use a learnable weight matrix
        # to reduce dimensions. This is a simplification of the full outer product approach for efficiency.
        self.fusion_fc = nn.Sequential(
            nn.Linear(rank * rank, 8192),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8192),
            nn.Linear(8192, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 51),
        )

    def forward(self, X1: torch.Tensor, X2: torch.Tensor):
        # Project video and audio features to a lower dimension
        X1 = F.relu(self.projection1(X1))
        X2 = F.relu(self.projection2(X2))

        # Compute outer product and flatten to a vector
        # The outer product here is simplified by reshaping and using matrix multiplication
        fusion_tensor = torch.bmm(X1.unsqueeze(2), X2.unsqueeze(1))
        fusion_vector = fusion_tensor.view(-1, self.rank * self.rank)

        # Use a learnable weight matrix to reduce the dimensionality of the fused vector
        # and to predict the final class scores
        output = self.fusion_fc(fusion_vector)

        return output

class CrossModalAttn(BaseFusion):

    def __init__(self, dim1: int, dim2: int, fuse_dim: int, num_heads: int, dropout: int, mode: int, projector: nn.Module = nn.Linear) -> None:
        super().__init__(dim1, dim2, fuse_dim, projector)

        if mode not in [0, 1, 2]:
            raise NotImplementedError("Only support a mode in [0, 1, 2]:\n"
                                      "Mode 0: return only modality 1 (Visual) conditioned on modality 2 (Audio)\n"
                                      "Mode 1: return only modality 2 (Audio) conditioned on modality 1 (Visual)\n"
                                      "Mode 2: return both"
                                      )
        self.mode = mode

        if mode != 0:
            # will need av cross attention
            self.cross_attn_10 = CrossAttentionBlock(fuse_dim, num_heads, dropout)
        if mode != 1:
            # will need va cross attention
            self.cross_attn_01 = CrossAttentionBlock(fuse_dim, num_heads, dropout)
        
    def forward(self, X1: torch.Tensor, X2: torch.Tensor) -> tuple:
        # project both modalities into common token size
        X1 = self.projection1(X1)
        X2 = self.projection2(X2)

        # perform cross modaliti attention
        to_return = [0, 0]
        if self.mode != 0:
            features_10 = self.cross_attn_10(X2, X1)
            to_return[1] = features_10
        if self.mode != 1:
            features_01 = self.cross_attn_01(X1, X2)
            to_return[0] = features_01
        
        return tuple(to_return)