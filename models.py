from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)
import torch.nn as nn
import torch

def resnet(layers: int) -> nn.Module:
    if layers == 18:
        return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1), 512
    if layers == 34:
        return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1), 512
    if layers == 50:
        return resnet34(weights=ResNet50_Weights.IMAGENET1K_V1), 2048
    if layers == 101:
        return resnet34(weights=ResNet101_Weights.IMAGENET1K_V1), 2048
    if layers == 152:
        return resnet34(weights=ResNet152_Weights.IMAGENET1K_V1), 2048
    
    raise NotImplementedError("Layers number must be one of <18, 34, 50, 101, 152>")

class AudioResNet(nn.Module):

    def __init__(self, layers: int, freeze: bool, num_classes: int, batchnorm: bool) -> None:
        super(AudioResNet, self).__init__()

        self.resnet, features = resnet(layers)
        self.resnet.fc = nn.Identity()

        self.freeze = freeze

        hidden_size = (features + num_classes) // 2
        self.fc = nn.Sequential(
            nn.Linear(features, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d() if batchnorm else nn.Identity(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            with torch.no_grad():
                X = self.resnet(X)
        else:
            X = self.resnet(X)
        
        return self.fc(X)
    
    def save(self, path: str) -> None:
        state_dict = self.state_dict()
        torch.save(state_dict, path)
    
    def load(self, path: str) -> None:
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)