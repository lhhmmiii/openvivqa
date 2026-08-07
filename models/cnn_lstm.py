import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ImageExtractor(nn.Module):
    def __init__(
        self,
        backbone_out_dim: int = 2048,
        hidden_dim: int = 1024,
        embedding_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        # Remove the classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(backbone_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.backbone(x)

        # Flatten from (N, C, 1, 1) -> (N, C)
        x = torch.flatten(x, 1)

        # Project to embedding space
        x = self.projector(x)

        # L2 normalize embeddings
        x = nn.functional.normalize(x, dim=1)

        return x
    
if __name__ == "__main__":
    # Example usage
    model = ImageExtractor(embedding_dim=512, pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print(output.shape)  # Should print torch.Size([1, 512])