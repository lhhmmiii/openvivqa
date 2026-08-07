import torch
import torch.nn as nn
import torch.nn.functional as F
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

class TextExtractor(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 2,
        output_dim: int = 512,
        bidirectional: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.projector = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
               containing token IDs.
        """

        # (B, L) -> (B, L, E)
        x = self.embedding(x)

        # outputs: (B, L, H)
        # hidden: (num_layers * num_directions, B, hidden_dim)
        _, (hidden, _) = self.lstm(x)

        if self.lstm.bidirectional:
            # Concatenate the last forward and backward hidden states
            features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            features = hidden[-1]

        features = self.projector(features)

        features = F.normalize(features, dim=1)

        return features

if __name__ == "__main__":
    # Example usage
    model = ImageExtractor(embedding_dim=512, pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels, 224x224 image
    output = model(dummy_input)
    print(output.shape)  # Should print torch.Size([1, 512])

    text_model = TextExtractor(vocab_size=10000, embedding_dim=300, hidden_dim=512, num_layers=2, output_dim=512)
    dummy_text_input = torch.randint(0, 10000, (1, 10))  # Batch size of 1, sequence length of 10
    text_output = text_model(dummy_text_input)
    print(text_output.shape)  # Should print torch.Size([1, 512])