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
        # (B, 3, H, W) -> (B, C, 1, 1)
        x = self.backbone(x)

        # Flatten from (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)

        # (B, C) -> (B, embedding_dim)
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
        dropout: float = 0.5,
    ):
        super().__init__()

        # Token embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

        # LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Head to project the hidden state to the desired output dimension
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len)
               containing token IDs.
        """

        # (B, L) -> (B, L, C) with B = batch size, L = sequence length, C = embedding dimension
        x = self.embedding(x)

        # outputs: (B, L, H)
        # hidden: (num_layers, B, hidden_dim)
        _, (hidden, _) = self.lstm(x)

        features = hidden[-1]
        features = self.projector(features)
        features = F.normalize(features, dim=1)

        return features

class Fusion(nn.Module):
    def __init__(
        self,
        image_feature_dim: int = 512,
        text_feature_dim: int = 512,
        hidden_dim: int = 512,
        output_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(image_feature_dim + text_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_features: (batch_size, image_feature_dim)
            text_features:  (batch_size, text_feature_dim)

        Returns:
            fused_features: (batch_size, output_dim)
        """
        # (B, image_feature_dim) + (B, text_feature_dim) -> (B, image_feature_dim + text_feature_dim)
        x = torch.cat([image_features, text_features], dim=1)
        # (B, image_feature_dim + text_feature_dim) -> (B, output_dim)
        x = self.fusion(x)
        return x

class LSTMDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 512,
        token_embedding_dim: int = 300,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embedding_dim,
        )

        self.init_hidden = nn.Linear(feature_dim, hidden_dim)
        self.init_cell = nn.Linear(feature_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=token_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        fused_features: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fused_features: (B, feature_dim)
            answer_tokens:  (B, seq_len)

        Returns:
            logits: (B, seq_len, vocab_size)
        """
        # (B, seq_len) -> (B, seq_len, embedding_dim)
        embeddings = self.embedding(answer_tokens)

        # (B, feature_dim) -> (1, B, hidden_dim)
        h0 = self.init_hidden(fused_features).unsqueeze(0)
        c0 = self.init_cell(fused_features).unsqueeze(0)

        # (B, seq_len, embedding_dim) -> (B, seq_len, hidden_dim)
        outputs, _ = self.lstm(
            embeddings,
            (h0, c0),
        )

        # (B, seq_len, hidden_dim) -> (B, seq_len, vocab_size)
        logits = self.classifier(outputs)

        return logits

class CNN_LSTM(nn.Module):
    """
    End-to-end CNN-LSTM model for Visual Question Answering.

    Pipeline:
        Image  -> ImageExtractor  -> image_features  (B, embed_dim)
        Question -> TextExtractor -> text_features    (B, embed_dim)
        [image_features, text_features] -> Fusion     -> fused (B, fusion_dim)
        fused + answer_tokens -> LSTMDecoder          -> logits (B, L, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 512,
        text_embedding_dim: int = 300,
        text_hidden_dim: int = 512,
        text_num_layers: int = 2,
        text_dropout: float = 0.5,
        fusion_hidden_dim: int = 512,
        fusion_output_dim: int = 512,
        fusion_dropout: float = 0.2,
        decoder_token_embedding_dim: int = 300,
        decoder_hidden_dim: int = 512,
        decoder_num_layers: int = 1,
        decoder_dropout: float = 0.2,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size

        # --- Sub-modules ---
        self.image_extractor = ImageExtractor(
            embedding_dim=embedding_dim,
            pretrained=pretrained_backbone,
        )

        self.text_extractor = TextExtractor(
            vocab_size=vocab_size,
            embedding_dim=text_embedding_dim,
            hidden_dim=text_hidden_dim,
            num_layers=text_num_layers,
            output_dim=embedding_dim,
            dropout=text_dropout,
        )

        self.fusion = Fusion(
            image_feature_dim=embedding_dim,
            text_feature_dim=embedding_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout=fusion_dropout,
        )

        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            feature_dim=fusion_output_dim,
            token_embedding_dim=decoder_token_embedding_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_num_layers,
            dropout=decoder_dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor,
        answer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher-forcing forward pass (used during training).

        Args:
            images:          (B, 3, H, W)
            question_tokens: (B, Lq)  - padded question token IDs
            answer_tokens:   (B, La)  - padded answer token IDs (shifted input)

        Returns:
            logits: (B, La, vocab_size)
        """
        # (B, 3, H, W) -> (B, embedding_dim)
        image_features = self.image_extractor(images)

        # (B, Lq) -> (B, embedding_dim)
        text_features = self.text_extractor(question_tokens)

        # (B, embedding_dim) + (B, embedding_dim) -> (B, fusion_output_dim)
        fused = self.fusion(image_features, text_features)

        # (B, fusion_output_dim) + (B, La) -> (B, La, vocab_size)
        logits = self.decoder(fused, answer_tokens)
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        question_tokens: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_len: int = 50,
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation (used during inference).

        Args:
            images:          (B, 3, H, W)
            question_tokens: (B, Lq)
            bos_token_id:    beginning-of-sequence token id
            eos_token_id:    end-of-sequence token id
            max_len:         maximum answer length

        Returns:
            generated_ids: (B, T) - token IDs of the generated answer
        """
        self.eval()

        image_features = self.image_extractor(images)
        text_features = self.text_extractor(question_tokens)
        fused = self.fusion(image_features, text_features)

        B = images.size(0)
        device = images.device

        # Initialize hidden state from fused features
        h = self.decoder.init_hidden(fused).unsqueeze(0)  # (1, B, H)
        c = self.decoder.init_cell(fused).unsqueeze(0)    # (1, B, H)

        # Start with BOS token
        input_token = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)

        generated = [input_token]
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            emb = self.decoder.embedding(input_token)     # (B, 1, E)
            output, (h, c) = self.decoder.lstm(emb, (h, c))  # (B, 1, H)
            logit = self.decoder.classifier(output)       # (B, 1, V)

            next_token = logit.argmax(dim=-1)             # (B, 1)
            generated.append(next_token)

            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

            input_token = next_token

        return torch.cat(generated, dim=1)  # (B, T)
