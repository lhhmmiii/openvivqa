import sys
from pathlib import Path

# Allow importing modules from project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms as T

from PIL import Image
from transformers import AutoTokenizer

from models.cnn_lstm import CNN_LSTM


# ============================================================
# Configuration
# ============================================================

TOKENIZER_NAME = "vinai/phobert-base"

CHECKPOINT_PATH = "checkpoints/best.pt"

MAX_QUESTION_LENGTH = 64
MAX_ANSWER_LENGTH = 64


# ============================================================
# Load tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id


# ============================================================
# Image preprocessing
# ============================================================

image_transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# Load model
# ============================================================

def load_model(device):
    """
    Load CNN_LSTM model from checkpoint.
    """

    vocab_size = len(tokenizer)

    model = CNN_LSTM(
        vocab_size=vocab_size,
        pretrained_backbone=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation loss: "
        f"{checkpoint['val_loss']:.4f}"
    )

    return model


# ============================================================
# Tokenize question
# ============================================================

def tokenize_question(
    question: str,
    device: torch.device,
):
    """
    Convert question string into token IDs.

    Shape:
        (1, sequence_length)
    """

    encoding = tokenizer(
        [question],
        padding=True,
        truncation=True,
        max_length=MAX_QUESTION_LENGTH,
        return_tensors="pt",
    )

    question_input_ids = encoding["input_ids"].to(device)

    return question_input_ids


# ============================================================
# Generate answer
# ============================================================

@torch.no_grad()
def generate_answer(
    model,
    image,
    question_input_ids,
    device,
):
    """
    Autoregressively generate an answer.

    Starts with BOS token and generates one token
    at a time until EOS or MAX_ANSWER_LENGTH.
    """

    # Start decoder with BOS
    answer_input_ids = torch.tensor(
        [[BOS_TOKEN_ID]],
        dtype=torch.long,
        device=device,
    )

    generated_tokens = []

    for _ in range(MAX_ANSWER_LENGTH):

        # Forward pass
        logits = model(
            image,
            question_input_ids,
            answer_input_ids,
        )

        # logits:
        # (batch_size, sequence_length, vocab_size)

        # Take prediction for the last token
        next_token_logits = logits[:, -1, :]

        # Greedy decoding
        next_token_id = torch.argmax(
            next_token_logits,
            dim=-1,
        )

        next_token_id = next_token_id.item()

        # Stop when EOS is generated
        if next_token_id == EOS_TOKEN_ID:
            break

        generated_tokens.append(next_token_id)

        # Append predicted token to decoder input
        next_token = torch.tensor(
            [[next_token_id]],
            dtype=torch.long,
            device=device,
        )

        answer_input_ids = torch.cat(
            [
                answer_input_ids,
                next_token,
            ],
            dim=1,
        )

    # Convert token IDs back to text
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


# ============================================================
# Inference
# ============================================================

def predict(
    model,
    image_path: str,
    question: str,
    device,
):
    """
    Run VQA inference for one image + question.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    image = image_transform(image)

    # Add batch dimension
    # (C, H, W) -> (1, C, H, W)
    image = image.unsqueeze(0)

    image = image.to(device)

    # --------------------------------------------------------
    # Tokenize question
    # --------------------------------------------------------

    question_input_ids = tokenize_question(
        question,
        device,
    )

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    answer = generate_answer(
        model=model,
        image=image,
        question_input_ids=question_input_ids,
        device=device,
    )

    return answer


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(device)

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    image_path = "datasets/test/test-images/000000000005.jpg"

    question = "Biển ghi gì? Gợi ý có chữ Kem"

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    answer = predict(
        model=model,
        image_path=image_path,
        question=question,
        device=device,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(f"Image:    {image_path}")
    print(f"Question: {question}")
    print(f"Answer:   {answer}")
    print("=" * 50)


if __name__ == "__main__":
    main()