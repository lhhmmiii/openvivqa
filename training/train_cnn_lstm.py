import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.optim as optim
from transformers import AutoTokenizer
import torchvision.transforms as T

from dataloader import get_dataloader
from models.cnn_lstm import CNN_LSTM

# Hyperparameters
TOKENIZER_NAME = "vinai/phobert-base"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_QUESTION_LENGTH = 64
MAX_ANSWER_LENGTH = 64

CHECKPOINT_DIR = Path("checkpoints") 

# ============================================================
# Tokenization
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

VOCAB_SIZE = len(tokenizer)

PAD_TOKEN_ID = tokenizer.pad_token_id
BOS_TOKEN_ID = tokenizer.bos_token_id
EOS_TOKEN_ID = tokenizer.eos_token_id

def tokenize_questions(
    questions: list[str],
    device: torch.device,
):
    """
    Convert question strings to PhoBERT token IDs.

    Returns:
        input_ids:      (B, Lq)
        attention_mask: (B, Lq)
    """

    encoding = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=MAX_QUESTION_LENGTH,
        return_tensors="pt",
    )

    return encoding["input_ids"].to(device)


def tokenize_answers(
    answers: list[str],
    device: torch.device,
):
    """
    Convert answer strings to token IDs.

    Example:

        Full answer:
            [BOS, The, dog, runs, EOS]

        Input:
            [BOS, The, dog, runs]

        Target:
            [The, dog, runs, EOS]
    """

    encoding = tokenizer(
        answers,
        padding=True,
        truncation=True,
        max_length=MAX_ANSWER_LENGTH,
        return_tensors="pt",
    )

    answer_ids = encoding["input_ids"].to(device)

    answer_input_ids = answer_ids[:, :-1]
    answer_target_ids = answer_ids[:, 1:]

    return answer_input_ids, answer_target_ids

# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        questions = batch["question"]  # list[str]
        answers = batch["answer"]      # list[str]

        # Tokenize questions and answers
        question_input_ids = tokenize_questions(questions, device)
        answer_input_ids, answer_target_ids = tokenize_answers(answers, device)
        # Forward
        logits = model(images, question_input_ids, answer_input_ids)

        # Compute loss 
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            answer_target_ids.reshape(-1),
        )

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    for batch in val_loader:

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

        # Tokenize
        question_input_ids = tokenize_questions(questions, device)
        answer_input_ids, answer_target_ids = tokenize_answers(answers, device)

        # Forward
        logits = model(images, question_input_ids, answer_input_ids)

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            answer_target_ids.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(val_loader)


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # Get dataloaders

    image_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    
    train_loader = get_dataloader(
        json_path="datasets/train/vlsp2023_train_data.json",
        image_dir="datasets/train/training-images",
        batch_size=BATCH_SIZE,
        shuffle=True,
        transform=image_transform,
        is_test=False,
    )
    val_loader = get_dataloader(
        json_path="datasets/dev/vlsp2023_dev_data.json",
        image_dir="datasets/dev/dev-images",
        batch_size=BATCH_SIZE,
        shuffle=False,
        transform=image_transform,
        is_test=False,
    )

    model = CNN_LSTM(
        vocab_size=VOCAB_SIZE,
        pretrained_backbone=True,
    ) 

    model = model.to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # ========================================================
    # Loss function
    # ========================================================
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN_ID,
    )

    # ========================================================
    # Checkpoints
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    # ========================================================
    # Training loop
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        # ====================================================
        # Save best model
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }

            torch.save(
                checkpoint,
                CHECKPOINT_DIR / "best.pt",
            )

            print("Saved best checkpoint.")


if __name__ == "__main__":
    main()