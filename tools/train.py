import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.data import get_dataloader
from src.data.transforms import get_image_transform
from src.models import build_model
from src.tokenization.vqa_tokenizer import VQATokenizer
from src.utils.config import load_config
from src.utils.checkpoint import save_checkpoint


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    vqa_tokenizer,
):
    model.train()

    total_loss = 0.0

    for batch in train_loader:
        images = batch["image"].to(device)
        questions = batch["question"]  # list[str]
        answers = batch["answer"]      # list[str]

        # Tokenize questions and answers
        question_input_ids = vqa_tokenizer.tokenize_questions(questions, device)
        answer_input_ids, answer_target_ids = vqa_tokenizer.tokenize_answers(answers, device)
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


@torch.no_grad()
def validate(
    model,
    val_loader,
    criterion,
    device,
    vqa_tokenizer,
):
    model.eval()

    total_loss = 0.0

    for batch in val_loader:

        images = batch["image"].to(device)
        questions = batch["question"]
        answers = batch["answer"]

        # Tokenize
        question_input_ids = vqa_tokenizer.tokenize_questions(questions, device)
        answer_input_ids, answer_target_ids = vqa_tokenizer.tokenize_answers(answers, device)

        # Forward
        logits = model(images, question_input_ids, answer_input_ids)

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            answer_target_ids.reshape(-1),
        )

        total_loss += loss.item()

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train a VQA model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_lstm.yml",
        help="Path to the YAML config file",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # ========================================================
    # Tokenizer
    # ========================================================

    vqa_tokenizer = VQATokenizer(
        tokenizer_name=cfg["data"]["tokenizer_name"],
        max_question_length=cfg["data"]["max_question_length"],
        max_answer_length=cfg["data"]["max_answer_length"],
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    image_transform = get_image_transform(cfg["data"]["image_size"])

    train_loader = get_dataloader(
        json_path=cfg["data"]["train_json"],
        image_dir=cfg["data"]["train_image_dir"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        transform=image_transform,
        max_samples=cfg["training"].get("max_samples"),
        num_workers=cfg["training"]["num_workers"],
        is_test=False,
    )
    val_loader = get_dataloader(
        json_path=cfg["data"]["val_json"],
        image_dir=cfg["data"]["val_image_dir"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        transform=image_transform,
        max_samples=cfg["training"].get("max_samples"),
        num_workers=cfg["training"]["num_workers"],
        is_test=False,
    )

    # ========================================================
    # Model
    # ========================================================

    model_cfg = cfg["model"]
    model = build_model(
        model_cfg["name"],
        vocab_size=vqa_tokenizer.vocab_size,
        embedding_dim=model_cfg["embedding_dim"],
        text_embedding_dim=model_cfg["text_embedding_dim"],
        text_hidden_dim=model_cfg["text_hidden_dim"],
        text_num_layers=model_cfg["text_num_layers"],
        text_dropout=model_cfg["text_dropout"],
        fusion_hidden_dim=model_cfg["fusion_hidden_dim"],
        fusion_output_dim=model_cfg["fusion_output_dim"],
        fusion_dropout=model_cfg["fusion_dropout"],
        decoder_token_embedding_dim=model_cfg["decoder_token_embedding_dim"],
        decoder_hidden_dim=model_cfg["decoder_hidden_dim"],
        decoder_num_layers=model_cfg["decoder_num_layers"],
        decoder_dropout=model_cfg["decoder_dropout"],
        pretrained_backbone=model_cfg["pretrained_backbone"],
    )

    model = model.to(device)

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # ========================================================
    # Loss function
    # ========================================================
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=vqa_tokenizer.pad_token_id,
    )

    # ========================================================
    # TensorBoard
    # ========================================================
    writer = SummaryWriter(log_dir=cfg["training"]["log_dir"])

    # ========================================================
    # Checkpoints
    # ========================================================

    checkpoint_dir = Path(cfg["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # ========================================================
    # Training loop
    # ========================================================

    num_epochs = cfg["training"]["num_epochs"]

    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            vqa_tokenizer=vqa_tokenizer,
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar(
            "Learning_Rate",
            optimizer.param_groups[0]["lr"],
            epoch,
        )

        val_loss = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            vqa_tokenizer=vqa_tokenizer,
        )

        writer.add_scalar("Loss/val", val_loss, epoch)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f}"
        )

        # ====================================================
        # Save best model
        # ====================================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                path=checkpoint_dir / "best.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
            )

            print("Saved best checkpoint.")


if __name__ == "__main__":
    main()
