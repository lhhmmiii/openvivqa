import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image

from src.data.transforms import get_image_transform
from src.models import build_model
from src.tokenization.vqa_tokenizer import VQATokenizer
from src.utils.config import load_config
from src.utils.checkpoint import load_checkpoint


def load_model(cfg, vqa_tokenizer, device):
    """
    Load model from checkpoint.
    """

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
        pretrained_backbone=False,
    )

    checkpoint = load_checkpoint(
        path=cfg["inference"]["checkpoint_path"],
        model=model,
        device=device,
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


@torch.no_grad()
def generate_answer(
    model,
    image,
    question_input_ids,
    device,
    vqa_tokenizer,
    max_answer_length,
):
    """
    Autoregressively generate an answer.

    Starts with BOS token and generates one token
    at a time until EOS or max_answer_length.
    """

    # Start decoder with BOS
    answer_input_ids = torch.tensor(
        [[vqa_tokenizer.bos_token_id]],
        dtype=torch.long,
        device=device,
    )

    generated_tokens = []

    for _ in range(max_answer_length):

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
        if next_token_id == vqa_tokenizer.eos_token_id:
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
    answer = vqa_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


def predict(
    model,
    image_path,
    question,
    device,
    vqa_tokenizer,
    image_transform,
    max_answer_length,
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

    question_input_ids = vqa_tokenizer.tokenize_question(
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
        vqa_tokenizer=vqa_tokenizer,
        max_answer_length=max_answer_length,
    )

    return answer


def main():
    parser = argparse.ArgumentParser(description="Run VQA inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cnn_lstm.yml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="datasets/test/test-images/000000000005.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Biển ghi gì? Gợi ý có chữ Kem",
        help="Question to ask about the image",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    vqa_tokenizer = VQATokenizer(
        tokenizer_name=cfg["data"]["tokenizer_name"],
        max_question_length=cfg["data"]["max_question_length"],
        max_answer_length=cfg["inference"]["max_answer_length"],
    )

    # --------------------------------------------------------
    # Image transform
    # --------------------------------------------------------

    image_transform = get_image_transform(cfg["data"]["image_size"])

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model(cfg, vqa_tokenizer, device)

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    answer = predict(
        model=model,
        image_path=args.image,
        question=args.question,
        device=device,
        vqa_tokenizer=vqa_tokenizer,
        image_transform=image_transform,
        max_answer_length=cfg["inference"]["max_answer_length"],
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()
    print("=" * 50)
    print(f"Image:    {args.image}")
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    print("=" * 50)


if __name__ == "__main__":
    main()
