import torch
from transformers import AutoTokenizer


class VQATokenizer:
    """Wraps a HuggingFace tokenizer with VQA-specific helpers."""

    def __init__(
        self,
        tokenizer_name: str = "vinai/phobert-base",
        max_question_length: int = 64,
        max_answer_length: int = 64,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    def tokenize_questions(
        self,
        questions: list[str],
        device: torch.device,
    ):
        """
        Convert question strings to PhoBERT token IDs.

        Returns:
            input_ids:      (B, Lq)
        """

        encoding = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )

        return encoding["input_ids"].to(device)

    def tokenize_answers(
        self,
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

        encoding = self.tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=self.max_answer_length,
            return_tensors="pt",
        )

        answer_ids = encoding["input_ids"].to(device)

        answer_input_ids = answer_ids[:, :-1]
        answer_target_ids = answer_ids[:, 1:]

        return answer_input_ids, answer_target_ids

    def tokenize_question(
        self,
        question: str,
        device: torch.device,
    ):
        """
        Convert a single question string into token IDs.

        Shape:
            (1, sequence_length)
        """

        encoding = self.tokenizer(
            [question],
            padding=True,
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt",
        )

        question_input_ids = encoding["input_ids"].to(device)

        return question_input_ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
