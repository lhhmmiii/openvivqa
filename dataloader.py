from typing import Any, Dict, List, Optional, Callable

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from dataset import OpenViVQADataset


def _collate_images(images: List[Any]):
    """Batch a list of images. Supports both Tensor (post-transform) and PIL.Image."""
    if len(images) == 0:
        return images
    if torch.is_tensor(images[0]):
        return torch.stack(images, dim=0)
    # images are still PIL.Image (no transform applied) -> return as list
    return list(images)


def _collate_text(texts: List[Any], pad_value: int = 0):
    """
    Batch a list of text fields.
    - If str -> return list[str] as-is (tokenize later, e.g. inside the
      training loop with a HuggingFace tokenizer).
    - If list[int] or 1-D Tensor (already tokenized) -> pad to the same
      length and return a Tensor [batch, max_len].
    """
    if len(texts) == 0:
        return texts

    first = texts[0]
    if isinstance(first, str):
        return list(texts)

    if isinstance(first, (list, tuple)):
        tensors = [torch.as_tensor(t, dtype=torch.long) for t in texts]
        return pad_sequence(tensors, batch_first=True, padding_value=pad_value)

    if torch.is_tensor(first):
        tensors = [t.long() for t in texts]
        return pad_sequence(tensors, batch_first=True, padding_value=pad_value)

    # some other type (e.g. dict returned by a HF tokenizer) -> leave as list
    return list(texts)


def make_collate_fn(pad_value: int = 0) -> Callable:
    """Returns a collate_fn for use with DataLoader, with a configurable pad_value."""

    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated: Dict[str, Any] = {}
        keys = batch[0].keys()

        for key in keys:
            values = [sample[key] for sample in batch]

            if key == "image":
                collated[key] = _collate_images(values)
            elif key in ("question", "answer"):
                collated[key] = _collate_text(values, pad_value=pad_value)
            else:
                # question_id, image_id, filename, ...
                collated[key] = list(values)

        return collated

    return collate_fn


def get_dataloader(
    json_path: str,
    image_dir: str,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    tokenizer: Optional[Callable] = None,
    is_test: bool = False,
    pad_value: int = 0,
    **dataloader_kwargs,
) -> DataLoader:
    """
    Quickly build a DataLoader for OpenViVQA.

    Example:
        train_loader = get_dataloader(
            json_path="datasets/train/vlsp2023_train_data.json",
            image_dir="datasets/train/training_images",
            batch_size=32,
            shuffle=True,
            transform=my_image_transform,
            tokenizer=my_tokenizer,   # can be None
            is_test=False,
        )
        for batch in train_loader:
            images = batch["image"]        # Tensor [B, C, H, W] or list[PIL.Image]
            questions = batch["question"]  # Tensor [B, L] or list[str]
            answers = batch["answer"]      # Tensor [B, L] or list[str]
    """
    dataset = OpenViVQADataset(
        json_path=json_path,
        image_dir=image_dir,
        transform=transform,
        tokenizer=tokenizer,
        is_test=is_test,
    )

    collate_fn = make_collate_fn(pad_value=pad_value)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )


if __name__ == "__main__":
    # Full usage example for the train / dev / test splits
    # (adjust the paths to match your own "datasets/..." layout)

    import torchvision.transforms as T

    image_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # train_loader = get_dataloader(
    #     json_path="datasets/train/vlsp2023_train_data.json",
    #     image_dir="datasets/train/training-images",
    #     batch_size=16,
    #     shuffle=True,
    #     transform=image_transform,
    #     is_test=False,
    # )
    

    dev_loader = get_dataloader(
        json_path="datasets/dev/vlsp2023_dev_data.json",
        image_dir="datasets/dev/dev-images",
        batch_size=16,
        shuffle=False,
        transform=image_transform,
        is_test=False,
    )

    # test_loader = get_dataloader(
    #     json_path="datasets/test/vlsp2023_test_data.json",
    #     image_dir="datasets/test/test-images",
    #     batch_size=16,
    #     shuffle=False,
    #     transform=image_transform,
    #     is_test=True,
    # )

    batch = next(iter(dev_loader))
    print("Batch keys:", batch.keys())
    print("Images:", batch["image"].shape)
    print("Number of questions in batch:", len(batch["question"]))
    print("Example question:", batch["question"][0])
    print("Example answer:", batch["answer"][0])