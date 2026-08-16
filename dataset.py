import os
import json
from typing import Callable, Dict, List, Optional, Any

from PIL import Image
from torch.utils.data import Dataset


class OpenViVQADataset(Dataset):
    """Dataset that reads directly from OpenViVQA JSON annotations and the image directory."""

    def __init__(
        self,
        json_path: str,
        image_dir: str,
        transform: Optional[Callable] = None,
        is_test: bool = False,
        return_image: bool = True,
    ) -> None:
        """
        Args:
            json_path: path to vlsp2023_train_data.json / vlsp2023_dev_data.json / vlsp2023_test_data.json
            image_dir: directory containing corresponding images (e.g. datasets/train/training_images)
            transform: image transform function (e.g. torchvision.transforms.Compose(...)).
                       If None, a PIL.Image (RGB) is returned unchanged.
            is_test: if True, the dataset is treated as a test set (no labels available)
            return_image: if False, images will not be loaded (useful when only
                          building text vocabularies; much faster).
        """

        self.json_path = json_path
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test
        self.return_image = return_image

        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # --- normalize "images": always convert to dict {str(image_id): filename} ---
        images_field = raw.get("images", {})
        if isinstance(images_field, dict):
            self.images: Dict[str, str] = {str(k): v for k, v in images_field.items()}
        else:
            raise ValueError("Field 'images' in JSON is not in a supported format.")

        annotations = raw.get("annotations", {}) 
        if not annotations:
            raise ValueError(f"Field 'annotations' not found in {json_path}")

        self.samples: List[Dict[str, Any]] = []
        for ann_id, ann in annotations.items():
            # Get image filename from image_id
            image_id = str(ann["image_id"])
            filename = self.images.get(image_id)
            if filename is None:
                continue
            
            # Get question and answer
            question = ann["question"]

            if not self.is_test:
                answer = ann["answer"]
                self.samples.append(
                    {
                        "question_id": ann_id,
                        "image_id": image_id,
                        "filename": filename,
                        "question": question,
                        "answer": answer,
                    }
                )
            else:
                self.samples.append(
                    {
                        "question_id": ann_id,
                        "image_id": image_id,
                        "filename": filename,
                        "question": question,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, filename: str):
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.samples[idx])  # copy to avoid modifying the original data

        if self.return_image:
            item["image"] = self._load_image(item["filename"])

        return item


if __name__ == "__main__":
    ds = OpenViVQADataset(
        json_path="datasets/dev/vlsp2023_dev_data.json",
        image_dir="datasets/dev/dev-images",
    )
    print("Number of samples:", len(ds))
    print("First sample:", {k: v for k, v in ds.samples[0].items()})