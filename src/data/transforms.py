import torchvision.transforms as T


def get_image_transform(image_size: int = 224):
    """Default image preprocessing for ImageNet-pretrained backbones."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
