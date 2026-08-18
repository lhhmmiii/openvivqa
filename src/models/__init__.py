from src.models.cnn_lstm import CNN_LSTM

MODEL_REGISTRY = {
    "cnn_lstm": CNN_LSTM,
}


def build_model(name: str, **kwargs):
    """Instantiate a model by name.

    Args:
        name: key in MODEL_REGISTRY (e.g. 'cnn_lstm').
        **kwargs: forwarded to the model constructor.

    Returns:
        nn.Module instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)
