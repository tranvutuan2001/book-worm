import mlx_lm

class MLXModel:
    """Shared resource for MLX model and tokenizer."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model, self.tokenizer = mlx_lm.load(model_path)
