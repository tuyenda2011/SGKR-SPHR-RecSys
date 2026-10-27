import logging
import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Detect and return the appropriate torch device (CUDA, MPS, CPU) and log information."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(
            f"Using CUDA device: {gpu_name} (PyTorch v{torch.__version__}, CUDA v{torch.version.cuda})"
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Using Apple MPS device (PyTorch v{torch.__version__})")
    else:
        device = torch.device("cpu")
        logger.info(f"Using CPU device (PyTorch v{torch.__version__})")

    return device
