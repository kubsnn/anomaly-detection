from .logger import setup_logging
from .paths import get_video_paths
from .snapshots import save_snapshot
from .snapshots import list_available_models
from .snapshots import select_model
from .evaluation import evaluate_model

__ALL__ = ['setup_logging', 'get_video_paths', 'save_snapshot', 'evaluate_model']



__version__ = '0.1.0'