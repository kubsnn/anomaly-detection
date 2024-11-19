import os
from .logger import setup_logging

logger = setup_logging(__name__)


def get_video_paths(base_path, subset_size, use_dvs, split, type='normal'):
    normal_dir = os.path.join(base_path, 'v2e', 'videos', type) if use_dvs else os.path.join(base_path, 'videos', type)
    split_dir = os.path.join(normal_dir, split)
    if not use_dvs:
        split_dir = os.path.join(split_dir, 'processed')
    video_paths = []

    if os.path.exists(split_dir):
        videos = sorted([os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(('.mp4', '.avi'))])
        total_files = len(videos)
        selected_files = videos[:subset_size]
        video_paths.extend(selected_files)
        logger.info(f"Found {total_files} files in '{split_dir}'. Selected {len(selected_files)}.")
    else:
        logger.warning(f"Split directory '{split_dir}' does not exist.")

    return video_paths
