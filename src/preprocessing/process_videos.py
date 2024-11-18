import cv2
import numpy as np
from pathlib import Path

from tqdm import tqdm
from contextlib import contextmanager

import sys
sys.path.append('..')
from utils import setup_logging


logger = setup_logging(__name__)

@contextmanager
def video_capture(path):
    # Safely handle video opening with automatic closing
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {path}")
        yield cap
    finally:
        cap.release()

@contextmanager
def video_writer(path, fps, size, is_color=False):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size, isColor=is_color)
    try:
        if not writer.isOpened():
            raise IOError(f"Failed to create video writer: {path}")
        yield writer
    finally:
        writer.release()


def detect_content_area(frame, threshold=30):
    if frame is None:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    target_ratio = 16 / 9

    def is_content_row(values, threshold):
        std = np.std(values)
        return std > threshold / 2

    def is_content_col(values, threshold):
        std = np.std(values)
        mean = np.mean(values)
        return std > threshold / 8 or (mean > threshold and mean < 255 - threshold)

    left = 0
    right = w - 1
    top = 0
    bottom = h - 1

    for x in range(w // 2):
        if is_content_col(gray[:, x], threshold):
            left = x
            break

    for x in range(w - 1, w // 2, -1):
        if is_content_col(gray[:, x], threshold):
            right = x
            break

    for y in range(h // 2):
        if is_content_row(gray[y, :], threshold):
            top = y
            break

    for y in range(h - 1, h // 2, -1):
        if is_content_row(gray[y, :], threshold):
            bottom = y
            break

    content_width = right - left
    content_height = bottom - top
    current_ratio = content_width / content_height

    # Force 16:9 ratio by cropping excess content
    if current_ratio < target_ratio:
        # Video is too tall 4:3 - crop height
        required_height = int(content_width / target_ratio)
        excess_height = content_height - required_height

        top += excess_height // 2
        bottom -= excess_height // 2
    else:
        # Video is too wide - crop width
        required_width = int(content_height * target_ratio)
        excess_width = content_width - required_width

        left += excess_width // 2
        right -= excess_width // 2

    # Add a small margin to ensure no white edges
    margin = 2
    left = left + margin
    right = right - margin
    top = top + margin
    bottom = bottom - margin

    return (left, right, top, bottom)


def process_video(input_path: Path, output_path: Path, target_fps: int = 3,
                  target_size: tuple = (320, 180)):

    with video_capture(input_path) as cap:
        content_coords = None
        sample_frames = 5
        frame_positions = np.linspace(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1, sample_frames, dtype=int)

        lefts, rights, tops, bottoms = [], [], [], []

        for pos in frame_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if not ret:
                continue

            coords = detect_content_area(frame)
            if coords:
                left, right, top, bottom = coords
                lefts.append(left)
                rights.append(right)
                tops.append(top)
                bottoms.append(bottom)

        if not lefts:
            raise ValueError(f"Could not detect content area in {input_path}")

        left = int(np.median(lefts))
        right = int(np.median(rights))
        top = int(np.median(tops))
        bottom = int(np.median(bottoms))

        content_coords = (left, right, top, bottom)

        with video_writer(output_path, target_fps, target_size, is_color=False) as out:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(1, original_fps // target_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            processed_count = 0

            with tqdm(total=total_frames // frame_interval,
                      desc=f"Processing {input_path.name}",
                      leave=False) as pbar:

                while frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        content = frame[top:bottom, left:right]

                        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)

                        resized = cv2.resize(gray, target_size,
                                             interpolation=cv2.INTER_AREA)

                        normalized = cv2.normalize(resized, None, 0, 255,
                                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                        out.write(normalized)
                        processed_count += 1
                        pbar.update(1)

                    frame_count += 1

            logger.info(f"Processed {processed_count} frames from {input_path.name}")
            logger.info(f"Crop coordinates: ({left}, {right}, {top}, {bottom})")

    return processed_count


def process_dataset(base_path: str):
    base_path = Path(base_path)
    normal_dir = base_path / 'videos' / 'fight'

    total_processed = 0
    total_errors = 0

    for split in ['train', 'val', 'test']:
        split_dir = normal_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        processed_dir = split_dir / 'processed'
        processed_dir.mkdir(exist_ok=True)

        video_files = list(split_dir.glob('*.mp4')) + list(split_dir.glob('*.avi'))
        video_files = [f for f in video_files if f.is_file()]

        logger.info(f"\nProcessing {split} split: {len(video_files)} videos")

        for video_path in tqdm(video_files, desc=f"{split} progress"):
            try:
                output_path = processed_dir / f"{video_path.stem}_processed.mp4"

                if output_path.exists():
                    logger.info(f"Skipping existing video: {output_path}")
                    continue

                processed_frames = process_video(video_path, output_path)
                total_processed += 1

            except Exception as e:
                logger.error(f"Error processing {video_path.name}: {str(e)}")
                total_errors += 1
                continue

    logger.info(f"\nProcessing completed:")
    logger.info(f"Successfully processed: {total_processed} videos")
    logger.info(f"Failed to process: {total_errors} videos")


def main():

    # dataset path
    base_path = "../../data/UBI_FIGHTS"

    try:
        logger.info("Starting video processing pipeline...")
        process_dataset(base_path)
        logger.info("Video processing completed successfully!")

    except KeyboardInterrupt:
        logger.info("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()