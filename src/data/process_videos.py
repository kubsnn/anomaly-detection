import cv2
from pathlib import Path
import logging
from tqdm import tqdm
from contextlib import contextmanager


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


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


def process_video(input_path: Path, output_path: Path, target_fps: int = 3,
                  target_size: tuple = (320, 180)):

    with video_capture(input_path) as cap:
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, original_fps // target_fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            raise ValueError(f"Invalid frame count in video: {input_path}")

        with video_writer(output_path, target_fps, target_size) as out:
            frame_count = 0
            processed_count = 0

            expected_frames = total_frames // frame_interval

            with tqdm(total=expected_frames,
                      desc=f"Processing {input_path.name}",
                      leave=False) as pbar:

                while frame_count < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()

                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                            normalized = cv2.normalize(gray, None, 0, 255,
                                                       cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                            resized = cv2.resize(normalized, target_size,
                                                 interpolation=cv2.INTER_AREA)
                            out.write(resized)
                            processed_count += 1
                            pbar.update(1)

                        except cv2.error as e:
                            raise RuntimeError(f"OpenCV error processing frame {frame_count}: {str(e)}")

                    frame_count += frame_interval

                    if processed_count >= expected_frames:
                        break

    return processed_count


def process_dataset(base_path: str):
    logger = setup_logging()
    base_path = Path(base_path)
    normal_dir = base_path / 'videos' / 'normal'

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
                    logger.debug(f"Skipping existing video: {output_path}")
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
    logger = setup_logging()

    # dataset base path
    base_path = "../../../UBI_FIGHTS"

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