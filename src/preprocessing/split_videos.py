import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add src/ directory to the path
from pathlib import Path
import cv2
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from frame_sampling import video_capture
from utils.logger import setup_logging

logger = setup_logging(__name__)



def write_segment(cap, start_frame, num_frames, segment_path, fps, width, height):
    """Write a segment of frames to a video file with fallback codecs."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(str(segment_path), fourcc, fps, (width, height))

    if not out.isOpened():
        # Fallback to XVID
        segment_path = segment_path.with_suffix('.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(segment_path), fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error(f"Could not create video writer for {segment_path}")
            return False

    try:
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
    finally:
        out.release()

    return True


import shutil

def split_video(video_path, split_dir, segment_length):
    """Split a single video into segments of the specified length."""
    try:
        with video_capture(video_path) as cap:
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return 0, 0

            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            logger.info(f"Processing video: {video_path.name}")
            logger.info(f"FPS: {fps}")
            logger.info(f"Total frames: {total_frames}")
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Required duration: {2 * segment_length} seconds")

            # Copy videos that are too short but at least 4 seconds
            if duration < 2 * segment_length:
                if duration >= 3:
                    copied_path = split_dir / video_path.name
                    try:
                        shutil.copy(video_path, copied_path)
                        logger.info(f"Copied short video to output: {copied_path}")
                        return 1, 0  # 1 copied video, 0 segments created
                    except Exception as e:
                        logger.error(f"Failed to copy short video: {video_path}, error: {str(e)}")
                else:
                    logger.info(f"Skipping video - too short (<3 seconds, got {duration:.2f}s)")
                return 0, 0

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            segment_frames = segment_length * fps

            num_segments = (total_frames - 1) // segment_frames
            segments_created = 0

            # Process full segments
            for segment in range(num_segments):
                start_frame = segment * segment_frames
                segment_path = split_dir / f"{video_path.stem}_s{segment}.mp4"
                if not write_segment(
                    cap, start_frame, segment_frames, segment_path, fps, frame_width, frame_height
                ):
                    logger.error(f"Failed to write segment: {segment_path}")
                    continue

                segments_created += 1

            # Handle last segment (remaining frames)
            start_frame = num_segments * segment_frames
            segment_path = split_dir / f"{video_path.stem}_s{num_segments}.mp4"
            if not write_segment(
                cap, start_frame, total_frames - start_frame, segment_path, fps, frame_width, frame_height
            ):
                logger.error(f"Failed to write last segment: {segment_path}")
            else:
                segments_created += 1

            return 0, segments_created  # 0 copied videos, N segments created
    except Exception as e:
        logger.error(f"Error splitting {video_path.name}: {str(e)}")
        return 0, 0


def split_processed_videos(base_path: Path, segment_length: int = 10, max_threads: int = 4):
    """Split processed videos into segments using multithreading."""
    categories = ['fight']

    for category in categories:
        input_dir = base_path / "v2e" / "videos" / category
        if not input_dir.exists():
            logger.warning(f"Directory not found: {input_dir}")
            continue

        split_dir = input_dir / "split"
        split_dir.mkdir(exist_ok=True)

        video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
        video_files = [f for f in video_files if f.is_file() and not f.name.startswith('split_')]

        logger.info(f"Found {len(video_files)} processed videos in {category}")

        total_segments = 0
        total_copied = 0
        total_videos = 0

        with ThreadPoolExecutor(max_threads) as executor:
            futures = [
                executor.submit(split_video, video_path, split_dir, segment_length)
                for video_path in video_files
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Splitting videos in {category}"):
                copied, segments = future.result()
                total_copied += copied
                total_segments += segments
                total_videos += 1

        logger.info(f"\nSplitting completed for {category}:")
        logger.info(f"Total videos processed: {total_videos}")
        logger.info(f"Total segments created: {total_segments}")
        logger.info(f"Total videos copied (too short): {total_copied}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split processed videos into segments")
    parser.add_argument("--base-path", type=str, default="../../data/UBI_FIGHTS",
                        help="Base directory containing the dataset")
    parser.add_argument("--segment-length", type=int, default=10,
                        help="Length of each segment in seconds")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of threads to use for parallel processing")
    return parser.parse_args()


def main():
    args = parse_args()
    base_path = Path(args.base_path)

    try:
        logger.info("Starting video splitting process...")
        logger.info(f"Base path: {base_path}")
        logger.info(f"Segment length: {args.segment_length} seconds")
        logger.info(f"Threads: {args.threads}")

        split_processed_videos(base_path, args.segment_length, args.threads)
        logger.info("Video splitting completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Video splitting failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
