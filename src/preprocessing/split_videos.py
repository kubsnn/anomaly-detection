import sys
import argparse

sys.path.append('..')
from pathlib import Path
import cv2
from tqdm import tqdm
from frame_sampling import video_capture, video_writer
from utils.logger import setup_logging

logger = setup_logging(__name__)


def split_processed_videos(base_path: Path, segment_length: int = 10):
    """Split processed videos into segments of specified length.
    Last segment will be segment_length + remainder frames.

    Args:
        base_path (Path): Base directory containing the dataset
        segment_length (int): Length of each segment in seconds
    """
    categories = ['normal', 'fight/cut_fights']

    for category in categories:
        input_dir = base_path / "videos" / category / "processed"
        if not input_dir.exists():
            logger.warning(f"Directory not found: {input_dir}")
            continue

        split_dir = input_dir / "split"
        split_dir.mkdir(exist_ok=True)

        video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi'))
        video_files = [f for f in video_files if f.is_file() and not f.name.startswith('split_')]

        logger.info(f"Found {len(video_files)} processed videos in {category}")
        total_split = 0
        segments_created = 0

        for video_path in tqdm(video_files, desc=f"Splitting videos in {category}"):
            try:
                with video_capture(video_path) as cap:
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = total_frames / fps

                    # Only split if video is longer than twice the segment length
                    if duration < 2 * segment_length:
                        continue

                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    segment_frames = segment_length * fps

                    # Calculate number of full segments, excluding the last one
                    num_full_segments = (total_frames - 1) // segment_frames
                    if num_full_segments > 1:  # Need at least 2 segments to have a remainder
                        num_segments = num_full_segments - 1  # Reserve last full segment to combine with remainder
                    else:
                        num_segments = num_full_segments

                    # Process full segments
                    for segment in range(num_segments):
                        start_frame = segment * segment_frames
                        end_frame = start_frame + segment_frames

                        segment_path = split_dir / f"split_{video_path.stem}_segment_{segment}{video_path.suffix}"

                        with video_writer(segment_path, fps, (frame_width, frame_height)) as out:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                            for _ in range(segment_frames):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                out.write(frame)

                        segments_created += 1

                    # Handle last segment (combine last full segment with remainder)
                    if num_full_segments > 0:
                        start_frame = num_segments * segment_frames
                        end_frame = total_frames

                        segment_path = split_dir / f"split_{video_path.stem}_segment_{num_segments}{video_path.suffix}"

                        with video_writer(segment_path, fps, (frame_width, frame_height)) as out:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                            for _ in range(start_frame, end_frame):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                out.write(frame)

                        segments_created += 1

                    # Delete original video after successful splitting
                    video_path.unlink()
                    total_split += 1

            except Exception as e:
                logger.error(f"Error splitting {video_path.name}: {str(e)}")
                continue

        logger.info(f"\nSplitting completed for {category}:")
        logger.info(f"Videos split: {total_split}")
        logger.info(f"Total segments created: {segments_created}")


def parse_args():
    parser = argparse.ArgumentParser(description="Split processed videos into segments")
    parser.add_argument("--base-path", type=str, default="../../data/UBI_FIGHTS",
                        help="Base directory containing the UBI-FIGHTS dataset")
    parser.add_argument("--segment-length", type=int, default=10,
                        help="Length of each segment in seconds")
    return parser.parse_args()


def main():
    """Command line interface for video splitting."""
    args = parse_args()
    base_path = Path(args.base_path)

    try:
        logger.info("Starting video splitting process...")
        logger.info(f"Base path: {base_path}")
        logger.info(f"Segment length: {args.segment_length} seconds")

        split_processed_videos(base_path, args.segment_length)
        logger.info("Video splitting completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\nProcessing interrupted by user")
    except Exception as e:
        logger.error(f"Video splitting failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()