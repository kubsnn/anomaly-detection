import sys

sys.path.append('..')
import pandas as pd
import cv2
from pathlib import Path
from utils.logger import setup_logging

logger = setup_logging(__name__)


def extract_fight_clips(base_path: Path, min_output_duration: float = 3.0):
    """
    Extract fight sequences from videos based on annotations.
    Discards any resulting videos shorter than min_output_duration seconds.

    Args:
        base_path (Path): Base directory containing the dataset
        min_output_duration (float): Minimum duration in seconds for output videos
    """
    videos_directory = base_path / "videos" / "fight"
    annotations_directory = base_path / "annotation"

    total_clips = 0
    discarded_clips = 0

    # Create output directory
    output_directory = videos_directory / "cut_fights"
    output_directory.mkdir(exist_ok=True)

    # Find all video files
    video_files = list(videos_directory.glob("*.mp4"))
    logger.info(f"Found {len(video_files)} videos to process")

    for video_path in video_files:
        annotation_path = annotations_directory / f"{video_path.stem}.csv"

        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_path}")
            continue

        try:
            annotations = pd.read_csv(annotation_path, header=None)
        except Exception as e:
            logger.error(f"Error reading CSV file {annotation_path}: {e}")
            continue

        logger.info(f"Processing {video_path.name}...")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"Error loading video: {video_path}")
            continue

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Extract fight sequences
            fight_sequences = []
            in_fight = False
            start_frame = 0

            for idx, row in annotations.iterrows():
                if row[0] == 1 and not in_fight:
                    in_fight = True
                    start_frame = idx
                elif row[0] == 0 and in_fight:
                    in_fight = False
                    end_frame = idx
                    fight_sequences.append((start_frame, end_frame))

            # Handle case where video ends during a fight
            if in_fight:
                end_frame = len(annotations)
                fight_sequences.append((start_frame, end_frame))

            # Process fight sequences
            for start_frame, end_frame in fight_sequences:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                output_path = output_directory / f"{video_path.stem}_cut_{start_frame}_{end_frame}.mp4"

                try:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps,
                                             (frame_width, frame_height))

                    frames_written = 0
                    for frame_num in range(start_frame, end_frame):
                        ret, frame = cap.read()
                        if not ret:
                            logger.error(f"Error reading frame {frame_num} in {video_path.name}")
                            break
                        writer.write(frame)
                        frames_written += 1

                    writer.release()

                    # Check output video duration
                    output_duration = frames_written / fps
                    if output_duration < min_output_duration:
                        output_path.unlink()
                        discarded_clips += 1
                        logger.debug(
                            f"Discarding {output_path.name}: duration {output_duration:.2f}s < {min_output_duration}s"
                        )
                    else:
                        total_clips += 1
                        logger.info(f"Saved fight clip: {output_path.name} ({output_duration:.1f}s)")

                except Exception as e:
                    logger.error(f"Error processing clip from frame {start_frame} to {end_frame} "
                                 f"in {video_path.name}: {e}")
                    if output_path.exists():
                        output_path.unlink()

        finally:
            cap.release()

    logger.info(f"\nExtraction completed:")
    logger.info(f"Total clips saved: {total_clips}")
    logger.info(f"Clips discarded (too short): {discarded_clips}")


if __name__ == "__main__":
    logger.warning("This script should be run through the pipeline, not directly.")
    logger.warning("Run 'python run_pipeline.py' instead.")