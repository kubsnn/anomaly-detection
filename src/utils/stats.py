import cv2
import os
from datetime import timedelta


def analyze_videos(directory):
    """Analyze all videos in a given directory and return statistics."""
    video_durations = []
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')

    for filename in os.listdir(directory):
        if filename.lower().endswith(valid_extensions):
            try:
                video_path = os.path.join(directory, filename)
                cap = cv2.VideoCapture(video_path)

                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps

                video_durations.append(duration)
                cap.release()
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    if not video_durations:
        return None

    return {
        'count': len(video_durations),
        'average_length': sum(video_durations) / len(video_durations),
        'max_length': max(video_durations)
    }


def format_duration(seconds):
    """Convert seconds to human-readable time format."""
    return str(timedelta(seconds=int(seconds)))


def main():
    directories = ['../../../UBI_FIGHTS/videos/normal', '../../../UBI_FIGHTS/videos/fight']

    for directory in directories:
        print(f"\nAnalyzing videos in '{directory}' directory:")
        stats = analyze_videos(directory)

        if stats:
            print(f"Number of videos: {stats['count']}")
            print(f"Average length: {format_duration(stats['average_length'])}")
            print(f"Longest video: {format_duration(stats['max_length'])}")
        else:
            print(f"No valid videos found in {directory} directory")


if __name__ == "__main__":
    main()