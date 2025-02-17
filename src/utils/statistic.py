import os
import cv2
import pandas as pd
from pathlib import Path
from datetime import timedelta


def get_video_duration(file_path):
    """
    Get duration of a video file in seconds
    """
    try:
        video = cv2.VideoCapture(str(file_path))
        if not video.isOpened():
            return None

        # Get total number of frames and frames per second
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # Calculate duration
        duration = total_frames / fps if fps > 0 else None

        video.release()
        return duration
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def format_time(seconds):
    """
    Convert seconds to HH:MM:SS format
    """
    return str(timedelta(seconds=int(seconds)))


def analyze_videos(directory):
    """
    Analyze all videos in the specified directory and calculate statistics
    """
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

    # Get all video files
    video_files = [
        f for f in Path(directory).glob('**/*')
        if f.suffix.lower() in video_extensions and f.is_file()
    ]

    if not video_files:
        return "No video files found in the specified directory."

    # Get durations
    durations = []
    file_stats = []

    for video_file in video_files:
        duration = get_video_duration(video_file)
        if duration is not None:
            durations.append(duration)
            file_stats.append({
                'file': video_file.name,
                'duration': duration,
                'duration_formatted': format_time(duration)
            })

    if not durations:
        return "Could not process any video files in the directory."

    # Calculate statistics
    stats = {
        'total_videos': len(file_stats),
        'mean_duration': sum(durations) / len(durations),
        'median_duration': pd.Series(durations).median(),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'std_deviation': pd.Series(durations).std(),
        'total_duration': sum(durations)
    }

    # Create a DataFrame for individual video stats
    df_files = pd.DataFrame(file_stats)

    # Format the results
    results = {
        'summary_stats': {
            'Total Videos': stats['total_videos'],
            'Mean Duration': format_time(stats['mean_duration']),
            'Median Duration': format_time(stats['median_duration']),
            'Min Duration': format_time(stats['min_duration']),
            'Max Duration': format_time(stats['max_duration']),
            'Standard Deviation': f"{stats['std_deviation']:.2f} seconds",
            'Total Duration': format_time(stats['total_duration'])
        },
        'individual_files': df_files.sort_values('duration', ascending=False)
    }

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Calculate video statistics for a directory')
    parser.add_argument('directory', help='Directory containing video files')
    args = parser.parse_args()

    results = analyze_videos(args.directory)

    if isinstance(results, str):
        print(results)
    else:
        print("\nSummary Statistics:")
        for stat, value in results['summary_stats'].items():
            print(f"{stat}: {value}")

        print("\nIndividual File Details:")
        print(results['individual_files'].to_string(index=False))