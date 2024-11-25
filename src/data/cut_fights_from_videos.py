import os
import pandas as pd
import cv2

videos_directory = "./UBI_FIGHTS/videos/fight/"
annotations_directory = "./UBI_FIGHTS/annotation/"

def extract_fight_clips():
    subfolders = ['val', 'train', 'test']
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(videos_directory, subfolder)
        output_subfolder_path = os.path.join(subfolder_path, "cut_fights")
        os.makedirs(output_subfolder_path, exist_ok=True)
        
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".mp4"):
                video_name = file_name
                annotation_name = file_name.replace(".mp4", ".csv")
                csv_path = os.path.join(annotations_directory, annotation_name)
                video_path = os.path.join(subfolder_path, video_name)

                if not os.path.exists(csv_path):
                    print(f"Warning: Annotation file {annotation_name} not found for {video_name}")
                    continue

                try:
                    annotations = pd.read_csv(csv_path, header=None)
                except Exception as e:
                    print(f"Error reading CSV file {annotation_name}: {e}")
                    continue

                print(f"Processing {video_name} with annotations {annotation_name}...")
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Error loading video {video_name}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

                for start_frame, end_frame in fight_sequences:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                    output_filename = f"{os.path.splitext(video_name)[0]}_cut_{start_frame}_{end_frame}.mp4"
                    output_path = os.path.join(output_subfolder_path, output_filename)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

                    try:
                        for frame_num in range(start_frame, end_frame):
                            ret, frame = cap.read()
                            if not ret:
                                print(f"Error reading frame {frame_num} in {video_name}")
                                break
                            out.write(frame)

                        print(f"Saved fight clip: {output_path}")
                    except Exception as e:
                        print(f"Error processing clip from frame {start_frame} to {end_frame} in {video_name}: {e}")
                    finally:
                        out.release()

                cap.release()

if __name__ == "__main__":
    extract_fight_clips()