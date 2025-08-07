import os
import cv2

def extract_frames(video_path, output_dir, fps_extract=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error opening {video_path}')
        return
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps_extract)
    count = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_dir, f'frame_{saved:06}.jpg'), frame)
            saved += 1
        count += 1
    cap.release()
    print(f'Extracted {saved} frames from {video_path}')

# Example batch extraction for all folders
import glob

# Process all three folders
folders = ['front view', 'side view batsman', 'side view bowler']
for folder in folders:
    print(f"\nProcessing folder: {folder}")
    pattern = f'data/{folder}/*.mp4'
    video_files = glob.glob(pattern)
    print(f"Found {len(video_files)} video files in {folder}")
    
    for video_file in video_files:
        output_dir = video_file.replace('.mp4', '_frames')
        extract_frames(video_file, output_dir, fps_extract=2)
