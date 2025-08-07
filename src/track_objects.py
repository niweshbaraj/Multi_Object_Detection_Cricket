"""
Cricket Player Tracking with DeepSORT
Tracks batsman, bowler, and umpire with consistent IDs across frames
"""

import cv2
import os
import glob
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def track_players(video_path, output_path, model_path='models/yolov8-cricket.pt'):
    """
    Track cricket players in video with consistent IDs
    
    Args:
        video_path: Input cricket video path
        output_path: Output video path with tracking IDs
        model_path: Path to trained YOLO model
    """
    print(f"Tracking: {os.path.basename(video_path)}")
    
    # Load model and tracker
    model = YOLO(model_path)
    deepsort = DeepSort(max_age=30, n_init=3)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    print(f"Processing {total_frames} frames...")
    
    frame_count = 0
    tracked_objects = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)[0]
        detections = []
        
        # Extract player detections only
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                
                # Only track specific player types
                if label not in ['batsman', 'bowler', 'umpire']:
                    continue
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = [x1, y1, x2-x1, y2-y1]  # Convert to [x,y,w,h] format
                detections.append((bbox, conf, label))
        
        # Update tracker
        tracks = deepsort.update_tracks(detections, frame=frame)
        
        # Draw tracking results
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            # Get bounding box
            l, t, w_, h_ = map(int, track.to_tlwh())
            
            # Draw tracking box (blue for tracked objects)
            cv2.rectangle(frame, (l, t), (l + w_, t + h_), (255, 0, 0), 2)
            
            # Draw track ID
            label_text = f'ID {track.track_id}'
            cv2.putText(frame, label_text, (l, t - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            tracked_objects += 1
        
        # Write frame
        out.write(frame)
        frame_count += 1
        
        # Progress update
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Completed: {os.path.basename(output_path)}")
    print(f"Total tracks created: {tracked_objects}")
    print(f"Frames processed: {frame_count}")
    
    return True

def process_tracking():
    """Process cricket videos for player tracking"""
    
    # Create output directory
    output_dir = 'outputs/tracked_videos'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find video files
    video_patterns = [
        'data/front view/*.mp4',
        'data/side view batsman/*.mp4',
        'data/side view bowler/*.mp4'
    ]
    
    all_videos = []
    for pattern in video_patterns:
        videos = glob.glob(pattern)
        all_videos.extend(videos)
    
    print(f"Found {len(all_videos)} videos to track")
    
    # Process first 2 videos for demonstration
    videos_to_process = all_videos[:2]
    print(f"Processing first {len(videos_to_process)} videos for demo")
    
    success_count = 0
    for i, video_path in enumerate(videos_to_process, 1):
        print(f"\n[{i}/{len(videos_to_process)}] Processing video...")
        
        # Create output filename
        video_name = os.path.basename(video_path)
        output_filename = f"tracked_{video_name}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Track video
        if track_players(video_path, output_path):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(videos_to_process)} videos tracked successfully")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    process_tracking()
