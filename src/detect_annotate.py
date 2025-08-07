"""
Cricket Object Detection and Video Annotation
Detects cricket objects (ball, players, stumps) and annotates videos with bounding boxes
"""

import cv2
import os
import glob
from ultralytics import YOLO

def annotate_video(video_path, output_path, model_path='models/yolov8-cricket.pt'):
    """
    Annotate single video with object detection bounding boxes
    
    Args:
        video_path: Path to input cricket video
        output_path: Path for annotated output video
        model_path: Path to trained YOLO model
    """
    # Load trained model
    model = YOLO(model_path)
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    
    frame_count = 0
    detection_count = 0
    
    # Define colors for each object class
    class_colors = {
        'ball': (0, 255, 0),      # Green
        'bat': (255, 255, 0),     # Yellow
        'batsman': (255, 0, 0),   # Red  
        'bowler': (0, 0, 255),    # Blue
        'player': (128, 0, 128),  # Purple
        'stumps': (255, 165, 0),  # Orange
        'umpire': (0, 255, 255)   # Cyan
    }
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run object detection on frame
        results = model(frame, conf=0.3)
        
        # Draw bounding boxes for detected objects
        if results[0].boxes is not None:
            for box in results[0].boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                # Get color for this class
                color = class_colors.get(class_name, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                label = f'{class_name} {confidence:.2f}'
                
                # Get text size for background rectangle
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                detection_count += 1
        
        # Write frame to output video
        out.write(frame)
        frame_count += 1
        
        # Show progress every 100 frames
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"Completed: {os.path.basename(output_path)}")
    print(f"Total detections: {detection_count}")
    print(f"Frames processed: {frame_count}")
    
    return True

def process_all_videos():
    """Process all cricket videos in data directory"""
    
    # Create output directory
    output_dir = 'outputs/annotated_videos'
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_patterns = [
        'data/front view/*.mp4',
        'data/side view batsman/*.mp4',
        'data/side view bowler/*.mp4'
    ]
    
    all_videos = []
    for pattern in video_patterns:
        videos = glob.glob(pattern)
        all_videos.extend(videos)
    
    print(f"Found {len(all_videos)} videos to process")
    
    # Process each video
    success_count = 0
    for i, video_path in enumerate(all_videos, 1):
        print(f"\n[{i}/{len(all_videos)}] Processing video...")
        
        # Create output filename
        video_name = os.path.basename(video_path)
        output_filename = f"annotated_{video_name}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Process video
        if annotate_video(video_path, output_path):
            success_count += 1
    
    print(f"\nSummary: {success_count}/{len(all_videos)} videos processed successfully")
    print(f"Output directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    process_all_videos()
