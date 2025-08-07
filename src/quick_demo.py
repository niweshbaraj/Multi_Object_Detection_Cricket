"""
Quick Demo: Cricket Object Detection - Process Sample Frames Only
For fast demonstration of object detection capabilities
"""

import cv2
import os
import glob
from ultralytics import YOLO

def demo_detection_on_frames(video_path, model_path='models/yolov8-cricket.pt', num_frames=10):
    """
    Quick demo: Process only sample frames from video for fast results
    
    Args:
        video_path: Path to input cricket video
        model_path: Path to trained YOLO model  
        num_frames: Number of frames to process for demo
    """
    # Load trained model
    model = YOLO(model_path)
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Demo processing: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames}, Processing: {num_frames} sample frames")
    
    # Create output directory for demo frames
    output_dir = 'outputs/demo_frames'
    os.makedirs(output_dir, exist_ok=True)
    
    detection_summary = {}
    
    # Process sample frames at intervals
    frame_interval = total_frames // num_frames if total_frames > num_frames else 1
    
    for i in range(num_frames):
        frame_number = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run object detection
        results = model(frame, conf=0.3)
        
        # Count detections
        frame_detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                frame_detections.append((class_name, confidence))
                
                # Update summary
                if class_name not in detection_summary:
                    detection_summary[class_name] = 0
                detection_summary[class_name] += 1
        
        # Save annotated frame
        if frame_detections:
            # Draw bounding boxes (same as main script)
            class_colors = {
                'ball': (0, 255, 0), 'bat': (255, 255, 0), 'batsman': (255, 0, 0),
                'bowler': (0, 0, 255), 'player': (128, 0, 128), 'stumps': (255, 165, 0),
                'umpire': (0, 255, 255)
            }
            
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                color = class_colors.get(class_name, (255, 255, 255))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Save frame
            frame_filename = f"frame_{frame_number:04d}_detections.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            
            print(f"Frame {frame_number}: {len(frame_detections)} objects detected")
    
    cap.release()
    
    # Print summary
    print(f"\nDetection Summary:")
    for obj_class, count in detection_summary.items():
        print(f"  {obj_class}: {count} detections")
    
    print(f"\nDemo frames saved to: {os.path.abspath(output_dir)}")
    return True

def run_quick_demo():
    """Run quick demo on one video"""
    
    # Find first available video
    video_patterns = [
        'data/front view/*.mp4',
        'data/side view batsman/*.mp4', 
        'data/side view bowler/*.mp4'
    ]
    
    demo_video = None
    for pattern in video_patterns:
        videos = glob.glob(pattern)
        if videos:
            demo_video = videos[0]  # Take first video found
            break
    
    if demo_video:
        print(f"Running quick demo on: {os.path.basename(demo_video)}")
        demo_detection_on_frames(demo_video, num_frames=5)
    else:
        print("No video files found for demo")

if __name__ == "__main__":
    run_quick_demo()
