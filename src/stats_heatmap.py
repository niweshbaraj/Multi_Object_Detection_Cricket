"""
Cricket Statistical Analysis and Heatmap Generation
Analyzes cricket videos to generate object position heatmaps and statistics
"""

import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import glob

def analyze_cricket_videos():
    """
    Main function to analyze cricket videos and generate statistics
    """
    # Initialize model
    model = YOLO('models/yolov8-cricket.pt')
    
    # Storage for analysis results
    ball_positions = []
    player_positions = defaultdict(list)
    detection_counts = defaultdict(int)
    
    # Find all video files to analyze
    video_patterns = [
        'data/front view/*.mp4',
        'data/side view batsman/*.mp4', 
        'data/side view bowler/*.mp4'
    ]
    
    all_videos = []
    for pattern in video_patterns:
        videos = glob.glob(pattern)
        all_videos.extend(videos)
    
    print(f"Found {len(all_videos)} videos to analyze")
    
    # Process first 2 videos for demonstration
    videos_to_process = all_videos[:2]
    
    # Analyze each video
    for video_path in videos_to_process:
        print(f"Analyzing: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to speed up analysis
            if frame_count % 5 == 0:
                results = model(frame, conf=0.3)
                
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        # Get object details
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = model.names[class_id]
                        
                        # Get center coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        
                        # Count detections
                        detection_counts[class_name] += 1
                        
                        # Store ball positions for heatmap
                        if class_name == 'ball' and confidence > 0.5:
                            ball_positions.append({
                                'frame': frame_count,
                                'x': center_x,
                                'y': center_y,
                                'confidence': confidence
                            })
                        
                        # Store player positions
                        elif class_name in ['batsman', 'bowler', 'umpire']:
                            player_positions[class_name].append({
                                'frame': frame_count,
                                'x': center_x,
                                'y': center_y,
                                'confidence': confidence
                            })
            
            frame_count += 1
        
        cap.release()
        print(f"Processed {frame_count} frames")
    
    # Generate outputs
    create_output_directories()
    generate_ball_heatmap(ball_positions)
    generate_player_heatmaps(player_positions)
    generate_detection_statistics(detection_counts)
    save_ball_trajectory_data(ball_positions)
    
    print("Statistical analysis completed!")
    print("Check outputs/heatmaps/ and outputs/stats/ for results")

def create_output_directories():
    """Create necessary output directories"""
    os.makedirs('outputs/heatmaps', exist_ok=True)
    os.makedirs('outputs/stats', exist_ok=True)

def generate_ball_heatmap(ball_positions):
    """Generate heatmap showing ball position frequency"""
    if not ball_positions:
        print("No ball positions detected for heatmap")
        return
    
    df = pd.DataFrame(ball_positions)
    
    plt.figure(figsize=(12, 8))
    plt.hist2d(df['x'], df['y'], bins=30, cmap='Reds')
    plt.colorbar(label='Ball Detection Frequency')
    plt.title('Cricket Ball Position Heatmap')
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.gca().invert_yaxis()  # Match image coordinate system
    
    output_path = 'outputs/heatmaps/ball_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ball heatmap saved: {output_path}")

def generate_player_heatmaps(player_positions):
    """Generate position heatmaps for each player type"""
    for player_type, positions in player_positions.items():
        if not positions:
            continue
        
        df = pd.DataFrame(positions)
        
        plt.figure(figsize=(10, 8))
        plt.hist2d(df['x'], df['y'], bins=25, cmap='Blues')
        plt.colorbar(label=f'{player_type.title()} Detection Frequency')
        plt.title(f'{player_type.title()} Position Heatmap')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        plt.gca().invert_yaxis()
        
        output_path = f'outputs/heatmaps/{player_type}_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{player_type.title()} heatmap saved: {output_path}")

def generate_detection_statistics(detection_counts):
    """Generate and save detection statistics"""
    # Create DataFrame
    stats_data = []
    for object_class, count in detection_counts.items():
        stats_data.append({'Object_Class': object_class, 'Total_Detections': count})
    
    df = pd.DataFrame(stats_data)
    df = df.sort_values('Total_Detections', ascending=False)
    
    # Save CSV
    csv_path = 'outputs/stats/detection_summary.csv'
    df.to_csv(csv_path, index=False)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Object_Class'], df['Total_Detections'], color='skyblue')
    plt.title('Cricket Object Detection Statistics')
    plt.xlabel('Object Classes')
    plt.ylabel('Number of Detections')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    chart_path = 'outputs/stats/detection_statistics.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Detection statistics saved:")
    print(f"  CSV: {csv_path}")
    print(f"  Chart: {chart_path}")

def save_ball_trajectory_data(ball_positions):
    """Save ball trajectory data for further analysis"""
    if not ball_positions:
        print("No ball trajectory data available")
        return
    
    df = pd.DataFrame(ball_positions)
    df = df.sort_values('frame')
    
    trajectory_path = 'outputs/stats/ball_trajectory.csv'
    df.to_csv(trajectory_path, index=False)
    
    print(f"Ball trajectory data saved: {trajectory_path}")

if __name__ == "__main__":
    analyze_cricket_videos()
