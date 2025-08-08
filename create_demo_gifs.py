import cv2
import os
from PIL import Image
import numpy as np

def video_to_gif(video_path, gif_path, max_duration=10, fps=2, max_width=640):
    """
    Convert video to GIF with optimized settings for GitHub README
    
    Args:
        video_path: Path to input video file
        gif_path: Path to output GIF file
        max_duration: Maximum duration in seconds
        fps: Output FPS for GIF
        max_width: Maximum width to resize to
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps
    
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"Original: {duration:.1f}s, {total_frames} frames, {original_fps:.1f} FPS")
    
    # Calculate frame sampling
    target_frames = min(int(max_duration * fps), total_frames)
    frame_interval = max(1, total_frames // target_frames)
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Sample frames at intervals
        if frame_count % frame_interval == 0:
            # Resize frame
            height, width = frame.shape[:2]
            if width > max_width:
                ratio = max_width / width
                new_width = max_width
                new_height = int(height * ratio)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
            if len(frames) >= target_frames:
                break
                
        frame_count += 1
    
    cap.release()
    
    if frames:
        # Save as GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000/fps),  # milliseconds per frame
            loop=0,
            optimize=True
        )
        print(f"Created GIF: {gif_path} ({len(frames)} frames)")
        return True
    else:
        print(f"No frames extracted from {video_path}")
        return False

def create_demo_gifs():
    """Create GIFs from selected annotated videos for README demo"""
    
    # Create output directory for GIFs
    gif_dir = "outputs/demo_gifs"
    os.makedirs(gif_dir, exist_ok=True)
    
    # Select representative videos for GIF conversion
    videos_to_convert = [
        {
            "input": "outputs/annotated_videos/annotated_Replay 2025-03-21 20-47-44.mp4",
            "output": "outputs/demo_gifs/cricket_detection_demo1.gif",
            "description": "Recent gameplay with multi-object detection"
        },
        {
            "input": "outputs/annotated_videos/annotated_20240920 - 1 - Camera1 - [18-57-11] [18-57-18].mp4", 
            "output": "outputs/demo_gifs/cricket_detection_demo2.gif",
            "description": "Front view with ball tracking"
        },
        {
            "input": "outputs/annotated_videos/annotated_20240920 - 30 - Camera5 - [19-23-12] [19-23-17].mp4",
            "output": "outputs/demo_gifs/cricket_detection_demo3.gif", 
            "description": "Side view batsman perspective"
        }
    ]
    
    print("🎬 Creating GIFs from annotated videos for README demo...")
    print("=" * 60)
    
    successful_conversions = []
    
    for video_info in videos_to_convert:
        input_path = video_info["input"]
        output_path = video_info["output"]
        
        if os.path.exists(input_path):
            if video_to_gif(input_path, output_path, max_duration=8, fps=1.5, max_width=600):
                successful_conversions.append({
                    "gif_path": output_path,
                    "description": video_info["description"]
                })
                print(f"✅ Success: {output_path}")
            else:
                print(f"❌ Failed: {output_path}")
        else:
            print(f"⚠️  Video not found: {input_path}")
        print("-" * 40)
    
    print(f"\n🎉 Created {len(successful_conversions)} GIFs for README demo")
    
    # Generate README markdown for the GIFs
    if successful_conversions:
        print("\n📝 Add this to your README.md:")
        print("=" * 50)
        for i, gif_info in enumerate(successful_conversions, 1):
            relative_path = gif_info["gif_path"]
            description = gif_info["description"]
            print(f"![Cricket Detection Demo {i}]({relative_path})")
            print(f"*{description}*\n")

if __name__ == "__main__":
    create_demo_gifs()
