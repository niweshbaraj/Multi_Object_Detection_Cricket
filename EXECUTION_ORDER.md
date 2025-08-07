# EXECUTION_ORDER.md

# Cricket Multi-Object Detection - Execution Order

## Phase 1: Data Preparation (Completed)
1. extract_frames.py - Extract frames from cricket videos
2. Roboflow annotation - Manual annotation of cricket objects  
3. train.py - Train YOLOv8 model on cricket dataset
4. Download best model to models/yolov8-cricket.pt

## Phase 2: Task Implementation

### Core Requirements:
```bash
# 1. Object Detection & Annotation
python src/detect_annotate.py
# Output: Annotated videos in outputs/annotated_videos/

# 2. Object Tracking (Bonus)
python src/track_objects.py  
# Output: Tracked videos in outputs/tracked_videos/

# 3. Statistical Analysis (Bonus)
python src/stats_heatmap.py
# Output: Heatmaps in outputs/heatmaps/, stats in outputs/stats/
```

### Optional Analysis:
```bash  
# 4. Model Performance Evaluation
python src/evaluate_model.py
# Output: Performance reports in outputs/model_evaluation/
```

## Phase 3: Submission Generation
```bash
# Create submission package
python generate_task_submission.py
# Output: Complete submission in TASK_SUBMISSION/
```

## Expected Timeline:
- Phase 2: 30-60 minutes (depending on video count)  
- Phase 3: 5-10 minutes
- Total: ~1 hour for complete task submission

## File Dependencies:
- All scripts require: models/yolov8-cricket.pt
- Input videos: data/front view/, data/side view batsman/, data/side view bowler/
- Python packages: ultralytics, opencv-python, deep-sort-realtime
