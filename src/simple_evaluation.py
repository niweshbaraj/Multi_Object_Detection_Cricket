"""
Model Evaluation for Cricket Object Detection
Compatible with GTX 1650 - Uses CPU validation to avoid memory issues
"""

import os
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_cricket_model():
    """
    Evaluate the cricket detection model performance
    Uses CPU validation to avoid GPU memory limitations
    """
    
    print("Starting Model Evaluation")
    
    # Create output directory
    output_dir = 'outputs/model_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = YOLO('models/yolov8-cricket.pt')
    
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print("Model Architecture:")
    print(f"Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    print(f"Classes: {model.model.names}")
    
    # Run validation on dataset
    print("\nRunning validation on cricket dataset...")
    try:
        # Use CPU for validation to avoid memory issues
        results = model.val(
            data='data/annotations/cricket.yaml',
            imgsz=320,  # Smaller image size for memory efficiency
            batch=1,    # Small batch size
            device='cpu',  # Force CPU usage
            verbose=False
        )
        
        # Extract key metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'Precision': results.box.mp,
            'Recall': results.box.mr
        }
        
        print("\nValidation Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
            
        # Save metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(f'{output_dir}/model_metrics.csv', index=False)
        
        # Class-wise performance analysis
        if hasattr(results.box, 'maps'):
            class_performance = {}
            for i, class_name in enumerate(model.model.names.values()):
                if i < len(results.box.maps):
                    class_performance[class_name] = results.box.maps[i]
            
            print("\nPer-Class Performance (mAP50-95):")
            for class_name, score in class_performance.items():
                print(f"{class_name}: {score:.4f}")
            
            # Save class performance data
            class_df = pd.DataFrame(list(class_performance.items()), 
                                  columns=['Class', 'mAP50-95'])
            class_df.to_csv(f'{output_dir}/class_performance.csv', index=False)
            
            # Create performance visualization
            plt.figure(figsize=(10, 6))
            plt.bar(class_performance.keys(), class_performance.values())
            plt.title('Per-Class Model Performance (mAP50-95)')
            plt.xlabel('Object Class')
            plt.ylabel('mAP50-95 Score')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/class_performance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Validation failed: {e}")
        print("Using alternative evaluation method...")
        
        # Alternative: Analyze detection statistics from video processing
        stats_file = 'outputs/stats/detection_summary.csv'
        if os.path.exists(stats_file):
            stats_df = pd.read_csv(stats_file)
            
            print("\nDetection Statistics Analysis:")
            total_detections = stats_df['Total_Detections'].sum()
            print(f"Total objects detected: {total_detections}")
            
            # Calculate detection rates
            stats_df['Detection_Rate'] = stats_df['Total_Detections'] / total_detections * 100
            
            print("\nObject Detection Rates:")
            for _, row in stats_df.iterrows():
                print(f"{row['Object_Class']}: {row['Total_Detections']} ({row['Detection_Rate']:.1f}%)")
            
            # Create detection distribution chart
            plt.figure(figsize=(10, 6))
            plt.pie(stats_df['Total_Detections'], labels=stats_df['Object_Class'], autopct='%1.1f%%')
            plt.title('Object Detection Distribution')
            plt.savefig(f'{output_dir}/detection_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"\nEvaluation complete. Results saved to: {output_dir}")
    return True

if __name__ == "__main__":
    evaluate_cricket_model()
