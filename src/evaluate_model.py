"""
Model Performance Evaluation for Cricket Object Detection
Evaluates YOLO model on validation dataset and generates comprehensive metrics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import glob
import cv2
import time

class ModelEvaluator:
    def __init__(self, model_path='models/yolov8-cricket.pt'):
        self.model = YOLO(model_path)
        self.class_names = ['ball', 'bat', 'batsman', 'bowler', 'player', 'stumps', 'umpire']
        self.results = {}
        
    def evaluate_on_validation_set(self):
        """Evaluate model on validation dataset"""
        print("🔍 Evaluating model on validation dataset...")
        
        # Use YOLO's built-in validation
        validation_results = self.model.val(
            data='data/annotations/cricket.yaml',
            imgsz=640,
            conf=0.25,
            iou=0.5,
            save_json=True,
            plots=True
        )
        
        self.results['validation'] = validation_results
        return validation_results
    
    def benchmark_inference_speed(self, test_images_dir='data/annotations/valid/images', num_samples=100):
        """Benchmark inference speed on sample images"""
        print("⚡ Benchmarking inference speed...")
        
        image_files = glob.glob(os.path.join(test_images_dir, '*.jpg'))[:num_samples]
        
        if not image_files:
            print(f"No images found in {test_images_dir}")
            return None
        
        times = []
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            start_time = time.time()
            results = self.model(img)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        
        speed_stats = {
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'min_time_ms': np.min(times) * 1000,
            'max_time_ms': np.max(times) * 1000,
            'std_time_ms': np.std(times) * 1000,
            'num_samples': len(times)
        }
        
        self.results['speed_benchmark'] = speed_stats
        
        print(f"Average Inference Time: {avg_time*1000:.2f} ms")
        print(f"Average FPS: {fps:.2f}")
        
        return speed_stats
    
    def analyze_class_performance(self):
        """Analyze per-class performance metrics"""
        print("📊 Analyzing per-class performance...")
        
        if 'validation' not in self.results:
            print("Run validation first!")
            return None
        
        val_results = self.results['validation']
        
        # Extract metrics if available
        try:
            metrics_data = []
            
            # Get per-class metrics from validation results
            if hasattr(val_results, 'results_dict'):
                results_dict = val_results.results_dict
                
                for i, class_name in enumerate(self.class_names):
                    metrics_data.append({
                        'class': class_name,
                        'precision': results_dict.get(f'metrics/precision(B)', 0),
                        'recall': results_dict.get(f'metrics/recall(B)', 0),
                        'mAP50': results_dict.get(f'metrics/mAP50(B)', 0),
                        'mAP50-95': results_dict.get(f'metrics/mAP50-95(B)', 0)
                    })
            
            df = pd.DataFrame(metrics_data)
            self.results['class_performance'] = df
            
            return df
            
        except Exception as e:
            print(f"Error analyzing class performance: {e}")
            return None
    
    def create_performance_visualizations(self, output_dir='outputs/model_evaluation/'):
        """Create comprehensive performance visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Speed Benchmark Visualization
        if 'speed_benchmark' in self.results:
            speed_data = self.results['speed_benchmark']
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Inference time distribution
            ax1.bar(['Avg Time'], [speed_data['avg_inference_time_ms']], 
                   color='skyblue', alpha=0.7)
            ax1.set_ylabel('Inference Time (ms)')
            ax1.set_title('Average Inference Time')
            ax1.text(0, speed_data['avg_inference_time_ms'] + 1, 
                    f"{speed_data['avg_inference_time_ms']:.1f} ms", 
                    ha='center')
            
            # FPS visualization
            ax2.bar(['FPS'], [speed_data['fps']], color='lightgreen', alpha=0.7)
            ax2.set_ylabel('Frames Per Second')
            ax2.set_title('Processing Speed')
            ax2.text(0, speed_data['fps'] + 0.5, 
                    f"{speed_data['fps']:.1f} FPS", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'speed_benchmark.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
        
        # 2. Model Architecture Summary
        self._plot_model_summary(output_dir)
        
        print(f"📈 Visualizations saved to: {output_dir}")
    
    def _plot_model_summary(self, output_dir):
        """Plot model architecture summary"""
        try:
            # Get model info
            model_info = {
                'Model': 'YOLOv8',
                'Classes': len(self.class_names),
                'Parameters': f'{sum(p.numel() for p in self.model.model.parameters()):,}',
                'Input Size': '640x640'
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            
            # Create table
            table_data = [[key, value] for key, value in model_info.items()]
            table = ax.table(cellText=table_data, 
                           colLabels=['Property', 'Value'],
                           cellLoc='left',
                           loc='center',
                           colWidths=[0.3, 0.7])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)
            
            # Style the table
            for i in range(len(table_data) + 1):
                table[(i, 0)].set_facecolor('#E6E6FA')
                table[(i, 1)].set_facecolor('#F0F8FF')
            
            plt.title('Model Configuration Summary', fontsize=16, fontweight='bold', pad=20)
            plt.savefig(os.path.join(output_dir, 'model_summary.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating model summary: {e}")
    
    def generate_evaluation_report(self, output_path='outputs/model_evaluation/evaluation_report.txt'):
        """Generate comprehensive evaluation report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CRICKET OBJECT DETECTION - MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Model Information
            f.write("MODEL INFORMATION:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Model Architecture: YOLOv8\n")
            f.write(f"Classes: {', '.join(self.class_names)}\n")
            f.write(f"Total Classes: {len(self.class_names)}\n\n")
            
            # Speed Benchmark
            if 'speed_benchmark' in self.results:
                speed = self.results['speed_benchmark']
                f.write("SPEED BENCHMARK:\n")
                f.write("-" * 16 + "\n")
                f.write(f"Average Inference Time: {speed['avg_inference_time_ms']:.2f} ms\n")
                f.write(f"Average FPS: {speed['fps']:.2f}\n")
                f.write(f"Min/Max Time: {speed['min_time_ms']:.2f}/{speed['max_time_ms']:.2f} ms\n")
                f.write(f"Tested on {speed['num_samples']} images\n\n")
            
            # Validation Results
            if 'validation' in self.results:
                f.write("VALIDATION RESULTS:\n")
                f.write("-" * 18 + "\n")
                f.write("Detailed validation metrics available in YOLO output files\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")
            f.write("• Model is suitable for real-time processing if FPS > 15\n")
            f.write("• Consider model optimization if inference time > 100ms\n")
            f.write("• Monitor precision/recall for critical classes (ball, batsman)\n")
            f.write("• Fine-tune confidence thresholds based on use case\n")
        
        print(f"📄 Evaluation report saved to: {output_path}")

def run_complete_evaluation():
    """Run complete model evaluation pipeline"""
    print("🚀 Starting Complete Model Evaluation...")
    print("="*50)
    
    evaluator = ModelEvaluator()
    
    # Run validation
    val_results = evaluator.evaluate_on_validation_set()
    
    # Benchmark speed
    speed_results = evaluator.benchmark_inference_speed()
    
    # Analyze class performance
    class_perf = evaluator.analyze_class_performance()
    
    # Create visualizations
    evaluator.create_performance_visualizations()
    
    # Generate report
    evaluator.generate_evaluation_report()
    
    print("\n" + "="*50)
    print("🎉 Model Evaluation Complete!")
    print("Check outputs/model_evaluation/ for detailed results")
    
    return evaluator

if __name__ == "__main__":
    evaluator = run_complete_evaluation()
