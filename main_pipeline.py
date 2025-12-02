#!/usr/bin/env python3
"""
Automated Parking Lot Occupancy Detection Pipeline
Using YOLOv5 → Mask R-CNN → DINO for complete automation

Author: MiniMax Agent
Date: 2025-12-02
"""

import os
import cv2
import numpy as np
import pandas as pd
import json
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Any
import yaml

# Import model classes
from yolo_detector import YOLODetector
from maskrcnn_segmenter import MaskRCNNSegmenter  
from dino_tracker import DINOModule
from parking_space_analyzer import ParkingSpaceAnalyzer
from visualizer import Visualizer
from data_processor import DataProcessor

class ParkingLotPipeline:
    """
    Complete automated pipeline for parking lot occupancy detection
    Sequential processing: YOLO → Mask R-CNN → DINO
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.setup_logging()
        
        # Initialize models
        self.yolo = YOLODetector(self.config['yolo'])
        self.maskrcnn = MaskRCNNSegmenter(self.config['maskrcnn'])
        self.dino = DINOModule(self.config['dino'])
        
        # Initialize utilities
        self.analyzer = ParkingSpaceAnalyzer()
        self.visualizer = Visualizer()
        self.processor = DataProcessor()
        
        # Results storage
        self.results = {}
        self.occupancy_data = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('parking_pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Pipeline initialized successfully")
    
    def process_dataset(self, dataset_path: str, output_path: str = "results") -> Dict[str, Any]:
        """
        Process the entire PKLot dataset automatically
        
        Args:
            dataset_path: Path to PKLot dataset
            output_path: Directory to save results
            
        Returns:
            Complete analysis results
        """
        self.logger.info(f"Starting automated processing of dataset: {dataset_path}")
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/visualizations", exist_ok=True)
        os.makedirs(f"{output_path}/data", exist_ok=True)
        
        # Get all images from dataset
        image_files = self._get_all_images(dataset_path)
        self.logger.info(f"Found {len(image_files)} images to process")
        
        processed_results = {}
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self._process_single_image(img_path)
                processed_results[img_path] = result
                
                # Save intermediate results
                self._save_image_result(img_path, result, output_path)
                
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        
        # Generate comprehensive analysis
        final_results = self._generate_final_analysis(processed_results, output_path)
        
        self.logger.info("Pipeline processing completed successfully")
        return final_results
    
    def _get_all_images(self, dataset_path: str) -> List[str]:
        """Get all image files from the dataset directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        return sorted(image_files)
    
    def _process_single_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline
        YOLO → Mask R-CNN → DINO
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.logger.debug(f"Processing image: {os.path.basename(image_path)}")
        
        # Step 1: YOLO - Initial vehicle detection
        yolo_results = self.yolo.detect(image)
        self.logger.debug(f"YOLO detected {len(yolo_results['detections'])} vehicles")
        
        # Step 2: Mask R-CNN - Precise segmentation
        maskrcnn_results = self.maskrcnn.segment(image, yolo_results)
        self.logger.debug(f"Mask R-CNN generated {len(maskrcnn_results['segments'])} segments")
        
        # Step 3: DINO - Refined detection and tracking
        dino_results = self.dino.refine_detection(image, maskrcnn_results)
        self.logger.debug(f"DINO refined to {len(dino_results['refined_detections'])} objects")
        
        # Combine results
        combined_results = {
            'image_path': image_path,
            'yolo_detections': yolo_results,
            'maskrcnn_segments': maskrcnn_results,
            'dino_refined': dino_results,
            'final_detections': dino_results['refined_detections']
        }
        
        return combined_results
    
    def _save_image_result(self, image_path: str, result: Dict, output_path: str):
        """Save intermediate results for an image"""
        img_name = Path(image_path).stem
        img_dir = os.path.dirname(image_path)
        
        # Create subdirectory structure
        rel_path = os.path.relpath(img_dir, start=os.path.dirname(self.config['dataset']['path']))
        save_dir = os.path.join(output_path, "visualizations", rel_path)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save annotated image
        annotated_img = self.visualizer.create_annotated_image(
            cv2.imread(image_path), result
        )
        cv2.imwrite(os.path.join(save_dir, f"{img_name}_annotated.jpg"), annotated_img)
    
    def _generate_final_analysis(self, processed_results: Dict, output_path: str) -> Dict[str, Any]:
        """Generate comprehensive final analysis"""
        self.logger.info("Generating final analysis...")
        
        # Analyze each image result
        all_results = []
        occupancy_summary = []
        
        for img_path, result in processed_results.items():
            img_analysis = self.analyzer.analyze_image_occupancy(result)
            img_analysis['image_path'] = img_path
            all_results.append(img_analysis)
            occupancy_summary.append({
                'image': os.path.basename(img_path),
                'total_spaces': img_analysis['total_spaces'],
                'occupied_spaces': img_analysis['occupied_spaces'],
                'vacant_spaces': img_analysis['vacant_spaces'],
                'occupancy_rate': img_analysis['occupancy_rate'],
                'vacant_space_ids': img_analysis['vacant_space_ids']
            })
        
        # Generate final report
        final_results = {
            'summary': {
                'total_images_processed': len(processed_results),
                'total_detections': sum(r['total_detections'] for r in all_results),
                'average_occupancy_rate': np.mean([r['occupancy_rate'] for r in all_results]),
                'occupancy_distribution': self._analyze_occupancy_distribution(occupancy_summary)
            },
            'detailed_results': all_results,
            'occupancy_summary': occupancy_summary,
            'vacant_space_mappings': self._generate_vacant_space_mappings(occupancy_summary)
        }
        
        # Save results to files
        self._save_final_results(final_results, output_path)
        
        return final_results
    
    def _analyze_occupancy_distribution(self, occupancy_summary: List[Dict]) -> Dict:
        """Analyze occupancy patterns across all images"""
        occupancy_rates = [item['occupancy_rate'] for item in occupancy_summary]
        
        return {
            'min_occupancy': min(occupancy_rates),
            'max_occupancy': max(occupancy_rates),
            'mean_occupancy': np.mean(occupancy_rates),
            'std_occupancy': np.std(occupancy_rates),
            'occupancy_categories': {
                'low': len([r for r in occupancy_rates if r < 0.3]),
                'medium': len([r for r in occupancy_rates if 0.3 <= r < 0.7]),
                'high': len([r for r in occupancy_rates if r >= 0.7])
            }
        }
    
    def _generate_vacant_space_mappings(self, occupancy_summary: List[Dict]) -> Dict:
        """Generate detailed mappings of vacant spaces for each image"""
        vacant_mappings = {}
        
        for item in occupancy_summary:
            img_name = item['image']
            vacant_mappings[img_name] = {
                'total_spaces': item['total_spaces'],
                'vacant_count': item['vacant_spaces'],
                'vacant_space_labels': item['vacant_space_ids'],
                'occupancy_status': self._categorize_occupancy(item['occupancy_rate'])
            }
        
        return vacant_mappings
    
    def _categorize_occupancy(self, rate: float) -> str:
        """Categorize occupancy rate"""
        if rate < 0.3:
            return "Low Occupancy"
        elif rate < 0.7:
            return "Medium Occupancy"
        else:
            return "High Occupancy"
    
    def _save_final_results(self, results: Dict, output_path: str):
        """Save final results to various formats"""
        # Save JSON
        with open(f"{output_path}/final_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save CSV summary
        df_summary = pd.DataFrame(results['occupancy_summary'])
        df_summary.to_csv(f"{output_path}/occupancy_summary.csv", index=False)
        
        # Save vacant space mappings
        with open(f"{output_path}/vacant_space_mappings.json", 'w') as f:
            json.dump(results['vacant_space_mappings'], f, indent=2)
        
        # Generate visualization
        self.visualizer.create_summary_visualizations(results, output_path)
        
        # Print summary
        self._print_summary(results)
    
    def _print_summary(self, results: Dict):
        """Print processing summary to console"""
        summary = results['summary']
        print("\n" + "="*60)
        print("PARKING LOT ANALYSIS - FINAL SUMMARY")
        print("="*60)
        print(f"Total Images Processed: {summary['total_images_processed']}")
        print(f"Total Vehicle Detections: {summary['total_detections']}")
        print(f"Average Occupancy Rate: {summary['average_occupancy_rate']:.2%}")
        print("\nOccupancy Distribution:")
        dist = summary['occupancy_distribution']
        print(f"  Low Occupancy (<30%): {dist['occupancy_categories']['low']} images")
        print(f"  Medium Occupancy (30-70%): {dist['occupancy_categories']['medium']} images")
        print(f"  High Occupancy (>70%): {dist['occupancy_categories']['high']} images")
        print("\nVacant Space Information:")
        print("Detailed mappings saved in 'vacant_space_mappings.json'")
        print("="*60)

def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = ParkingLotPipeline()
    
    # Set dataset path (update this path to your PKLot dataset location)
    dataset_path = "path/to/your/pklot_dataset"  # Update this path
    
    # Process dataset
    results = pipeline.process_dataset(dataset_path, output_path="parking_results")
    
    return results

if __name__ == "__main__":
    results = main()