#!/usr/bin/env python3
"""
Example usage of the Automated Parking Lot Detection Pipeline
Demonstrates how to run the complete YOLO → Mask R-CNN → DINO pipeline

Author: MiniMax Agent
Date: 2025-12-02

Usage:
    python example_usage.py
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_pipeline import ParkingLotPipeline

def main():
    """Main execution function"""
    
    print("🚀 Automated Parking Lot Detection Pipeline")
    print("=" * 60)
    print("YOLOv5 → Mask R-CNN → DINO → Occupancy Analysis")
    print("Optimized for RTX 4060 8GB, 16GB RAM")
    print("=" * 60)
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Step 1: Initialize Pipeline
        print("\n📋 Step 1: Initializing Pipeline...")
        pipeline = ParkingLotPipeline("config.yaml")
        print("✅ Pipeline initialized successfully")
        
        # Step 2: Set Dataset Path
        print("\n📁 Step 2: Setting up Dataset Path...")
        dataset_path = "path/to/your/pklot_dataset"  # UPDATE THIS PATH
        
        # Check if dataset path exists
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path not found: {dataset_path}")
            print("\n" + "="*60)
            print("SETUP INSTRUCTIONS:")
            print("="*60)
            print("1. Download PKLot dataset from:")
            print("   http://www.ic.unicamp.br/~rocha/pub/datasets/pklot/")
            print("\n2. Extract the dataset to a folder")
            print("\n3. Update the 'dataset_path' variable in this script")
            print("   with your actual dataset location")
            print("\n4. Run this script again")
            print("\nExample paths:")
            print("   Windows: 'C:/Users/YourName/PKLot'")
            print("   Linux/Mac: '/home/username/PKLot'")
            print("="*60)
            return
        
        print(f"✅ Dataset found: {dataset_path}")
        
        # Count images in dataset
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_count = 0
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_count += 1
        
        print(f"📊 Found {image_count} images in dataset")
        
        if image_count == 0:
            print("❌ No image files found in the dataset directory")
            print("   Please ensure your dataset contains image files")
            return
        
        # Step 3: Process Dataset
        print(f"\n🔄 Step 3: Processing {image_count} images...")
        print("   This may take some time depending on dataset size...")
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_{timestamp}"
        
        print(f"📤 Results will be saved to: {output_path}/")
        
        # Run the complete pipeline
        results = pipeline.process_dataset(dataset_path, output_path)
        
        # Step 4: Display Results
        print("\n" + "="*60)
        print("🎉 PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Display summary statistics
        summary = results.get('summary', {})
        occupancy_summary = results.get('occupancy_summary', [])
        
        print(f"📊 SUMMARY STATISTICS:")
        print(f"   Total Images Processed: {summary.get('total_images_processed', 0)}")
        print(f"   Total Vehicle Detections: {summary.get('total_detections', 0)}")
        print(f"   Average Occupancy Rate: {summary.get('average_occupancy_rate', 0):.2%}")
        
        if 'occupancy_distribution' in summary:
            dist = summary['occupancy_distribution']
            print(f"\n📈 OCCUPANCY DISTRIBUTION:")
            print(f"   Low Occupancy (<30%): {dist['occupancy_categories']['low']} images")
            print(f"   Medium Occupancy (30-70%): {dist['occupancy_categories']['medium']} images")
            print(f"   High Occupancy (>70%): {dist['occupancy_categories']['high']} images")
        
        # Display vacant space summary
        vacant_mappings = results.get('vacant_space_mappings', {})
        if vacant_mappings:
            total_vacant = sum(data['vacant_count'] for data in vacant_mappings.values())
            total_spaces = sum(data['total_spaces'] for data in vacant_mappings.values())
            
            print(f"\n🅿️ VACANT SPACE ANALYSIS:")
            print(f"   Total Vacant Spaces (All Images): {total_vacant}")
            print(f"   Total Parking Spaces (All Images): {total_spaces}")
            print(f"   Average Vacancy Rate: {total_vacant/total_spaces:.2%}" if total_spaces > 0 else "   Average Vacancy Rate: 0%")
            
            # Show images with most vacant spaces
            sorted_images = sorted(vacant_mappings.items(), 
                                 key=lambda x: x[1]['vacant_count'], reverse=True)
            
            print(f"\n🎯 TOP 5 IMAGES WITH MOST VACANT SPACES:")
            for i, (image_name, data) in enumerate(sorted_images[:5]):
                vacancy_rate = data['vacant_count']/data['total_spaces'] if data['total_spaces'] > 0 else 0
                print(f"   {i+1}. {image_name}: {data['vacant_count']}/{data['total_spaces']} vacant ({vacancy_rate:.1%})")
        
        # Step 5: Output Files Information
        print(f"\n📁 OUTPUT FILES GENERATED:")
        print(f"   📊 {output_path}/final_results.json - Complete analysis results")
        print(f"   📈 {output_path}/occupancy_summary.csv - Summary statistics")
        print(f"   📋 {output_path}/vacant_space_mappings.json - Detailed vacant space info")
        print(f"   🎨 {output_path}/visualizations/ - Annotated images")
        print(f"   📑 {output_path}/vacancy_summary_report.txt - Text summary")
        print(f"   📊 {output_path}/summary_report.png - Statistical charts")
        
        # Step 6: Next Steps
        print(f"\n🔍 NEXT STEPS:")
        print(f"   1. Review annotated images in: {output_path}/visualizations/")
        print(f"   2. Check detailed vacant space list in CSV/JSON files")
        print(f"   3. Analyze the summary charts and reports")
        print(f"   4. Modify config.yaml to adjust detection parameters if needed")
        
        # Step 7: Display Vacant Space List (Sample)
        if occupancy_summary:
            print(f"\n🅿️ SAMPLE VACANT SPACE IDENTIFICATION:")
            sample_images = occupancy_summary[:3]  # Show first 3 images as example
            
            for item in sample_images:
                image_name = item['image']
                vacant_ids = item.get('vacant_space_ids', [])
                print(f"   📷 {image_name}:")
                if vacant_ids:
                    print(f"      Vacant Spaces: {', '.join(vacant_ids[:5])}{'...' if len(vacant_ids) > 5 else ''}")
                    print(f"      Total Vacant: {len(vacant_ids)}")
                else:
                    print(f"      No vacant spaces detected")
        
        print("\n" + "="*60)
        print("🎯 PIPELINE EXECUTION COMPLETED!")
        print("Check the results directory for detailed output files.")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\n🔧 TROUBLESHOOTING:")
        print("   1. Check if all dependencies are installed (run: python setup.py)")
        print("   2. Verify dataset path is correct")
        print("   3. Ensure sufficient GPU memory (8GB+ recommended)")
        print("   4. Check configuration in config.yaml")
        print("   5. Review error logs for more details")
        
        # Print the actual error for debugging
        import traceback
        print(f"\n🔍 Detailed error information:")
        traceback.print_exc()

def show_configuration_help():
    """Show help for configuration options"""
    print("\n⚙️ CONFIGURATION OPTIONS:")
    print("="*50)
    print("Edit config.yaml to customize:")
    print("\n🚗 Vehicle Detection:")
    print("   - yolo.confidence_threshold (default: 0.5)")
    print("   - yolo.vehicle_classes (COCO classes: 2=car, 3=truck, etc.)")
    print("\n🅿️ Parking Space Analysis:")
    print("   - parking_analysis.space_detection.grid_spacing_x/y")
    print("   - parking_analysis.occupancy_detection.vehicle_overlap_threshold")
    print("\n🎨 Visualization:")
    print("   - visualizer.colors (RGB values for occupied/vacant spaces)")
    print("   - visualizer.output.save_annotated_images")
    print("\n⚡ Performance:")
    print("   - hardware.recommended_settings.batch_size")
    print("   - performance.gpu_memory_fraction")
    print("="*50)

def check_system_requirements():
    """Check system requirements"""
    print("\n🔍 SYSTEM REQUIREMENTS CHECK:")
    print("="*40)
    
    # Check Python version
    import sys
    if sys.version_info >= (3, 8):
        print(f"✅ Python version: {sys.version}")
    else:
        print(f"❌ Python version: {sys.version} (Requires 3.8+)")
        return False
    
    # Check PyTorch and CUDA
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU (slower)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Check OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    # Check disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free // (1024**3)
    if free_gb >= 5:
        print(f"✅ Disk space: {free_gb}GB free")
    else:
        print(f"⚠️  Disk space: {free_gb}GB free (5GB+ recommended)")
    
    print("="*40)
    return True

if __name__ == "__main__":
    # Show system check
    if not check_system_requirements():
        print("\n❌ System requirements not met. Please run: python setup.py")
        sys.exit(1)
    
    # Show configuration help
    show_configuration_help()
    
    # Run main function
    main()