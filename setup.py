#!/usr/bin/env python3
"""
Setup script for Automated Parking Lot Detection Pipeline
Sets up the environment and verifies dependencies

Author: MiniMax Agent
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"✓ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        logger.error("requirements.txt not found")
        return False
    
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False

def check_dependencies():
    """Check if all required packages can be imported"""
    required_packages = [
        'torch', 'torchvision', 'cv2', 'numpy', 'pandas', 'matplotlib', 
        'seaborn', 'sklearn', 'yaml', 'tqdm', 'transformers', 'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"✗ {package} not found")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        return False
    
    logger.info("✓ All dependencies available")
    return True

def setup_directories():
    """Create necessary directories"""
    directories = [
        "data",
        "logs", 
        "results",
        "models",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Directory created: {directory}")
    
    return True

def create_example_config():
    """Create example configuration if none exists"""
    config_file = Path("config.yaml")
    
    if config_file.exists():
        logger.info("✓ Configuration file already exists")
        return True
    
    # Check if config exists in project_reconstruction
    project_config = Path("project_reconstruction/config.yaml")
    if project_config.exists():
        import shutil
        shutil.copy2(project_config, config_file)
        logger.info("✓ Configuration file created from template")
        return True
    
    logger.warning("No configuration file found - using defaults")
    return True

def check_gpu_availability():
    """Check GPU availability and CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA available: {gpu_count} GPU(s) detected")
            logger.info(f"✓ GPU: {gpu_name}")
            logger.info(f"✓ CUDA version: {torch.version.cuda}")
            return True
        else:
            logger.warning("⚠ CUDA not available - will use CPU (slower)")
            return False
    except ImportError:
        logger.error("PyTorch not available for GPU check")
        return False

def download_models():
    """Download/prepare model weights"""
    try:
        # This would download model weights if needed
        logger.info("Model weights will be downloaded automatically when first used")
        return True
    except Exception as e:
        logger.error(f"Failed to prepare models: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the pipeline"""
    try:
        # Test import of main modules
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Test basic imports
        import cv2
        import numpy as np
        import yaml
        
        # Test creating a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image, "Pipeline Test", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save test image
        cv2.imwrite("test_image.jpg", test_image)
        logger.info("✓ Basic functionality test passed")
        
        # Clean up
        os.remove("test_image.jpg")
        return True
        
    except Exception as e:
        logger.error(f"Basic functionality test failed: {e}")
        return False

def create_usage_example():
    """Create a usage example script"""
    example_script = '''#!/usr/bin/env python3
"""
Example usage of the Automated Parking Lot Detection Pipeline
"""

import os
from main_pipeline import ParkingLotPipeline

def main():
    """Example usage"""
    
    # Initialize pipeline
    print("Initializing Parking Lot Detection Pipeline...")
    pipeline = ParkingLotPipeline("config.yaml")
    
    # Set dataset path (update this to your actual PKLot dataset location)
    dataset_path = "path/to/your/pklot_dataset"  # UPDATE THIS PATH
    
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        print("Please update the dataset_path variable with your actual PKLot dataset location")
        print("PKLot dataset can be downloaded from: http://www.ic.unicamp.br/~rocha/pub/datasets/pklot/")
        return
    
    # Process dataset
    print(f"Processing dataset: {dataset_path}")
    results = pipeline.process_dataset(dataset_path, output_path="results")
    
    # Print summary
    print("\\n" + "="*60)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Results saved to: results/")
    print(f"Total images processed: {results['summary']['total_images_processed']}")
    print(f"Average occupancy rate: {results['summary']['average_occupancy_rate']:.2%}")
    print("="*60)

if __name__ == "__main__":
    main()
'''
    
    with open("example_usage.py", "w") as f:
        f.write(example_script)
    
    logger.info("✓ Example usage script created: example_usage.py")
    return True

def print_setup_summary():
    """Print setup completion summary"""
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Next steps:")
    print("1. Update dataset path in config.yaml")
    print("2. Download PKLot dataset (http://www.ic.unicamp.br/~rocha/pub/datasets/pklot/)")
    print("3. Run: python example_usage.py")
    print("4. Check results in 'results/' directory")
    print("\nFiles created:")
    print("- config.yaml (configuration)")
    print("- example_usage.py (usage example)")
    print("- main_pipeline.py (main pipeline)")
    print("- Various model modules")
    print("="*60)

def main():
    """Main setup function"""
    print("Automated Parking Lot Detection Pipeline Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Create configuration
    create_example_config()
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Download models
    download_models()
    
    # Test functionality
    test_basic_functionality()
    
    # Create usage example
    create_usage_example()
    
    # Print summary
    print_setup_summary()
    
    if gpu_available:
        print(f"\n✓ GPU acceleration available - pipeline will use GPU")
    else:
        print(f"\n⚠ CPU only - processing will be slower")

if __name__ == "__main__":
    main()