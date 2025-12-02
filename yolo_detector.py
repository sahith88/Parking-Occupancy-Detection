#!/usr/bin/env python3
"""
YOLOv5 Vehicle Detection Module
Optimized for RTX 4060 8GB GPU

Author: MiniMax Agent
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Any
import logging

class YOLODetector:
    """YOLOv5-based vehicle detection for parking lot images"""
    
    def __init__(self, config: Dict):
        """
        Initialize YOLOv5 detector
        
        Args:
            config: YOLO configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load model
        model_path = config.get('model_path', 'yolov8n.pt')  # Using YOLOv8 nano for efficiency
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"YOLO model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect vehicles in image using YOLOv5
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary containing detection results
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(
            rgb_image,
            conf=self.config.get('confidence_threshold', 0.5),
            iou=self.config.get('iou_threshold', 0.45),
            imgsz=self.config.get('image_size', 640),
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for vehicle classes (car, truck, bus, motorcycle)
                    vehicle_classes = self.config.get('vehicle_classes', [2, 3, 5, 7])  # COCO dataset vehicle classes
                    
                    if class_id in vehicle_classes:
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': self.model.names.get(class_id, 'unknown'),
                            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                            'area': (x2 - x1) * (y2 - y1)
                        }
                        detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Apply NMS if multiple detections overlap
        detections = self._apply_nms(detections)
        
        result_dict = {
            'detections': detections,
            'image_shape': image.shape,
            'detection_count': len(detections),
            'processing_info': {
                'model': 'YOLOv8',
                'device': self.device,
                'confidence_threshold': self.config.get('confidence_threshold', 0.5),
                'image_size': self.config.get('image_size', 640)
            }
        }
        
        self.logger.debug(f"YOLO detected {len(detections)} vehicles")
        return result_dict
    
    def _apply_nms(self, detections: List[Dict], nms_threshold: float = 0.3) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detection dictionaries
            nms_threshold: IoU threshold for suppression
            
        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections
        
        # Convert to format for cv2.dnn.NMSBoxes
        boxes = []
        scores = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, width, height
            scores.append(det['confidence'])
        
        try:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.config.get('confidence_threshold', 0.5), nms_threshold)
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [detections[i] for i in indices]
            else:
                return []
        except Exception as e:
            self.logger.warning(f"NMS failed: {e}, returning original detections")
            return detections
    
    def get_vehicle_masks(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Generate binary masks for detected vehicles
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Binary mask with vehicle regions
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def refine_detections_for_parking(self, detections: List[Dict], image_shape: Tuple) -> List[Dict]:
        """
        Refine detections specifically for parking space analysis
        
        Args:
            detections: List of detection dictionaries
            image_shape: Shape of the input image
            
        Returns:
            Refined detections
        """
        height, width = image_shape[:2]
        
        # Filter out detections that are too small or too large
        refined = []
        min_area = self.config.get('min_vehicle_area', 500)  # Minimum area in pixels
        max_area = (width * height) * 0.3  # Maximum 30% of image area
        
        for det in detections:
            area = det['area']
            if min_area <= area <= max_area:
                # Check if vehicle is likely parked (not moving)
                # Based on aspect ratio and position
                bbox = det['bbox']
                w = bbox[2] - bbox[0]
                h = bbox[1] - bbox[3]
                
                if w > 0 and h > 0:
                    aspect_ratio = w / h
                    # Vehicles typically have aspect ratios between 0.5 and 3.0
                    if 0.5 <= aspect_ratio <= 3.0:
                        refined.append(det)
        
        self.logger.debug(f"Refined from {len(detections)} to {len(refined)} detections")
        return refined

    def batch_detect(self, image_paths: List[str]) -> Dict[str, Dict]:
        """
        Detect vehicles in batch of images
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            Dictionary mapping image paths to detection results
        """
        results = {}
        
        for img_path in image_paths:
            try:
                image = cv2.imread(img_path)
                if image is not None:
                    results[img_path] = self.detect(image)
                else:
                    self.logger.warning(f"Could not load image: {img_path}")
            except Exception as e:
                self.logger.error(f"Error processing {img_path}: {e}")
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            'model_name': self.model.model_name,
            'device': self.device,
            'model_parameters': sum(p.numel() for p in self.model.model.parameters()),
            'config': self.config
        }