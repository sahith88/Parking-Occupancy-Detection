#!/usr/bin/env python3
"""
Mask R-CNN Instance Segmentation Module
Using Detectron2 for precise vehicle segmentation

Author: MiniMax Agent
"""

import torch
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from typing import Dict, List, Tuple, Any
import logging
import os

class MaskRCNNSegmenter:
    """Mask R-CNN-based instance segmentation for precise vehicle boundaries"""
    
    def __init__(self, config: Dict):
        """
        Initialize Mask R-CNN segmenter
        
        Args:
            config: Mask R-CNN configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup configuration
        self.cfg = self._setup_config()
        
        # Initialize predictor
        try:
            self.predictor = DefaultPredictor(self.cfg)
            self.logger.info("Mask R-CNN model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Mask R-CNN model: {e}")
            raise
    
    def _setup_config(self) -> Any:
        """Setup Detectron2 configuration"""
        cfg = get_cfg()
        
        # Add project-specific config
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # Model weights
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        # Set device
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.config.get('confidence_threshold', 0.5)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.get('confidence_threshold', 0.5)
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.config.get('nms_threshold', 0.3)
        
        # Input settings
        cfg.INPUT.MIN_SIZE_TEST = self.config.get('min_size', 800)
        cfg.INPUT.MAX_SIZE_TEST = self.config.get('max_size', 1333)
        
        # Batch size
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config.get('batch_size_per_image', 512)
        
        return cfg
    
    def segment(self, image: np.ndarray, yolo_results: Dict) -> Dict[str, Any]:
        """
        Perform instance segmentation on image using Mask R-CNN
        
        Args:
            image: Input image as numpy array
            yolo_results: Results from YOLO detection to guide segmentation
            
        Returns:
            Dictionary containing segmentation results
        """
        # Run inference
        outputs = self.predictor(image)
        
        # Extract instances
        instances = outputs["instances"].to(self.cfg.MODEL.DEVICE)
        
        # Filter for vehicle classes
        vehicle_instances = self._filter_vehicle_instances(instances)
        
        # Process segmentation masks
        segments = []
        for i, instance in enumerate(vehicle_instances):
            segment = self._process_instance(instance, i)
            segments.append(segment)
        
        # Apply spatial refinement using YOLO detections
        refined_segments = self._refine_with_yolo(segments, yolo_results)
        
        result_dict = {
            'segments': refined_segments,
            'mask_shape': image.shape,
            'segment_count': len(refined_segments),
            'raw_instances': len(vehicle_instances),
            'processing_info': {
                'model': 'Mask R-CNN (Detectron2)',
                'device': self.cfg.MODEL.DEVICE,
                'confidence_threshold': self.config.get('confidence_threshold', 0.5)
            }
        }
        
        self.logger.debug(f"Mask R-CNN generated {len(refined_segments)} segments")
        return result_dict
    
    def _filter_vehicle_instances(self, instances: Any) -> List:
        """Filter instances for vehicle classes only"""
        vehicle_classes = self.config.get('vehicle_classes', [2, 3, 5, 7])  # COCO vehicle classes
        
        # Get predicted classes
        pred_classes = instances.pred_classes.cpu().numpy()
        
        # Filter vehicle instances
        vehicle_indices = [i for i, cls in enumerate(pred_classes) if cls in vehicle_classes]
        vehicle_instances = [instances[i] for i in vehicle_indices]
        
        # Filter by confidence
        confidence_threshold = self.config.get('confidence_threshold', 0.5)
        confident_instances = []
        
        for instance in vehicle_instances:
            if instance.scores > confidence_threshold:
                confident_instances.append(instance)
        
        return confident_instances
    
    def _process_instance(self, instance: Any, index: int) -> Dict[str, Any]:
        """Process a single instance to extract segmentation information"""
        # Get bounding box
        bbox = instance.pred_boxes.tensor.cpu().numpy()[0]
        
        # Get mask
        mask = instance.pred_masks.cpu().numpy()
        
        # Get class information
        class_id = int(instance.pred_classes.cpu().numpy())
        confidence = float(instance.scores.cpu().numpy())
        
        # Calculate mask statistics
        mask_2d = mask[0] if len(mask.shape) == 3 else mask
        mask_area = np.sum(mask_2d)
        
        # Convert bbox to [x1, y1, x2, y2] format
        x1, y1, x2, y2 = bbox
        
        segment = {
            'segment_id': index,
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'mask': mask_2d,
            'mask_area': int(mask_area),
            'class_id': class_id,
            'confidence': confidence,
            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
            'area': float((x2 - x1) * (y2 - y1))
        }
        
        return segment
    
    def _refine_with_yolo(self, segments: List[Dict], yolo_results: Dict) -> List[Dict]:
        """
        Refine segments using YOLO detection results
        
        Args:
            segments: List of segmentation results
            yolo_results: YOLO detection results
            
        Returns:
            Refined segments
        """
        if not yolo_results.get('detections'):
            return segments
        
        refined_segments = []
        yolo_detections = yolo_results['detections']
        
        for segment in segments:
            seg_bbox = segment['bbox']
            
            # Find matching YOLO detection
            best_match = None
            best_iou = 0
            
            for det in yolo_detections:
                iou = self._calculate_iou(seg_bbox, det['bbox'])
                
                if iou > best_iou and iou > self.config.get('matching_iou_threshold', 0.3):
                    best_iou = iou
                    best_match = det
            
            # Add YOLO information to segment if match found
            if best_match:
                segment['yolo_bbox'] = best_match['bbox']
                segment['yolo_confidence'] = best_match['confidence']
                segment['iou_with_yolo'] = best_iou
                segment['refined'] = True
            else:
                segment['refined'] = False
            
            refined_segments.append(segment)
        
        return refined_segments
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi1 >= xi2 or yi1 >= yi2:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def create_segmentation_mask(self, segments: List[Dict], image_shape: Tuple) -> np.ndarray:
        """
        Create a combined segmentation mask from all segments
        
        Args:
            segments: List of segmentation results
            image_shape: Shape of the original image
            
        Returns:
            Combined segmentation mask
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        for i, segment in enumerate(segments):
            segment_mask = segment['mask']
            
            # Ensure mask matches image size
            if segment_mask.shape[:2] != image_shape[:2]:
                segment_mask = cv2.resize(segment_mask, (image_shape[1], image_shape[0]))
            
            # Add to combined mask with different values for each segment
            mask[segment_mask > 0] = (i + 1) * 50  # Different intensity for each segment
        
        return mask
    
    def get_segment_statistics(self, segments: List[Dict]) -> Dict[str, Any]:
        """Get statistics about the segmentation results"""
        if not segments:
            return {'total_segments': 0}
        
        areas = [seg['mask_area'] for seg in segments]
        confidences = [seg['confidence'] for seg in segments]
        
        return {
            'total_segments': len(segments),
            'average_mask_area': np.mean(areas),
            'min_mask_area': np.min(areas),
            'max_mask_area': np.max(areas),
            'total_mask_area': np.sum(areas),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'segments_by_confidence': {
                'high': len([c for c in confidences if c >= 0.8]),
                'medium': len([c for c in confidences if 0.5 <= c < 0.8]),
                'low': len([c for c in confidences if c < 0.5])
            }
        }
    
    def visualize_segments(self, image: np.ndarray, segments: List[Dict], output_path: str = None):
        """
        Create visualization of segmentation results
        
        Args:
            image: Original image
            segments: Segmentation results
            output_path: Path to save visualization
        """
        # Create visualizer
        metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0] if self.cfg.DATASETS.TRAIN else "coco_2017_train")
        
        # Create instances
        instances = self._segments_to_instances(segments)
        
        # Visualize
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.2)
        out = v.draw_instance_predictions(instances.to("cpu"))
        
        result_image = out.get_image()[:, :, ::-1]
        
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        return result_image
    
    def _segments_to_instances(self, segments: List[Dict]):
        """Convert segments back to Detectron2 instances format"""
        from detectron2.structures import Instances, Boxes, BitMasks
        
        if not segments:
            return Instances()
        
        # Extract data
        boxes = torch.tensor([seg['bbox'] for seg in segments])
        scores = torch.tensor([seg['confidence'] for seg in segments])
        classes = torch.tensor([seg['class_id'] for seg in segments])
        
        # Create masks
        masks = []
        for seg in segments:
            mask = seg['mask']
            if len(mask.shape) == 2:
                mask = mask[None]
            masks.append(mask)
        
        # Stack masks
        masks_tensor = torch.tensor(np.stack(masks))
        
        # Create instances
        instances = Instances(image_size=(int(max(seg['mask'].shape)), int(max(seg['mask'].shape))))
        instances.set("pred_boxes", Boxes(boxes))
        instances.set("pred_classes", classes)
        instances.set("scores", scores)
        instances.set("pred_masks", BitMasks(masks_tensor))
        
        return instances