#!/usr/bin/env python3
"""
Parking Space Analysis Module
Core logic for determining parking space occupancy and vacant space identification

Author: MiniMax Agent
"""

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy import ndimage
from sklearn.cluster import DBSCAN
import math

class ParkingSpaceAnalyzer:
    """
    Analyzes parking lot occupancy based on detection results
    Determines which specific parking spaces are vacant/occupied
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize parking space analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Parking space grid
        self.parking_spaces = []
        self.space_labels = []
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'parking_space_detection': {
                'grid_spacing_x': 100,  # pixels between parking spaces horizontally
                'grid_spacing_y': 80,   # pixels between parking spaces vertically
                'space_width': 80,      # typical parking space width
                'space_height': 120,    # typical parking space height
                'min_space_area': 3000, # minimum area to consider as parking space
                'overlap_threshold': 0.3 # overlap threshold for space assignment
            },
            'occupancy_detection': {
                'vehicle_overlap_threshold': 0.5,  # percentage of space occupied by vehicle
                'empty_space_threshold': 0.1,      # maximum vehicle overlap for empty space
                'confidence_threshold': 0.6        # minimum confidence for occupancy decision
            },
            'clustering': {
                'eps': 50,  # maximum distance between parking spaces
                'min_samples': 2  # minimum samples for DBSCAN clustering
            }
        }
    
    def analyze_image_occupancy(self, detection_results: Dict) -> Dict[str, Any]:
        """
        Analyze occupancy for a single image
        
        Args:
            detection_results: Complete detection results from pipeline
            
        Returns:
            Dictionary containing occupancy analysis
        """
        # Extract detection information
        detections = detection_results.get('final_detections', [])
        
        if not detections:
            self.logger.warning("No detections found for occupancy analysis")
            return self._empty_analysis_result()
        
        # Step 1: Detect parking spaces
        image_shape = detection_results.get('maskrcnn_segments', {}).get('mask_shape', (480, 640))
        parking_spaces = self._detect_parking_spaces(image_shape, detection_results)
        
        # Step 2: Assign vehicles to parking spaces
        space_assignments = self._assign_vehicles_to_spaces(detections, parking_spaces)
        
        # Step 3: Determine occupancy status
        occupancy_results = self._determine_occupancy(space_assignments, parking_spaces)
        
        # Step 4: Generate space labels and mappings
        labeled_spaces = self._generate_space_labels(occupancy_results, parking_spaces)
        
        # Step 5: Calculate statistics
        stats = self._calculate_statistics(occupancy_results, labeled_spaces)
        
        analysis_result = {
            'total_spaces': len(parking_spaces),
            'occupied_spaces': len(occupancy_results['occupied']),
            'vacant_spaces': len(occupancy_results['vacant']),
            'occupancy_rate': len(occupancy_results['occupied']) / len(parking_spaces) if parking_spaces else 0,
            'detections_used': len(detections),
            'space_assignments': space_assignments,
            'occupancy_results': occupancy_results,
            'labeled_spaces': labeled_spaces,
            'parking_spaces': parking_spaces,
            'vacant_space_ids': [space['id'] for space in labeled_spaces if space['status'] == 'vacant'],
            'occupied_space_ids': [space['id'] for space in labeled_spaces if space['status'] == 'occupied'],
            'total_detections': len(detections),
            'analysis_confidence': self._calculate_analysis_confidence(space_assignments, detections)
        }
        
        self.logger.debug(f"Occupancy analysis: {analysis_result['occupied_spaces']}/{analysis_result['total_spaces']} occupied")
        return analysis_result
    
    def _detect_parking_spaces(self, image_shape: Tuple, detection_results: Dict) -> List[Dict]:
        """
        Detect parking spaces in the image
        
        Args:
            image_shape: Shape of the image (height, width, channels)
            detection_results: Detection results from pipeline
            
        Returns:
            List of detected parking spaces
        """
        height, width = image_shape[:2]
        
        # Method 1: Extract parking spaces from segmentation if available
        segments = detection_results.get('maskrcnn_segments', {}).get('segments', [])
        if segments:
            spaces = self._extract_spaces_from_segments(segments, height, width)
            if len(spaces) > 0:
                return self._refine_spaces_with_clustering(spaces)
        
        # Method 2: Generate parking spaces using grid-based approach
        spaces = self._generate_grid_spaces(height, width)
        
        return spaces
    
    def _extract_spaces_from_segments(self, segments: List[Dict], height: int, width: int) -> List[Dict]:
        """Extract parking spaces from segmentation results"""
        spaces = []
        
        # Analyze segment sizes to identify potential parking spaces
        for i, segment in enumerate(segments):
            bbox = segment['bbox']
            x1, y1, x2, y2 = bbox
            space_width = x2 - x1
            space_height = y2 - y1
            
            # Check if segment size matches parking space dimensions
            if (self.config['parking_space_detection']['min_space_area'] <= 
                space_width * space_height <= 
                (width * height) * 0.1):  # Max 10% of image area
                
                space = {
                    'id': f'space_{i}',
                    'bbox': [x1, y1, x2, y2],
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2],
                    'area': space_width * space_height,
                    'dimensions': [space_width, space_height],
                    'source': 'segment',
                    'confidence': segment.get('confidence', 0.5)
                }
                spaces.append(space)
        
        return spaces
    
    def _generate_grid_spaces(self, height: int, width: int) -> List[Dict]:
        """Generate parking spaces using grid-based approach"""
        spaces = []
        config = self.config['parking_space_detection']
        
        # Grid parameters
        space_width = config['space_width']
        space_height = config['space_height']
        spacing_x = config['grid_spacing_x']
        spacing_y = config['grid_spacing_y']
        
        # Calculate grid positions
        x_positions = list(range(space_width//2, width - space_width//2, spacing_x))
        y_positions = list(range(space_height//2, height - space_height//2, spacing_y))
        
        space_id = 0
        for y in y_positions:
            for x in x_positions:
                # Define parking space boundaries
                x1 = max(0, x - space_width // 2)
                x2 = min(width, x + space_width // 2)
                y1 = max(0, y - space_height // 2)
                y2 = min(height, y + space_height // 2)
                
                # Check space area
                space_area = (x2 - x1) * (y2 - y1)
                if space_area >= config['min_space_area']:
                    space = {
                        'id': f'space_{space_id:03d}',
                        'bbox': [x1, y1, x2, y2],
                        'center': [x, y],
                        'area': space_area,
                        'dimensions': [space_width, space_height],
                        'source': 'grid'
                    }
                    spaces.append(space)
                    space_id += 1
        
        return spaces
    
    def _refine_spaces_with_clustering(self, spaces: List[Dict]) -> List[Dict]:
        """Refine parking spaces using clustering"""
        if len(spaces) < 2:
            return spaces
        
        # Extract center coordinates for clustering
        centers = np.array([[space['center'][0], space['center'][1]] for space in spaces])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.config['clustering']['eps'], 
                          min_samples=self.config['clustering']['min_samples'])
        cluster_labels = clustering.fit_predict(centers)
        
        # Reorganize spaces by clusters
        clustered_spaces = {}
        for i, label in enumerate(cluster_labels):
            if label not in clustered_spaces:
                clustered_spaces[label] = []
            clustered_spaces[label].append(spaces[i])
        
        # Keep spaces from the largest cluster
        if clustered_spaces:
            largest_cluster = max(clustered_spaces.values(), key=len)
            refined_spaces = self._filter_spaces_by_density(largest_cluster)
            return refined_spaces
        
        return spaces
    
    def _filter_spaces_by_density(self, spaces: List[Dict]) -> List[Dict]:
        """Filter spaces based on density and consistency"""
        if len(spaces) < 2:
            return spaces
        
        # Calculate areas and dimensions
        areas = [space['area'] for space in spaces]
        widths = [space['dimensions'][0] for space in spaces]
        heights = [space['dimensions'][1] for space in spaces]
        
        # Calculate median values
        median_area = np.median(areas)
        median_width = np.median(widths)
        median_height = np.median(heights)
        
        # Filter out outliers
        filtered_spaces = []
        for space in spaces:
            area_ratio = abs(space['area'] - median_area) / median_area
            width_ratio = abs(space['dimensions'][0] - median_width) / median_width
            height_ratio = abs(space['dimensions'][1] - median_height) / median_height
            
            # Keep spaces that are within 50% of median values
            if (area_ratio < 0.5 and width_ratio < 0.5 and height_ratio < 0.5):
                filtered_spaces.append(space)
        
        return filtered_spaces if filtered_spaces else spaces
    
    def _assign_vehicles_to_spaces(self, detections: List[Dict], parking_spaces: List[Dict]) -> List[Dict]:
        """
        Assign detected vehicles to parking spaces
        
        Args:
            detections: List of vehicle detections
            parking_spaces: List of parking spaces
            
        Returns:
            List of space assignments
        """
        assignments = []
        
        for space in parking_spaces:
            space_bbox = space['bbox']
            assigned_vehicles = []
            
            for detection in detections:
                if detection.get('suppressed', False):
                    continue
                
                vehicle_bbox = detection['bbox']
                overlap_ratio = self._calculate_overlap_ratio(space_bbox, vehicle_bbox)
                
                if overlap_ratio >= self.config['occupancy_detection']['vehicle_overlap_threshold']:
                    assigned_vehicles.append({
                        'detection': detection,
                        'overlap_ratio': overlap_ratio,
                        'confidence': detection.get('confidence', 0.0)
                    })
            
            assignment = {
                'space_id': space['id'],
                'space_bbox': space_bbox,
                'assigned_vehicles': assigned_vehicles,
                'assignment_confidence': max([v['confidence'] for v in assigned_vehicles], default=0.0)
            }
            assignments.append(assignment)
        
        return assignments
    
    def _calculate_overlap_ratio(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
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
        
        # Calculate areas
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Return overlap ratio relative to smaller area
        smaller_area = min(area1, area2)
        return intersection / smaller_area if smaller_area > 0 else 0.0
    
    def _determine_occupancy(self, assignments: List[Dict], parking_spaces: List[Dict]) -> Dict[str, List[Dict]]:
        """Determine which spaces are occupied/vacant"""
        occupied = []
        vacant = []
        uncertain = []
        
        for assignment in assignments:
            space_id = assignment['space_id']
            assigned_vehicles = assignment['assigned_vehicles']
            assignment_conf = assignment['assignment_confidence']
            
            # Determine occupancy based on assigned vehicles
            if len(assigned_vehicles) == 0:
                # No vehicles assigned
                vacant.append({
                    'space_id': space_id,
                    'space_bbox': assignment['space_bbox'],
                    'confidence': 1.0 - assignment_conf,
                    'reason': 'no_vehicles_assigned'
                })
            elif len(assigned_vehicles) == 1:
                # Single vehicle assigned
                vehicle = assigned_vehicles[0]
                if (vehicle['overlap_ratio'] >= self.config['occupancy_detection']['vehicle_overlap_threshold'] and
                    vehicle['confidence'] >= self.config['occupancy_detection']['confidence_threshold']):
                    occupied.append({
                        'space_id': space_id,
                        'space_bbox': assignment['space_bbox'],
                        'vehicle': vehicle,
                        'confidence': vehicle['confidence'],
                        'reason': 'single_vehicle_high_overlap'
                    })
                else:
                    uncertain.append({
                        'space_id': space_id,
                        'space_bbox': assignment['space_bbox'],
                        'assigned_vehicles': assigned_vehicles,
                        'confidence': vehicle['confidence'],
                        'reason': 'single_vehicle_low_overlap'
                    })
            else:
                # Multiple vehicles assigned
                total_confidence = sum(v['confidence'] for v in assigned_vehicles)
                avg_overlap = np.mean([v['overlap_ratio'] for v in assigned_vehicles])
                
                if avg_overlap >= self.config['occupancy_detection']['vehicle_overlap_threshold']:
                    occupied.append({
                        'space_id': space_id,
                        'space_bbox': assignment['space_bbox'],
                        'vehicles': assigned_vehicles,
                        'confidence': min(1.0, total_confidence / len(assigned_vehicles)),
                        'reason': 'multiple_vehicles_high_overlap'
                    })
                else:
                    uncertain.append({
                        'space_id': space_id,
                        'space_bbox': assignment['space_bbox'],
                        'assigned_vehicles': assigned_vehicles,
                        'confidence': avg_overlap,
                        'reason': 'multiple_vehicles_low_overlap'
                    })
        
        return {
            'occupied': occupied,
            'vacant': vacant,
            'uncertain': uncertain
        }
    
    def _generate_space_labels(self, occupancy_results: Dict, parking_spaces: List[Dict]) -> List[Dict]:
        """Generate labeled spaces with detailed information"""
        labeled_spaces = []
        
        # Create space lookup
        space_lookup = {space['id']: space for space in parking_spaces}
        
        # Process occupied spaces
        for occupied_space in occupancy_results['occupied']:
            space_id = occupied_space['space_id']
            space = space_lookup.get(space_id, {})
            
            labeled_space = {
                'id': space_id,
                'status': 'occupied',
                'confidence': occupied_space['confidence'],
                'reason': occupied_space['reason'],
                'bbox': occupied_space['space_bbox'],
                'center': space.get('center', []),
                'area': space.get('area', 0),
                'label': f"{space_id} (Occupied)",
                'occupancy_percentage': 100.0,
                'detection_details': occupied_space
            }
            labeled_spaces.append(labeled_space)
        
        # Process vacant spaces
        for vacant_space in occupancy_results['vacant']:
            space_id = vacant_space['space_id']
            space = space_lookup.get(space_id, {})
            
            labeled_space = {
                'id': space_id,
                'status': 'vacant',
                'confidence': vacant_space['confidence'],
                'reason': vacant_space['reason'],
                'bbox': vacant_space['space_bbox'],
                'center': space.get('center', []),
                'area': space.get('area', 0),
                'label': f"{space_id} (Vacant)",
                'occupancy_percentage': 0.0,
                'detection_details': vacant_space
            }
            labeled_spaces.append(labeled_space)
        
        # Process uncertain spaces
        for uncertain_space in occupancy_results['uncertain']:
            space_id = uncertain_space['space_id']
            space = space_lookup.get(space_id, {})
            
            labeled_space = {
                'id': space_id,
                'status': 'uncertain',
                'confidence': uncertain_space['confidence'],
                'reason': uncertain_space['reason'],
                'bbox': uncertain_space['space_bbox'],
                'center': space.get('center', []),
                'area': space.get('area', 0),
                'label': f"{space_id} (Uncertain)",
                'occupancy_percentage': uncertain_space['confidence'] * 100.0,
                'detection_details': uncertain_space
            }
            labeled_spaces.append(labeled_space)
        
        return labeled_spaces
    
    def _calculate_statistics(self, occupancy_results: Dict, labeled_spaces: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        total_spaces = len(labeled_spaces)
        occupied = len(occupancy_results['occupied'])
        vacant = len(occupancy_results['vacant'])
        uncertain = len(occupancy_results['uncertain'])
        
        # Calculate confidence statistics
        all_confidences = [space['confidence'] for space in labeled_spaces]
        occupied_confidences = [space['confidence'] for space in labeled_spaces if space['status'] == 'occupied']
        vacant_confidences = [space['confidence'] for space in labeled_spaces if space['status'] == 'vacant']
        
        return {
            'total_spaces': total_spaces,
            'occupied_spaces': occupied,
            'vacant_spaces': vacant,
            'uncertain_spaces': uncertain,
            'occupancy_rate': occupied / total_spaces if total_spaces > 0 else 0,
            'vacancy_rate': vacant / total_spaces if total_spaces > 0 else 0,
            'uncertainty_rate': uncertain / total_spaces if total_spaces > 0 else 0,
            'confidence_stats': {
                'overall': {
                    'mean': np.mean(all_confidences) if all_confidences else 0,
                    'median': np.median(all_confidences) if all_confidences else 0,
                    'std': np.std(all_confidences) if all_confidences else 0
                },
                'occupied': {
                    'mean': np.mean(occupied_confidences) if occupied_confidences else 0,
                    'median': np.median(occupied_confidences) if occupied_confidences else 0,
                    'std': np.std(occupied_confidences) if occupied_confidences else 0
                },
                'vacant': {
                    'mean': np.mean(vacant_confidences) if vacant_confidences else 0,
                    'median': np.median(vacant_confidences) if vacant_confidences else 0,
                    'std': np.std(vacant_confidences) if vacant_confidences else 0
                }
            }
        }
    
    def _calculate_analysis_confidence(self, assignments: List[Dict], detections: List[Dict]) -> float:
        """Calculate overall confidence in the analysis"""
        if not assignments:
            return 0.0
        
        # Factors contributing to confidence
        factors = []
        
        # Factor 1: Detection confidence
        if detections:
            avg_detection_conf = np.mean([d.get('confidence', 0) for d in detections])
            factors.append(avg_detection_conf)
        
        # Factor 2: Assignment confidence
        assignment_confidences = [a['assignment_confidence'] for a in assignments]
        avg_assignment_conf = np.mean(assignment_confidences)
        factors.append(avg_assignment_conf)
        
        # Factor 3: Space coverage
        spaces_with_assignments = len([a for a in assignments if a['assigned_vehicles']])
        coverage_ratio = spaces_with_assignments / len(assignments)
        factors.append(coverage_ratio)
        
        # Factor 4: Consistency (fewer uncertain spaces = higher confidence)
        uncertain_count = len([a for a in assignments if len(a['assigned_vehicles']) != 1])
        uncertainty_penalty = uncertain_count / len(assignments)
        factors.append(1 - uncertainty_penalty)
        
        return np.mean(factors) if factors else 0.0
    
    def _empty_analysis_result(self) -> Dict[str, Any]:
        """Return empty analysis result when no detections available"""
        return {
            'total_spaces': 0,
            'occupied_spaces': 0,
            'vacant_spaces': 0,
            'occupancy_rate': 0.0,
            'detections_used': 0,
            'space_assignments': [],
            'occupancy_results': {'occupied': [], 'vacant': [], 'uncertain': []},
            'labeled_spaces': [],
            'vacant_space_ids': [],
            'occupied_space_ids': [],
            'total_detections': 0,
            'analysis_confidence': 0.0
        }
    
    def get_vacant_space_summary(self, analysis_result: Dict) -> Dict[str, Any]:
        """Get summary specifically focused on vacant spaces"""
        vacant_spaces = [space for space in analysis_result['labeled_spaces'] if space['status'] == 'vacant']
        
        return {
            'total_vacant': len(vacant_spaces),
            'vacant_space_list': [space['id'] for space in vacant_spaces],
            'vacant_space_details': [
                {
                    'id': space['id'],
                    'location': space['center'],
                    'bbox': space['bbox'],
                    'area': space['area'],
                    'confidence': space['confidence']
                }
                for space in vacant_spaces
            ],
            'high_confidence_vacant': len([s for s in vacant_spaces if s['confidence'] > 0.8]),
            'low_confidence_vacant': len([s for s in vacant_spaces if s['confidence'] < 0.6])
        }