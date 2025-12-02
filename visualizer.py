#!/usr/bin/env python3
"""
Visualization Module
Creates comprehensive visualizations for parking lot analysis results

Author: MiniMax Agent
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os
import logging
from datetime import datetime

class Visualizer:
    """
    Creates visualizations for parking lot analysis
    Generates annotated images, charts, and summary reports
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize visualizer"""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Color schemes
        self.colors = {
            'occupied': (0, 0, 255),      # Red
            'vacant': (0, 255, 0),        # Green
            'uncertain': (0, 255, 255),   # Cyan
            'vehicle': (255, 0, 0),       # Blue
            'background': (128, 128, 128) # Gray
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
    
    def create_annotated_image(self, image: np.ndarray, detection_results: Dict) -> np.ndarray:
        """
        Create comprehensive annotated image showing all detection results
        
        Args:
            image: Original image
            detection_results: Complete detection results
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Get labeled spaces
        analyzer_results = detection_results.get('final_detections', {})
        labeled_spaces = analyzer_results.get('labeled_spaces', [])
        
        # Step 1: Draw parking spaces
        annotated = self._draw_parking_spaces(annotated, labeled_spaces)
        
        # Step 2: Draw vehicle detections
        annotated = self._draw_vehicle_detections(annotated, detection_results)
        
        # Step 3: Add statistics overlay
        annotated = self._add_statistics_overlay(annotated, analyzer_results)
        
        # Step 4: Add legend
        annotated = self._add_legend(annotated)
        
        return annotated
    
    def _draw_parking_spaces(self, image: np.ndarray, labeled_spaces: List[Dict]) -> np.ndarray:
        """Draw parking spaces with occupancy status"""
        for space in labeled_spaces:
            bbox = space['bbox']
            status = space['status']
            confidence = space['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on status
            if status == 'occupied':
                color = self.colors['occupied']
                thickness = 3
            elif status == 'vacant':
                color = self.colors['vacant']
                thickness = 2
            else:  # uncertain
                color = self.colors['uncertain']
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Add space label
            label = f"{space['id']}"
            label_color = (0, 0, 0) if status == 'vacant' else (255, 255, 255)
            
            # Add background for label
            (text_w, text_h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            cv2.rectangle(image, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
            
            # Add text
            cv2.putText(image, label, (x1, y1 - 5), self.font, self.font_scale, label_color, self.font_thickness)
            
            # Add confidence score
            conf_label = f"{confidence:.2f}"
            cv2.putText(image, conf_label, (x1, y2 + text_h + 5), self.font, self.font_scale * 0.5, color, 1)
        
        return image
    
    def _draw_vehicle_detections(self, image: np.ndarray, detection_results: Dict) -> np.ndarray:
        """Draw vehicle detections from all models"""
        # Draw YOLO detections
        yolo_results = detection_results.get('yolo_detections', {})
        if yolo_results.get('detections'):
            image = self._draw_yolo_detections(image, yolo_results['detections'])
        
        # Draw Mask R-CNN segments
        maskrcnn_results = detection_results.get('maskrcnn_segments', {})
        if maskrcnn_results.get('segments'):
            image = self._draw_maskrcnn_segments(image, maskrcnn_results['segments'])
        
        # Draw DINO refined detections
        dino_results = detection_results.get('dino_refined', {})
        if dino_results.get('refined_detections'):
            image = self._draw_dino_detections(image, dino_results['refined_detections'])
        
        return image
    
    def _draw_yolo_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw YOLO detection boxes"""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw thin blue box for YOLO
            cv2.rectangle(image, (x1, y1), (x2, y2), self.colors['vehicle'], 1)
            
            # Add confidence score
            label = f"YOLO: {confidence:.2f}"
            cv2.putText(image, label, (x1, y1 - 20), self.font, self.font_scale * 0.5, (255, 255, 255), 1)
        
        return image
    
    def _draw_maskrcnn_segments(self, image: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Draw Mask R-CNN segmentation masks"""
        for segment in segments:
            mask = segment['mask']
            bbox = segment['bbox']
            
            # Create colored overlay
            overlay = np.zeros_like(image, dtype=np.uint8)
            overlay[mask > 0] = (0, 255, 0)  # Green overlay
            
            # Blend with original image
            alpha = 0.3
            image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return image
    
    def _draw_dino_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw DINO refined detections"""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection.get('confidence', 0)
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw thick purple box for DINO
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Add confidence score
            label = f"DINO: {confidence:.2f}"
            cv2.putText(image, label, (x1, y2 + 15), self.font, self.font_scale * 0.5, (255, 0, 255), 1)
        
        return image
    
    def _add_statistics_overlay(self, image: np.ndarray, analyzer_results: Dict) -> np.ndarray:
        """Add statistics overlay to image"""
        # Create semi-transparent overlay
        overlay = np.zeros_like(image)
        
        # Get statistics
        total_spaces = analyzer_results.get('total_spaces', 0)
        occupied_spaces = analyzer_results.get('occupied_spaces', 0)
        vacant_spaces = analyzer_results.get('vacant_spaces', 0)
        occupancy_rate = analyzer_results.get('occupancy_rate', 0)
        
        # Create text
        stats_text = [
            f"Total Spaces: {total_spaces}",
            f"Occupied: {occupied_spaces}",
            f"Vacant: {vacant_spaces}",
            f"Occupancy Rate: {occupancy_rate:.1%}"
        ]
        
        # Position overlay in top-left corner
        overlay_height = len(stats_text) * 30 + 20
        overlay_width = 300
        
        # Draw background
        cv2.rectangle(overlay, (10, 10), (overlay_width, overlay_height), (0, 0, 0), -1)
        
        # Add text
        y_offset = 30
        for text in stats_text:
            cv2.putText(overlay, text, (20, y_offset), self.font, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Blend with original image
        alpha = 0.7
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        return image
    
    def _add_legend(self, image: np.ndarray) -> np.ndarray:
        """Add legend to the image"""
        legend = [
            ("Occupied Space", self.colors['occupied']),
            ("Vacant Space", self.colors['vacant']),
            ("YOLO Detection", self.colors['vehicle']),
            ("DINO Refinement", (255, 0, 255))
        ]
        
        legend_height = len(legend) * 25 + 20
        legend_width = 200
        
        # Position in bottom-right corner
        y_start = image.shape[0] - legend_height - 10
        x_start = image.shape[1] - legend_width - 10
        
        # Draw legend background
        cv2.rectangle(image, (x_start, y_start), (x_start + legend_width, y_start + legend_height), (255, 255, 255), -1)
        cv2.rectangle(image, (x_start, y_start), (x_start + legend_width, y_start + legend_height), (0, 0, 0), 2)
        
        # Add legend items
        y_offset = y_start + 25
        for text, color in legend:
            # Draw color box
            cv2.rectangle(image, (x_start + 10, y_offset - 15), (x_start + 30, y_offset + 5), color, -1)
            
            # Add text
            cv2.putText(image, text, (x_start + 35, y_offset), self.font, 0.5, (0, 0, 0), 1)
            y_offset += 25
        
        return image
    
    def create_summary_visualizations(self, results: Dict, output_path: str):
        """Create comprehensive summary visualizations"""
        self.logger.info("Creating summary visualizations...")
        
        # Create occupancy distribution chart
        self._create_occupancy_distribution_chart(results, output_path)
        
        # Create vacant space heatmap
        self._create_vacant_space_analysis_chart(results, output_path)
        
        # Create confidence analysis chart
        self._create_confidence_analysis_chart(results, output_path)
        
        # Create summary report
        self._create_summary_report(results, output_path)
        
        # Create vacant space list
        self._create_vacant_space_list(results, output_path)
    
    def _create_occupancy_distribution_chart(self, results: Dict, output_path: str):
        """Create occupancy rate distribution chart"""
        occupancy_summary = results.get('occupancy_summary', [])
        
        if not occupancy_summary:
            return
        
        # Extract occupancy rates
        occupancy_rates = [item['occupancy_rate'] for item in occupancy_summary]
        images = [item['image'] for item in occupancy_summary]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(occupancy_rates, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Occupancy Rate')
        ax1.set_ylabel('Number of Images')
        ax1.set_title('Distribution of Occupancy Rates')
        ax1.grid(True, alpha=0.3)
        
        # Line plot
        ax2.plot(range(len(occupancy_rates)), occupancy_rates, marker='o', alpha=0.7)
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Occupancy Rate')
        ax2.set_title('Occupancy Rate Across Images')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/occupancy_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_vacant_space_analysis_chart(self, results: Dict, output_path: str):
        """Create vacant space analysis visualization"""
        occupancy_summary = results.get('occupancy_summary', [])
        
        if not occupancy_summary:
            return
        
        # Extract data
        vacant_counts = [item['vacant_spaces'] for item in occupancy_summary]
        occupied_counts = [item['occupied_spaces'] for item in occupancy_summary]
        images = [item['image'] for item in occupancy_summary]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Stacked bar chart
        x_pos = range(len(images))
        ax1.bar(x_pos, occupied_counts, label='Occupied', color='red', alpha=0.7)
        ax1.bar(x_pos, vacant_counts, bottom=occupied_counts, label='Vacant', color='green', alpha=0.7)
        ax1.set_xlabel('Images')
        ax1.set_ylabel('Number of Spaces')
        ax1.set_title('Parking Space Occupancy by Image')
        ax1.legend()
        ax1.set_xticks(x_pos[::5])  # Show every 5th label
        ax1.set_xticklabels([images[i] for i in x_pos[::5]], rotation=45)
        
        # Vacant space count trend
        ax2.plot(x_pos, vacant_counts, marker='o', color='green', label='Vacant Spaces')
        ax2.set_xlabel('Image Index')
        ax2.set_ylabel('Number of Vacant Spaces')
        ax2.set_title('Vacant Spaces Trend')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/vacant_space_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_analysis_chart(self, results: Dict, output_path: str):
        """Create confidence analysis visualization"""
        occupancy_summary = results.get('occupancy_summary', [])
        
        if not occupancy_summary:
            return
        
        # Create confidence distribution chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Extract confidence data for analysis
        analysis_confidences = []
        for item in occupancy_summary:
            # Calculate average confidence for this image
            vacant_ids = item.get('vacant_space_ids', [])
            if vacant_ids:
                # This would be expanded with actual confidence data
                analysis_confidences.append(0.8)  # Placeholder
        
        if analysis_confidences:
            ax.hist(analysis_confidences, bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Analysis Confidence')
            ax.set_ylabel('Number of Images')
            ax.set_title('Distribution of Analysis Confidence')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No confidence data available', 
                   horizontalalignment='center', verticalalignment='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Confidence Analysis')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_report(self, results: Dict, output_path: str):
        """Create comprehensive summary report"""
        summary = results.get('summary', {})
        occupancy_summary = results.get('occupancy_summary', [])
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Overall statistics pie chart
        if occupancy_summary:
            total_occupied = sum(item['occupied_spaces'] for item in occupancy_summary)
            total_vacant = sum(item['vacant_spaces'] for item in occupancy_summary)
            
            labels = ['Occupied', 'Vacant']
            sizes = [total_occupied, total_vacant]
            colors = ['red', 'green']
            explode = (0.1, 0)
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            ax1.set_title('Overall Space Distribution')
        
        # 2. Occupancy rate statistics
        if occupancy_summary:
            occupancy_rates = [item['occupancy_rate'] for item in occupancy_summary]
            ax2.hist(occupancy_rates, bins=10, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_xlabel('Occupancy Rate')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Occupancy Rate Distribution')
            ax2.axvline(np.mean(occupancy_rates), color='red', linestyle='--', label=f'Mean: {np.mean(occupancy_rates):.2%}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Processing statistics
        processing_stats = {
            'Total Images': summary.get('total_images_processed', 0),
            'Total Detections': summary.get('total_detections', 0),
            'Avg Detections/Image': summary.get('total_detections', 0) / max(summary.get('total_images_processed', 1), 1)
        }
        
        ax3.bar(processing_stats.keys(), processing_stats.values(), color=['skyblue', 'lightgreen', 'orange'])
        ax3.set_title('Processing Statistics')
        ax3.set_ylabel('Count')
        
        # 4. Occupancy categories
        if 'occupancy_distribution' in summary:
            dist = summary['occupancy_distribution']
            categories = ['Low (<30%)', 'Medium (30-70%)', 'High (>70%)']
            counts = [dist['occupancy_categories']['low'], 
                     dist['occupancy_categories']['medium'], 
                     dist['occupancy_categories']['high']]
            colors = ['lightgreen', 'yellow', 'lightcoral']
            
            ax4.bar(categories, counts, color=colors)
            ax4.set_title('Occupancy Category Distribution')
            ax4.set_ylabel('Number of Images')
            ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/summary_report.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_vacant_space_list(self, results: Dict, output_path: str):
        """Create detailed list of vacant spaces"""
        vacant_mappings = results.get('vacant_space_mappings', {})
        
        # Create DataFrame
        vacant_data = []
        for image_name, data in vacant_mappings.items():
            vacant_data.append({
                'Image': image_name,
                'Total Spaces': data['total_spaces'],
                'Vacant Count': data['vacant_count'],
                'Vacancy Rate': f"{data['vacant_count']/data['total_spaces']:.1%}" if data['total_spaces'] > 0 else "0%",
                'Occupancy Status': data['occupancy_status'],
                'Vacant Space Labels': ', '.join(data['vacant_space_labels']) if data['vacant_space_labels'] else 'None'
            })
        
        if vacant_data:
            df = pd.DataFrame(vacant_data)
            df.to_csv(f"{output_path}/vacant_spaces_detailed.csv", index=False)
            
            # Create a summary table visualization
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = df.values
            headers = df.columns.tolist()
            
            table = ax.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(headers)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            plt.title('Detailed Vacant Space Analysis', fontsize=16, fontweight='bold', pad=20)
            plt.savefig(f"{output_path}/vacant_spaces_table.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_comparison_visualization(self, results_before: Dict, results_after: Dict, output_path: str):
        """Create before/after comparison visualization"""
        # This would create side-by-side comparisons
        # Implementation depends on specific comparison requirements
        pass
    
    def save_individual_annotations(self, image_results: Dict, image_path: str, output_path: str):
        """Save individual annotations for each image"""
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Create detailed annotation file
        annotation_file = os.path.join(output_path, f"{image_name}_annotation.txt")
        
        with open(annotation_file, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write occupancy details
            labeled_spaces = image_results.get('labeled_spaces', [])
            
            f.write("PARKING SPACE ANALYSIS:\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Spaces: {image_results.get('total_spaces', 0)}\n")
            f.write(f"Occupied: {image_results.get('occupied_spaces', 0)}\n")
            f.write(f"Vacant: {image_results.get('vacant_spaces', 0)}\n")
            f.write(f"Occupancy Rate: {image_results.get('occupancy_rate', 0):.2%}\n\n")
            
            # List vacant spaces
            vacant_spaces = [s for s in labeled_spaces if s['status'] == 'vacant']
            f.write("VACANT SPACES:\n")
            f.write("-" * 20 + "\n")
            for space in vacant_spaces:
                f.write(f"Space ID: {space['id']}\n")
                f.write(f"  Location: {space['center']}\n")
                f.write(f"  Bounding Box: {space['bbox']}\n")
                f.write(f"  Confidence: {space['confidence']:.3f}\n")
                f.write(f"  Area: {space['area']:.0f} pixels²\n\n")
            
            # List occupied spaces
            occupied_spaces = [s for s in labeled_spaces if s['status'] == 'occupied']
            f.write("OCCUPIED SPACES:\n")
            f.write("-" * 20 + "\n")
            for space in occupied_spaces:
                f.write(f"Space ID: {space['id']}\n")
                f.write(f"  Location: {space['center']}\n")
                f.write(f"  Confidence: {space['confidence']:.3f}\n\n")
        
        self.logger.info(f"Detailed annotations saved to {annotation_file}")
    
    def create_vacancy_summary_report(self, results: Dict, output_path: str) -> str:
        """Create a concise vacancy summary report"""
        summary = results.get('summary', {})
        vacant_mappings = results.get('vacant_space_mappings', {})
        
        report_lines = [
            "PARKING LOT VACANCY ANALYSIS REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Images Analyzed: {summary.get('total_images_processed', 0)}",
            f"Average Occupancy Rate: {summary.get('average_occupancy_rate', 0):.2%}",
            "",
            "VACANT SPACE SUMMARY:",
            "-" * 30
        ]
        
        # Add summary statistics
        if vacant_mappings:
            total_vacant = sum(data['vacant_count'] for data in vacant_mappings.values())
            total_spaces = sum(data['total_spaces'] for data in vacant_mappings.values())
            
            report_lines.extend([
                f"Total Vacant Spaces Across All Images: {total_vacant}",
                f"Total Parking Spaces Across All Images: {total_spaces}",
                f"Average Vacancy Rate: {total_vacant/total_spaces:.2%}" if total_spaces > 0 else "Average Vacancy Rate: 0%",
                "",
                "DETAILED VACANT SPACE LIST:",
                "-" * 35
            ])
            
            # List images with highest vacancy
            sorted_images = sorted(vacant_mappings.items(), 
                                 key=lambda x: x[1]['vacant_count'], reverse=True)
            
            for image_name, data in sorted_images[:10]:  # Top 10
                report_lines.append(f"{image_name}:")
                report_lines.append(f"  Total Spaces: {data['total_spaces']}")
                report_lines.append(f"  Vacant: {data['vacant_count']} ({data['vacant_count']/data['total_spaces']:.1%})")
                report_lines.append(f"  Status: {data['occupancy_status']}")
                if data['vacant_space_labels']:
                    report_lines.append(f"  Vacant Space IDs: {', '.join(data['vacant_space_labels'][:5])}" + 
                                      (" ..." if len(data['vacant_space_labels']) > 5 else ""))
                report_lines.append("")
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = os.path.join(output_path, "vacancy_summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Vacancy summary report saved to {report_path}")
        return report_content