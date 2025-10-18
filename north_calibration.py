import cv2
import numpy as np
import math
from typing import Tuple, Optional, List, Dict
from core_types import LatLon
from geo import haversine_m, latlon_to_xy_m, xy_m_to_latlon

class NorthCalibration:
    """
    Advanced north direction calibration for drone navigation.
    Handles automatic north detection and camera orientation.
    """
    
    def __init__(self, 
                 image_width: int = 1920,
                 image_height: int = 1080,
                 camera_angle_deg: float = 60.0):
        """
        Args:
            image_width: Image width in pixels
            image_height: Image height in pixels
            camera_angle_deg: Camera angle relative to ground
        """
        self.image_width = image_width
        self.image_height = image_height
        self.camera_angle_deg = camera_angle_deg
        
        # North direction parameters
        self.north_orientation_deg = 0.0  # Current north orientation
        self.north_confidence = 0.0  # Confidence in north direction
        self.north_history = []
        self.max_history = 20
        
        # Building-based north detection
        self.building_north_detector = BuildingNorthDetector()
        
        # Sun/shadow-based north detection
        self.shadow_north_detector = ShadowNorthDetector()
        
        # Road-based north detection
        self.road_north_detector = RoadNorthDetector()
        
    def detect_north_direction(self, 
                             frame: np.ndarray, 
                             buildings: List[Dict] = None,
                             roads: List[Dict] = None) -> Tuple[float, float]:
        """
        Detect north direction using multiple methods.
        Returns: (north_angle_deg, confidence)
        """
        north_angles = []
        confidences = []
        
        # Method 1: Building-based north detection
        if buildings:
            building_north, building_conf = self.building_north_detector.detect_north(
                frame, buildings
            )
            if building_north is not None:
                north_angles.append(building_north)
                confidences.append(building_conf)
        
        # Method 2: Shadow-based north detection
        shadow_north, shadow_conf = self.shadow_north_detector.detect_north(frame)
        if shadow_north is not None:
            north_angles.append(shadow_north)
            confidences.append(shadow_conf)
        
        # Method 3: Road-based north detection
        if roads:
            road_north, road_conf = self.road_north_detector.detect_north(frame, roads)
            if road_north is not None:
                north_angles.append(road_north)
                confidences.append(road_conf)
        
        # Combine results
        if not north_angles:
            return self.north_orientation_deg, 0.0
        
        # Weighted average of north angles
        total_weight = sum(confidences)
        if total_weight == 0:
            return self.north_orientation_deg, 0.0
        
        weighted_north = sum(angle * conf for angle, conf in zip(north_angles, confidences))
        weighted_north /= total_weight
        
        # Normalize to 0-360 degrees
        weighted_north = weighted_north % 360
        
        # Update north orientation with smoothing
        if self.north_orientation_deg is not None:
            # Smooth transition to new north direction
            alpha = min(0.3, total_weight / 3.0)  # Adaptive smoothing
            self.north_orientation_deg = (
                self.north_orientation_deg * (1 - alpha) + 
                weighted_north * alpha
            )
        else:
            self.north_orientation_deg = weighted_north
        
        # Update confidence
        self.north_confidence = min(1.0, total_weight / 3.0)
        
        # Store in history
        self.north_history.append((self.north_orientation_deg, self.north_confidence))
        if len(self.north_history) > self.max_history:
            self.north_history.pop(0)
        
        return self.north_orientation_deg, self.north_confidence
    
    def get_north_vector(self) -> Tuple[float, float]:
        """Get north direction vector in image coordinates."""
        if self.north_orientation_deg is None:
            return (0.0, -1.0)  # Default: up is north
        
        # Convert angle to vector
        angle_rad = math.radians(self.north_orientation_deg)
        north_x = math.sin(angle_rad)
        north_y = -math.cos(angle_rad)  # Negative because image Y is downward
        
        return north_x, north_y
    
    def rotate_to_north(self, dx: float, dy: float) -> Tuple[float, float]:
        """Rotate movement vector to align with north direction."""
        if self.north_orientation_deg is None or self.north_orientation_deg == 0:
            return dx, dy
        
        # Rotate vector
        angle_rad = math.radians(-self.north_orientation_deg)  # Negative for counter-clockwise
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_x = dx * cos_a - dy * sin_a
        rotated_y = dx * sin_a + dy * cos_a
        
        return rotated_x, rotated_y
    
    def draw_north_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Draw north direction indicator on frame."""
        if self.north_orientation_deg is None:
            return frame
        
        # Get north vector
        north_x, north_y = self.get_north_vector()
        
        # Draw north arrow in center of frame
        center_x, center_y = self.image_width // 2, self.image_height // 2
        arrow_length = 50
        
        # Calculate arrow end point
        end_x = int(center_x + north_x * arrow_length)
        end_y = int(center_y + north_y * arrow_length)
        
        # Draw arrow
        color = (0, 255, 0) if self.north_confidence > 0.5 else (0, 255, 255)
        thickness = 3 if self.north_confidence > 0.5 else 2
        
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                       color, thickness, tipLength=0.3)
        
        # Add "N" label
        label_x = end_x + 10
        label_y = end_y - 10
        cv2.putText(frame, "N", (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Add confidence text
        conf_text = f"North: {self.north_orientation_deg:.1f}° (conf: {self.north_confidence:.2f})"
        cv2.putText(frame, conf_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame

class BuildingNorthDetector:
    """Detect north direction based on building orientations."""
    
    def detect_north(self, frame: np.ndarray, buildings: List[Dict]) -> Tuple[Optional[float], float]:
        """Detect north direction from building orientations."""
        if not buildings:
            return None, 0.0
        
        # Analyze building orientations
        orientations = []
        confidences = []
        
        for building in buildings:
            # Extract building region
            x1, y1, x2, y2 = building.get('bbox', [0, 0, 0, 0])
            if x2 <= x1 or y2 <= y1:
                continue
            
            building_roi = frame[y1:y2, x1:x2]
            if building_roi.size == 0:
                continue
            
            # Analyze building orientation
            orientation, confidence = self._analyze_building_orientation(building_roi)
            if orientation is not None:
                orientations.append(orientation)
                confidences.append(confidence)
        
        if not orientations:
            return None, 0.0
        
        # Calculate weighted average orientation
        total_weight = sum(confidences)
        if total_weight == 0:
            return None, 0.0
        
        weighted_orientation = sum(angle * conf for angle, conf in zip(orientations, confidences))
        weighted_orientation /= total_weight
        
        return weighted_orientation, min(1.0, total_weight / len(buildings))
    
    def _analyze_building_orientation(self, building_roi: np.ndarray) -> Tuple[Optional[float], float]:
        """Analyze building orientation using edge detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(building_roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0
        
        # Analyze largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit rectangle to contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Convert to north-relative angle
        north_angle = (90 - angle) % 360
        
        # Calculate confidence based on contour quality
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter == 0:
            return None, 0.0
        
        compactness = 4 * math.pi * area / (perimeter ** 2)
        confidence = min(1.0, compactness * 2)  # Higher compactness = more rectangular
        
        return north_angle, confidence

class ShadowNorthDetector:
    """Detect north direction based on shadows and lighting."""
    
    def detect_north(self, frame: np.ndarray) -> Tuple[Optional[float], float]:
        """Detect north direction from shadows."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Analyze lighting patterns
        # This is a simplified version - in reality, you'd need more sophisticated analysis
        
        # Look for shadow patterns that indicate sun direction
        shadow_angle, confidence = self._analyze_shadow_patterns(gray)
        
        if shadow_angle is not None:
            # Convert shadow direction to north direction
            # Shadows point away from sun, sun is generally south
            north_angle = (shadow_angle + 180) % 360
            return north_angle, confidence
        
        return None, 0.0
    
    def _analyze_shadow_patterns(self, gray: np.ndarray) -> Tuple[Optional[float], float]:
        """Analyze shadow patterns to determine sun direction."""
        # This is a simplified implementation
        # In reality, you'd need to analyze actual shadow patterns
        
        # For now, return a placeholder
        return None, 0.0

class RoadNorthDetector:
    """Detect north direction based on road orientations."""
    
    def detect_north(self, frame: np.ndarray, roads: List[Dict]) -> Tuple[Optional[float], float]:
        """Detect north direction from road orientations."""
        if not roads:
            return None, 0.0
        
        # Analyze road orientations
        orientations = []
        confidences = []
        
        for road in roads:
            # Extract road region
            x1, y1, x2, y2 = road.get('bbox', [0, 0, 0, 0])
            if x2 <= x1 or y2 <= y1:
                continue
            
            road_roi = frame[y1:y2, x1:x2]
            if road_roi.size == 0:
                continue
            
            # Analyze road orientation
            orientation, confidence = self._analyze_road_orientation(road_roi)
            if orientation is not None:
                orientations.append(orientation)
                confidences.append(confidence)
        
        if not orientations:
            return None, 0.0
        
        # Calculate weighted average orientation
        total_weight = sum(confidences)
        if total_weight == 0:
            return None, 0.0
        
        weighted_orientation = sum(angle * conf for angle, conf in zip(orientations, confidences))
        weighted_orientation /= total_weight
        
        return weighted_orientation, min(1.0, total_weight / len(roads))
    
    def _analyze_road_orientation(self, road_roi: np.ndarray) -> Tuple[Optional[float], float]:
        """Analyze road orientation using line detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=30, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return None, 0.0
        
        # Analyze line orientations
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
            angles.append(angle)
        
        if not angles:
            return None, 0.0
        
        # Calculate dominant orientation
        # Group similar angles
        angle_groups = {}
        for angle in angles:
            # Group angles within 10 degrees
            group_key = round(angle / 10) * 10
            if group_key not in angle_groups:
                angle_groups[group_key] = []
            angle_groups[group_key].append(angle)
        
        # Find largest group
        largest_group = max(angle_groups.values(), key=len)
        dominant_angle = np.mean(largest_group)
        
        # Calculate confidence based on group size
        confidence = min(1.0, len(largest_group) / 10.0)
        
        return dominant_angle, confidence
