import cv2
import numpy as np
import math
from typing import Tuple, Optional, List
from core_types import LatLon
from geo import haversine_m, latlon_to_xy_m, xy_m_to_latlon

class CameraCalibration:
    """
    Advanced camera calibration for 60-degree downward angle drone navigation.
    Handles perspective distortion, altitude compensation, and north orientation.
    """
    
    def __init__(self, 
                 camera_angle_deg: float = 60.0,
                 camera_height_m: float = 100.0,
                 focal_length_px: float = 1000.0,
                 image_width: int = 1920,
                 image_height: int = 1080):
        """
        Args:
            camera_angle_deg: Camera angle relative to ground (60 degrees)
            camera_height_m: Drone altitude in meters
            focal_length_px: Camera focal length in pixels
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.camera_angle_deg = camera_angle_deg
        self.camera_height_m = camera_height_m
        self.focal_length_px = focal_length_px
        self.image_width = image_width
        self.image_height = image_height
        
        # Calculate camera parameters
        self.angle_rad = math.radians(camera_angle_deg)
        self.tan_angle = math.tan(self.angle_rad)
        
        # Ground plane parameters
        self.ground_plane_distance = camera_height_m / math.sin(self.angle_rad)
        self.pixel_to_meter_ratio = self._calculate_pixel_to_meter_ratio()
        
        # Perspective correction matrix
        self.perspective_matrix = self._calculate_perspective_matrix()
        
    def _calculate_pixel_to_meter_ratio(self) -> float:
        """Calculate pixel-to-meter ratio based on camera angle and altitude."""
        # At 60 degrees, the ground appears at a distance
        ground_distance = self.camera_height_m / self.tan_angle
        
        # Calculate how many meters per pixel at ground level
        # This is an approximation - in reality, it varies across the image
        meters_per_pixel = ground_distance / (self.focal_length_px * 2)
        return meters_per_pixel
    
    def _calculate_perspective_matrix(self) -> np.ndarray:
        """Calculate perspective correction matrix for 60-degree angle."""
        # This is a simplified perspective correction
        # In reality, you'd need more sophisticated camera calibration
        
        # Source points (distorted view from 60-degree angle)
        src_points = np.float32([
            [0, 0],  # Top-left
            [self.image_width, 0],  # Top-right
            [self.image_width, self.image_height],  # Bottom-right
            [0, self.image_height]  # Bottom-left
        ])
        
        # Destination points (corrected view)
        # Compensate for perspective distortion
        perspective_factor = 0.8  # Adjust based on angle
        dst_points = np.float32([
            [self.image_width * (1 - perspective_factor) / 2, 0],
            [self.image_width * (1 + perspective_factor) / 2, 0],
            [self.image_width, self.image_height],
            [0, self.image_height]
        ])
        
        return cv2.getPerspectiveTransform(src_points, dst_points)
    
    def correct_perspective(self, frame: np.ndarray) -> np.ndarray:
        """Apply perspective correction to frame."""
        if self.perspective_matrix is not None:
            return cv2.warpPerspective(frame, self.perspective_matrix, 
                                      (self.image_width, self.image_height))
        return frame
    
    def get_ground_distance_at_pixel(self, x: int, y: int) -> float:
        """
        Calculate distance to ground at specific pixel location.
        This accounts for the 60-degree camera angle.
        """
        # Distance from image center
        center_x, center_y = self.image_width // 2, self.image_height // 2
        dx = x - center_x
        dy = y - center_y
        
        # Calculate distance based on camera angle
        pixel_distance = math.sqrt(dx**2 + dy**2)
        angle_from_center = math.atan2(pixel_distance, self.focal_length_px)
        
        # Ground distance at this pixel
        ground_distance = self.camera_height_m / math.tan(self.angle_rad + angle_from_center)
        return ground_distance
    
    def pixel_to_ground_coordinates(self, x: int, y: int) -> Tuple[float, float]:
        """
        Convert pixel coordinates to ground coordinates.
        Accounts for 60-degree camera angle and perspective distortion.
        """
        # Get distance to ground at this pixel
        ground_distance = self.get_ground_distance_at_pixel(x, y)
        
        # Calculate ground coordinates
        # X coordinate (east-west)
        center_x = self.image_width // 2
        x_offset = (x - center_x) * ground_distance / self.focal_length_px
        
        # Y coordinate (north-south) - affected by camera angle
        center_y = self.image_height // 2
        y_offset = (y - center_y) * ground_distance / self.focal_length_px
        
        # Apply camera angle compensation
        y_offset *= math.cos(self.angle_rad)
        
        return x_offset, y_offset
    
    def ground_to_pixel_coordinates(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """
        Convert ground coordinates to pixel coordinates.
        Reverse of pixel_to_ground_coordinates.
        """
        # Calculate pixel coordinates
        center_x = self.image_width // 2
        center_y = self.image_height // 2
        
        # X coordinate
        x_pixel = int(center_x + x_m * self.focal_length_px / self.ground_plane_distance)
        
        # Y coordinate (with angle compensation)
        y_pixel = int(center_y + y_m * self.focal_length_px / (self.ground_plane_distance * math.cos(self.angle_rad)))
        
        return x_pixel, y_pixel
    
    def calculate_movement_vector(self, 
                                prev_pixel: Tuple[int, int], 
                                curr_pixel: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate movement vector in meters from pixel movement.
        Accounts for camera angle and perspective.
        """
        prev_x, prev_y = prev_pixel
        curr_x, curr_y = curr_pixel
        
        # Get ground coordinates for both points
        prev_x_m, prev_y_m = self.pixel_to_ground_coordinates(prev_x, prev_y)
        curr_x_m, curr_y_m = self.pixel_to_ground_coordinates(curr_x, curr_y)
        
        # Calculate movement vector
        dx_m = curr_x_m - prev_x_m
        dy_m = curr_y_m - prev_y_m
        
        return dx_m, dy_m
    
    def adjust_detection_boxes(self, boxes: List, frame_shape: Tuple[int, int]) -> List:
        """
        Adjust detection boxes for perspective distortion.
        This helps with building detection at 60-degree angle.
        """
        adjusted_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            
            # Get ground coordinates for box corners
            tl_x, tl_y = self.pixel_to_ground_coordinates(int(x1), int(y1))
            br_x, br_y = self.pixel_to_ground_coordinates(int(x2), int(y2))
            
            # Calculate adjusted box dimensions
            width_m = abs(br_x - tl_x)
            height_m = abs(br_y - tl_y)
            
            # Filter boxes based on realistic building dimensions
            min_building_size = 5.0  # meters
            max_building_size = 100.0  # meters
            
            if (min_building_size <= width_m <= max_building_size and 
                min_building_size <= height_m <= max_building_size):
                
                # Adjust confidence based on perspective
                perspective_factor = min(1.0, width_m * height_m / 100.0)
                adjusted_conf = box.conf * perspective_factor
                
                adjusted_boxes.append(type(box)(
                    xyxy=box.xyxy,
                    cls_id=box.cls_id,
                    conf=adjusted_conf
                ))
        
        return adjusted_boxes
    
    def get_north_direction_vector(self) -> Tuple[float, float]:
        """
        Get north direction vector in pixel coordinates.
        Assumes camera is pointing north.
        """
        # North is "up" in the image when camera points north
        # But we need to account for the 60-degree angle
        north_x = 0.0
        north_y = -1.0 * math.cos(self.angle_rad)  # Compensate for camera angle
        
        return north_x, north_y
    
    def rotate_to_north(self, dx: float, dy: float, north_orientation_deg: float = 0.0) -> Tuple[float, float]:
        """
        Rotate movement vector to align with north direction.
        """
        if north_orientation_deg == 0:
            return dx, dy
        
        # Rotate vector
        angle_rad = math.radians(north_orientation_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        
        rotated_x = dx * cos_a - dy * sin_a
        rotated_y = dx * sin_a + dy * cos_a
        
        return rotated_x, rotated_y
    
    def update_altitude(self, new_height_m: float):
        """Update drone altitude for better calibration."""
        self.camera_height_m = new_height_m
        self.ground_plane_distance = new_height_m / math.sin(self.angle_rad)
        self.pixel_to_meter_ratio = self._calculate_pixel_to_meter_ratio()
    
    def get_calibration_info(self) -> dict:
        """Get current calibration parameters."""
        return {
            'camera_angle_deg': self.camera_angle_deg,
            'camera_height_m': self.camera_height_m,
            'pixel_to_meter_ratio': self.pixel_to_meter_ratio,
            'ground_plane_distance': self.ground_plane_distance,
            'focal_length_px': self.focal_length_px
        }
