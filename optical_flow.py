import cv2
import numpy as np
from typing import Tuple, Optional, List
from core_types import LatLon
from geo import haversine_m, latlon_to_xy_m, xy_m_to_latlon
from camera_calibration import CameraCalibration

class OpticalFlowTracker:
    """
    Optical Flow tracker for drone navigation with camera angle compensation.
    Handles 60-degree downward camera angle and north orientation.
    """
    
    def __init__(self, 
                 camera_angle_deg: float = 60.0,
                 north_orientation_deg: float = 0.0,
                 flow_alpha: float = 0.3,
                 camera_height_m: float = 100.0):
        """
        Args:
            camera_angle_deg: Camera angle relative to ground (60 degrees)
            north_orientation_deg: Camera north orientation (0 = pointing north)
            flow_alpha: Smoothing factor for flow integration
            camera_height_m: Drone altitude in meters
        """
        self.camera_angle_deg = camera_angle_deg
        self.north_orientation_deg = north_orientation_deg
        self.flow_alpha = flow_alpha
        self.camera_height_m = camera_height_m
        
        # Enhanced camera calibration
        self.camera_calibration = CameraCalibration(
            camera_angle_deg=camera_angle_deg,
            camera_height_m=camera_height_m
        )
        
        # Optical Flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.prev_gray = None
        self.prev_features = None
        self.tracked_features = []
        self.flow_history = []
        self.max_history = 10
        
    def _compensate_camera_angle(self, flow_x: float, flow_y: float) -> Tuple[float, float]:
        """
        Compensate for 60-degree camera angle to get ground-level movement.
        Uses advanced camera calibration for better accuracy.
        """
        # Use camera calibration for more accurate compensation
        # This accounts for perspective distortion and altitude
        compensated_x, compensated_y = self.camera_calibration.calculate_movement_vector(
            (0, 0), (flow_x, flow_y)
        )
        
        return compensated_x, compensated_y
    
    def _rotate_to_north(self, flow_x: float, flow_y: float) -> Tuple[float, float]:
        """
        Rotate flow vectors to align with north orientation.
        Uses camera calibration for more accurate rotation.
        """
        return self.camera_calibration.rotate_to_north(
            flow_x, flow_y, self.north_orientation_deg
        )
    
    def update(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Update optical flow tracking.
        Returns: (flow_x, flow_y, confidence) or None if insufficient features
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            # Detect initial features
            self.prev_features = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            return None
        
        if self.prev_features is None or len(self.prev_features) == 0:
            # Re-detect features
            self.prev_features = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            self.prev_gray = gray
            return None
        
        # Calculate optical flow
        new_features, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_features, None, **self.lk_params
        )
        
        # Filter good features
        good_old = self.prev_features[status == 1]
        good_new = new_features[status == 1]
        
        if len(good_old) < 5:  # Need minimum features for reliable flow
            # Re-detect features
            self.prev_features = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.feature_params
            )
            self.prev_gray = gray
            return None
        
        # Calculate flow vectors
        flow_vectors = good_new - good_old
        flow_x = np.mean(flow_vectors[:, 0])
        flow_y = np.mean(flow_vectors[:, 1])
        
        # Calculate confidence based on feature quality and consistency
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2)
        feature_consistency = 1.0 - (np.std(flow_vectors) / (flow_magnitude + 1e-6))
        confidence = min(1.0, len(good_old) / 50.0) * feature_consistency
        
        # Apply camera angle compensation
        compensated_x, compensated_y = self._compensate_camera_angle(flow_x, flow_y)
        
        # Apply north orientation
        final_x, final_y = self._rotate_to_north(compensated_x, compensated_y)
        
        # Store flow history for smoothing
        self.flow_history.append((final_x, final_y, confidence))
        if len(self.flow_history) > self.max_history:
            self.flow_history.pop(0)
        
        # Smooth flow using history
        if len(self.flow_history) > 1:
            weights = np.exp(np.linspace(-1, 0, len(self.flow_history)))
            weights = weights / weights.sum()
            
            smooth_x = sum(w * h[0] for w, h in zip(weights, self.flow_history))
            smooth_y = sum(w * h[1] for w, h in zip(weights, self.flow_history))
            avg_confidence = sum(w * h[2] for w, h in zip(weights, self.flow_history))
        else:
            smooth_x, smooth_y = final_x, final_y
            avg_confidence = confidence
        
        # Update for next frame
        self.prev_gray = gray
        self.prev_features = good_new.reshape(-1, 1, 2)
        
        return smooth_x, smooth_y, avg_confidence
    
    def get_movement_estimate(self, 
                            current_pos: LatLon, 
                            pixel_to_meter_ratio: float = None) -> Optional[LatLon]:
        """
        Estimate new position based on optical flow with altitude compensation.
        
        Args:
            current_pos: Current GPS position
            pixel_to_meter_ratio: Conversion ratio from pixels to meters (auto-calculated if None)
        """
        if not self.flow_history:
            return None
            
        # Get latest flow
        latest_flow = self.flow_history[-1]
        flow_x, flow_y, confidence = latest_flow
        
        if confidence < 0.3:  # Low confidence threshold
            return None
        
        # Use camera calibration for accurate pixel-to-meter conversion
        if pixel_to_meter_ratio is None:
            pixel_to_meter_ratio = self.camera_calibration.pixel_to_meter_ratio
        
        # Convert pixel flow to meters with altitude compensation
        movement_x_m = flow_x * pixel_to_meter_ratio
        movement_y_m = flow_y * pixel_to_meter_ratio
        
        # Apply altitude-based scaling
        altitude_factor = self.camera_height_m / 100.0  # Normalize to 100m altitude
        movement_x_m *= altitude_factor
        movement_y_m *= altitude_factor
        
        # Convert to lat/lon
        new_lat, new_lon = xy_m_to_latlon(
            movement_x_m, movement_y_m, 
            current_pos.lat, current_pos.lon
        )
        
        return LatLon(new_lat, new_lon)
    
    def update_altitude(self, new_height_m: float):
        """Update drone altitude for better optical flow accuracy."""
        self.camera_height_m = new_height_m
        self.camera_calibration.update_altitude(new_height_m)
    
    def draw_flow_visualization(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw optical flow visualization on frame.
        """
        if self.prev_features is None or len(self.prev_features) == 0:
            return frame
        
        # Draw feature points
        for point in self.prev_features:
            x, y = point.ravel()
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw flow vectors if available
        if len(self.flow_history) > 0:
            latest_flow = self.flow_history[-1]
            flow_x, flow_y, confidence = latest_flow
            
            # Draw flow arrow in center
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Scale flow for visualization
            scale = 50.0
            end_x = int(center_x + flow_x * scale)
            end_y = int(center_y + flow_y * scale)
            
            # Color based on confidence
            color = (0, 255, 0) if confidence > 0.5 else (0, 255, 255)
            thickness = 2 if confidence > 0.5 else 1
            
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), 
                           color, thickness, tipLength=0.3)
            
            # Add confidence text
            cv2.putText(frame, f"Flow: {confidence:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
