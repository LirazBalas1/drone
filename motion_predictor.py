import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque
from dataclasses import dataclass
from core_types import LatLon
from geo import haversine_m, latlon_to_xy_m, xy_m_to_latlon
from adaptive_learning import AdaptiveLearningSystem, PerformanceMetrics

@dataclass
class MotionState:
    """Current motion state."""
    position: LatLon
    velocity: Tuple[float, float]  # (lat_velocity, lon_velocity) in m/s
    acceleration: Tuple[float, float]  # (lat_acceleration, lon_acceleration) in m/s²
    timestamp: float
    confidence: float

@dataclass
class PredictionResult:
    """Motion prediction result."""
    predicted_position: LatLon
    predicted_velocity: Tuple[float, float]
    confidence: float
    time_horizon: float  # seconds into future

class MotionPredictor:
    """
    Advanced motion prediction system for drone navigation.
    Uses adaptive learning and multiple prediction methods.
    """
    
    def __init__(self, 
                 prediction_horizon: float = 2.0,  # seconds
                 history_size: int = 50,
                 learning_system: Optional[AdaptiveLearningSystem] = None):
        """
        Args:
            prediction_horizon: How far into future to predict (seconds)
            history_size: Number of motion states to keep in history
            learning_system: Adaptive learning system for improvement
        """
        self.prediction_horizon = prediction_horizon
        self.history_size = history_size
        self.learning_system = learning_system or AdaptiveLearningSystem()
        
        # Motion history
        self.motion_history: deque = deque(maxlen=history_size)
        self.last_position: Optional[LatLon] = None
        self.last_timestamp: Optional[float] = None
        
        # Prediction parameters
        self.velocity_smoothing = 0.3
        self.acceleration_smoothing = 0.2
        self.confidence_threshold = 0.3
        
        # Performance tracking
        self.prediction_errors = []
        self.max_error_history = 100
    
    def update_motion_state(self, 
                          position: LatLon, 
                          timestamp: float,
                          detected_objects: List[Dict] = None) -> MotionState:
        """Update motion state with new position."""
        
        # Calculate velocity and acceleration
        velocity = (0.0, 0.0)
        acceleration = (0.0, 0.0)
        confidence = 1.0
        
        if self.last_position is not None and self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                # Calculate velocity
                lat_vel = (position.lat - self.last_position.lat) * 111000 / dt  # m/s
                lon_vel = (position.lon - self.last_position.lat) * 111000 * np.cos(np.radians(position.lat)) / dt
                velocity = (lat_vel, lon_vel)
                
                # Calculate acceleration from velocity history
                if len(self.motion_history) > 0:
                    prev_velocity = self.motion_history[-1].velocity
                    lat_acc = (velocity[0] - prev_velocity[0]) / dt
                    lon_acc = (velocity[1] - prev_velocity[1]) / dt
                    acceleration = (lat_acc, lon_acc)
                
                # Calculate confidence based on motion consistency
                confidence = self._calculate_motion_confidence(velocity, acceleration)
        
        # Create motion state
        motion_state = MotionState(
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            timestamp=timestamp,
            confidence=confidence
        )
        
        # Add to history
        self.motion_history.append(motion_state)
        
        # Learn from motion pattern
        if self.last_position is not None:
            pattern = self.learning_system.learn_motion_pattern(
                position, self.last_position, timestamp
            )
        
        # Update tracking
        self.last_position = position
        self.last_timestamp = timestamp
        
        return motion_state
    
    def _calculate_motion_confidence(self, 
                                   velocity: Tuple[float, float], 
                                   acceleration: Tuple[float, float]) -> float:
        """Calculate confidence in motion state."""
        if len(self.motion_history) < 3:
            return 0.5
        
        # Get recent velocities for comparison
        recent_velocities = [state.velocity for state in list(self.motion_history)[-3:]]
        
        # Calculate velocity consistency
        vel_consistency = 1.0
        if len(recent_velocities) > 1:
            vel_diffs = []
            for i in range(1, len(recent_velocities)):
                prev_vel = recent_velocities[i-1]
                curr_vel = recent_velocities[i]
                diff = np.sqrt((curr_vel[0] - prev_vel[0])**2 + (curr_vel[1] - prev_vel[1])**2)
                vel_diffs.append(diff)
            
            avg_vel_diff = np.mean(vel_diffs)
            vel_consistency = max(0.1, 1.0 - avg_vel_diff / 10.0)  # Normalize by 10 m/s
        
        # Calculate acceleration smoothness
        acc_magnitude = np.sqrt(acceleration[0]**2 + acceleration[1]**2)
        acc_smoothness = max(0.1, 1.0 - acc_magnitude / 5.0)  # Normalize by 5 m/s²
        
        # Combine factors
        confidence = (vel_consistency * 0.6 + acc_smoothness * 0.4)
        return min(1.0, confidence)
    
    def predict_motion(self, 
                      current_time: float,
                      use_learning: bool = True) -> Optional[PredictionResult]:
        """Predict future motion using multiple methods."""
        
        if len(self.motion_history) < 2:
            return None
        
        # Get current motion state
        current_state = self.motion_history[-1]
        
        # Method 1: Linear extrapolation
        linear_pred = self._predict_linear(current_state, current_time)
        
        # Method 2: Polynomial fitting
        poly_pred = self._predict_polynomial(current_time)
        
        # Method 3: Learning-based prediction
        learning_pred = None
        if use_learning and self.learning_system:
            learning_pred = self.learning_system.predict_next_movement(
                current_state.position, 
                np.sqrt(current_state.velocity[0]**2 + current_state.velocity[1]**2),
                current_time
            )
        
        # Combine predictions
        final_pred = self._combine_predictions(linear_pred, poly_pred, learning_pred, current_time)
        
        return final_pred
    
    def _predict_linear(self, 
                       current_state: MotionState, 
                       current_time: float) -> Optional[PredictionResult]:
        """Linear extrapolation prediction."""
        dt = self.prediction_horizon
        
        # Predict position using current velocity
        lat_offset = current_state.velocity[0] * dt / 111000
        lon_offset = current_state.velocity[1] * dt / (111000 * np.cos(np.radians(current_state.position.lat)))
        
        predicted_lat = current_state.position.lat + lat_offset
        predicted_lon = current_state.position.lon + lon_offset
        
        predicted_position = LatLon(predicted_lat, predicted_lon)
        predicted_velocity = current_state.velocity
        
        # Calculate confidence based on velocity stability
        confidence = current_state.confidence * 0.8
        
        return PredictionResult(
            predicted_position=predicted_position,
            predicted_velocity=predicted_velocity,
            confidence=confidence,
            time_horizon=self.prediction_horizon
        )
    
    def _predict_polynomial(self, current_time: float) -> Optional[PredictionResult]:
        """Polynomial fitting prediction."""
        if len(self.motion_history) < 5:
            return None
        
        # Extract position history
        positions = [(state.position.lat, state.position.lon) for state in self.motion_history]
        timestamps = [state.timestamp for state in self.motion_history]
        
        # Normalize timestamps
        base_time = timestamps[0]
        t_norm = [(t - base_time) for t in timestamps]
        
        # Fit polynomial to lat and lon separately
        try:
            lat_coeffs = np.polyfit(t_norm, [p[0] for p in positions], min(2, len(positions)-1))
            lon_coeffs = np.polyfit(t_norm, [p[1] for p in positions], min(2, len(positions)-1))
            
            # Predict future position
            future_t = (current_time - base_time) + self.prediction_horizon
            predicted_lat = np.polyval(lat_coeffs, future_t)
            predicted_lon = np.polyval(lon_coeffs, future_t)
            
            predicted_position = LatLon(predicted_lat, predicted_lon)
            
            # Calculate predicted velocity (derivative of polynomial)
            lat_vel_coeffs = np.polyder(lat_coeffs)
            lon_vel_coeffs = np.polyder(lon_coeffs)
            
            predicted_lat_vel = np.polyval(lat_vel_coeffs, future_t) * 111000
            predicted_lon_vel = np.polyval(lon_coeffs, future_t) * 111000 * np.cos(np.radians(predicted_lat))
            
            predicted_velocity = (predicted_lat_vel, predicted_lon_vel)
            
            # Calculate confidence based on fit quality
            lat_fit_error = np.mean([(np.polyval(lat_coeffs, t) - p[0])**2 for t, p in zip(t_norm, positions)])
            lon_fit_error = np.mean([(np.polyval(lon_coeffs, t) - p[1])**2 for t, p in zip(t_norm, positions)])
            
            fit_error = (lat_fit_error + lon_fit_error) / 2
            confidence = max(0.1, 1.0 - fit_error * 1000)  # Scale error
            
            return PredictionResult(
                predicted_position=predicted_position,
                predicted_velocity=predicted_velocity,
                confidence=confidence,
                time_horizon=self.prediction_horizon
            )
            
        except np.linalg.LinAlgError:
            return None
    
    def _combine_predictions(self, 
                           linear_pred: Optional[PredictionResult],
                           poly_pred: Optional[PredictionResult],
                           learning_pred: Optional[Tuple[LatLon, float]],
                           current_time: float) -> Optional[PredictionResult]:
        """Combine multiple predictions into final result."""
        
        predictions = []
        weights = []
        
        # Add linear prediction
        if linear_pred and linear_pred.confidence > self.confidence_threshold:
            predictions.append(linear_pred)
            weights.append(linear_pred.confidence)
        
        # Add polynomial prediction
        if poly_pred and poly_pred.confidence > self.confidence_threshold:
            predictions.append(poly_pred)
            weights.append(poly_pred.confidence)
        
        # Add learning prediction
        if learning_pred:
            learning_pos, learning_speed = learning_pred
            learning_vel = (learning_speed, 0.0)  # Simplified
            learning_result = PredictionResult(
                predicted_position=learning_pos,
                predicted_velocity=learning_vel,
                confidence=0.7,  # Default confidence for learning
                time_horizon=self.prediction_horizon
            )
            predictions.append(learning_result)
            weights.append(0.7)
        
        if not predictions:
            return None
        
        # Weighted average of predictions
        total_weight = sum(weights)
        if total_weight == 0:
            return None
        
        # Calculate weighted average position
        weighted_lat = sum(p.predicted_position.lat * w for p, w in zip(predictions, weights)) / total_weight
        weighted_lon = sum(p.predicted_position.lon * w for p, w in zip(predictions, weights)) / total_weight
        
        # Calculate weighted average velocity
        weighted_lat_vel = sum(p.predicted_velocity[0] * w for p, w in zip(predictions, weights)) / total_weight
        weighted_lon_vel = sum(p.predicted_velocity[1] * w for p, w in zip(predictions, weights)) / total_weight
        
        # Calculate combined confidence
        combined_confidence = min(1.0, np.mean(weights))
        
        return PredictionResult(
            predicted_position=LatLon(weighted_lat, weighted_lon),
            predicted_velocity=(weighted_lat_vel, weighted_lon_vel),
            confidence=combined_confidence,
            time_horizon=self.prediction_horizon
        )
    
    def update_prediction_accuracy(self, 
                                 predicted_position: LatLon,
                                 actual_position: LatLon):
        """Update prediction accuracy for learning."""
        error = haversine_m(
            predicted_position.lat, predicted_position.lon,
            actual_position.lat, actual_position.lon
        )
        
        self.prediction_errors.append(error)
        if len(self.prediction_errors) > self.max_error_history:
            self.prediction_errors.pop(0)
        
        # Update learning system with performance metrics
        if self.learning_system and len(self.prediction_errors) >= 5:
            avg_error = np.mean(self.prediction_errors[-5:])
            prediction_accuracy = max(0.0, 1.0 - avg_error / 100.0)  # Normalize by 100m
            
            metrics = PerformanceMetrics(
                detection_accuracy=0.8,  # Placeholder
                prediction_accuracy=prediction_accuracy,
                stability_score=0.8,  # Placeholder
                speed_accuracy=0.8,  # Placeholder
                timestamp=self.last_timestamp or 0.0
            )
            
            self.learning_system.update_performance(metrics)
    
    def get_motion_statistics(self) -> Dict[str, float]:
        """Get motion statistics."""
        if not self.motion_history:
            return {}
        
        velocities = [np.sqrt(v.velocity[0]**2 + v.velocity[1]**2) for v in self.motion_history]
        confidences = [v.confidence for v in self.motion_history]
        
        return {
            'avg_speed': np.mean(velocities),
            'max_speed': np.max(velocities),
            'avg_confidence': np.mean(confidences),
            'prediction_accuracy': 1.0 - np.mean(self.prediction_errors) / 100.0 if self.prediction_errors else 0.0,
            'motion_history_size': len(self.motion_history)
        }
