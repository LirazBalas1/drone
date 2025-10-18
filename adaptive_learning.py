import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from core_types import LatLon
from geo import haversine_m

@dataclass
class PerformanceMetrics:
    """Performance metrics for adaptive learning."""
    detection_accuracy: float
    prediction_accuracy: float
    stability_score: float
    speed_accuracy: float
    timestamp: float

@dataclass
class MotionPattern:
    """Learned motion pattern."""
    direction: float  # degrees
    speed: float  # m/s
    confidence: float
    frequency: int  # how often this pattern occurs
    last_seen: float

class AdaptiveLearningSystem:
    """
    Advanced adaptive learning system for drone navigation.
    Learns from performance and improves over time.
    """
    
    def __init__(self, 
                 learning_rate: float = 0.1,
                 memory_size: int = 1000,
                 adaptation_threshold: float = 0.1):
        """
        Args:
            learning_rate: How fast the system learns (0.0-1.0)
            memory_size: Maximum number of patterns to remember
            adaptation_threshold: Minimum change to trigger adaptation
        """
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.adaptation_threshold = adaptation_threshold
        
        # Learning components
        self.performance_history = deque(maxlen=memory_size)
        self.motion_patterns: Dict[str, MotionPattern] = {}
        self.adaptive_parameters = self._initialize_parameters()
        
        # Learning state
        self.is_learning = True
        self.learning_confidence = 0.0
        self.adaptation_count = 0
        
    def _initialize_parameters(self) -> Dict[str, float]:
        """Initialize adaptive parameters."""
        return {
            'detection_confidence_threshold': 0.15,
            'motion_prediction_weight': 0.3,
            'stability_alpha': 0.2,
            'speed_smoothing_factor': 0.3,
            'building_boost_factor': 1.5,
            'flow_blend_factor': 0.4
        }
    
    def update_performance(self, metrics: PerformanceMetrics):
        """Update performance metrics and trigger learning."""
        self.performance_history.append(metrics)
        
        if len(self.performance_history) >= 10:  # Need minimum data
            self._analyze_performance()
            self._adapt_parameters()
    
    def _analyze_performance(self):
        """Analyze recent performance to identify patterns."""
        if len(self.performance_history) < 5:
            return
        
        recent_metrics = list(self.performance_history)[-10:]
        
        # Calculate performance trends
        detection_trend = np.mean([m.detection_accuracy for m in recent_metrics])
        prediction_trend = np.mean([m.prediction_accuracy for m in recent_metrics])
        stability_trend = np.mean([m.stability_score for m in recent_metrics])
        
        # Update learning confidence
        overall_performance = (detection_trend + prediction_trend + stability_trend) / 3
        self.learning_confidence = min(1.0, overall_performance)
        
        # Determine if learning is effective
        if len(self.performance_history) >= 20:
            old_performance = np.mean([m.detection_accuracy for m in list(self.performance_history)[-20:-10]])
            new_performance = np.mean([m.detection_accuracy for m in recent_metrics])
            
            if new_performance > old_performance + 0.05:
                self.is_learning = True
            elif new_performance < old_performance - 0.1:
                self.is_learning = False
    
    def _adapt_parameters(self):
        """Adapt parameters based on performance analysis."""
        if not self.is_learning or len(self.performance_history) < 10:
            return
        
        recent_metrics = list(self.performance_history)[-10:]
        
        # Analyze detection performance
        detection_accuracy = np.mean([m.detection_accuracy for m in recent_metrics])
        if detection_accuracy < 0.7:
            # Increase detection sensitivity
            self.adaptive_parameters['detection_confidence_threshold'] *= 0.95
            self.adaptive_parameters['building_boost_factor'] *= 1.05
        elif detection_accuracy > 0.9:
            # Decrease sensitivity to reduce false positives
            self.adaptive_parameters['detection_confidence_threshold'] *= 1.02
            self.adaptive_parameters['building_boost_factor'] *= 0.98
        
        # Analyze prediction performance
        prediction_accuracy = np.mean([m.prediction_accuracy for m in recent_metrics])
        if prediction_accuracy < 0.6:
            # Increase motion prediction weight
            self.adaptive_parameters['motion_prediction_weight'] = min(0.5, 
                self.adaptive_parameters['motion_prediction_weight'] * 1.1)
        elif prediction_accuracy > 0.8:
            # Decrease motion prediction weight
            self.adaptive_parameters['motion_prediction_weight'] = max(0.1,
                self.adaptive_parameters['motion_prediction_weight'] * 0.95)
        
        # Analyze stability
        stability_score = np.mean([m.stability_score for m in recent_metrics])
        if stability_score < 0.7:
            # Increase smoothing
            self.adaptive_parameters['stability_alpha'] = min(0.4,
                self.adaptive_parameters['stability_alpha'] * 1.1)
            self.adaptive_parameters['speed_smoothing_factor'] = min(0.5,
                self.adaptive_parameters['speed_smoothing_factor'] * 1.05)
        
        # Ensure parameters stay within reasonable bounds
        self._clamp_parameters()
        self.adaptation_count += 1
    
    def _clamp_parameters(self):
        """Ensure parameters stay within reasonable bounds."""
        self.adaptive_parameters['detection_confidence_threshold'] = np.clip(
            self.adaptive_parameters['detection_confidence_threshold'], 0.05, 0.5)
        self.adaptive_parameters['motion_prediction_weight'] = np.clip(
            self.adaptive_parameters['motion_prediction_weight'], 0.1, 0.8)
        self.adaptive_parameters['stability_alpha'] = np.clip(
            self.adaptive_parameters['stability_alpha'], 0.05, 0.6)
        self.adaptive_parameters['building_boost_factor'] = np.clip(
            self.adaptive_parameters['building_boost_factor'], 1.0, 3.0)
        self.adaptive_parameters['flow_blend_factor'] = np.clip(
            self.adaptive_parameters['flow_blend_factor'], 0.1, 0.8)
    
    def learn_motion_pattern(self, 
                           current_pos: LatLon, 
                           previous_pos: LatLon,
                           timestamp: float) -> Optional[MotionPattern]:
        """Learn motion patterns from movement data."""
        if previous_pos is None:
            return None
        
        # Calculate movement vector
        distance = haversine_m(previous_pos.lat, previous_pos.lon, 
                             current_pos.lat, current_pos.lon)
        
        if distance < 0.1:  # Too small movement
            return None
        
        # Calculate direction (simplified)
        lat_diff = current_pos.lat - previous_pos.lat
        lon_diff = current_pos.lon - previous_pos.lon
        direction = np.degrees(np.arctan2(lon_diff, lat_diff))
        
        # Create pattern key
        direction_bucket = int(direction // 30) * 30  # 30-degree buckets
        pattern_key = f"dir_{direction_bucket}"
        
        # Update or create pattern
        if pattern_key in self.motion_patterns:
            pattern = self.motion_patterns[pattern_key]
            # Update with exponential moving average
            alpha = self.learning_rate
            pattern.direction = (1 - alpha) * pattern.direction + alpha * direction
            pattern.speed = (1 - alpha) * pattern.speed + alpha * distance
            pattern.confidence = min(1.0, pattern.confidence + 0.1)
            pattern.frequency += 1
            pattern.last_seen = timestamp
        else:
            self.motion_patterns[pattern_key] = MotionPattern(
                direction=direction,
                speed=distance,
                confidence=0.5,
                frequency=1,
                last_seen=timestamp
            )
        
        return self.motion_patterns[pattern_key]
    
    def predict_next_movement(self, 
                            current_pos: LatLon,
                            current_speed: float,
                            timestamp: float) -> Optional[Tuple[LatLon, float]]:
        """Predict next movement based on learned patterns."""
        if not self.motion_patterns:
            return None
        
        # Find most relevant pattern
        best_pattern = None
        best_score = 0.0
        
        for pattern in self.motion_patterns.values():
            # Calculate relevance score
            time_decay = max(0.1, 1.0 - (timestamp - pattern.last_seen) / 3600)  # 1 hour decay
            frequency_score = min(1.0, pattern.frequency / 10.0)
            confidence_score = pattern.confidence
            
            score = time_decay * frequency_score * confidence_score
            
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        if best_pattern is None or best_score < 0.3:
            return None
        
        # Predict next position
        direction_rad = np.radians(best_pattern.direction)
        predicted_speed = best_pattern.speed
        
        # Calculate predicted position
        lat_offset = predicted_speed * np.cos(direction_rad) / 111000  # Rough conversion
        lon_offset = predicted_speed * np.sin(direction_rad) / (111000 * np.cos(np.radians(current_pos.lat)))
        
        predicted_lat = current_pos.lat + lat_offset
        predicted_lon = current_pos.lon + lon_offset
        
        return LatLon(predicted_lat, predicted_lon), predicted_speed
    
    def get_adaptive_parameters(self) -> Dict[str, float]:
        """Get current adaptive parameters."""
        return self.adaptive_parameters.copy()
    
    def get_learning_status(self) -> Dict[str, any]:
        """Get current learning status."""
        return {
            'is_learning': self.is_learning,
            'learning_confidence': self.learning_confidence,
            'adaptation_count': self.adaptation_count,
            'pattern_count': len(self.motion_patterns),
            'performance_history_size': len(self.performance_history)
        }
    
    def save_learning_state(self, filepath: str):
        """Save learning state to file."""
        state = {
            'adaptive_parameters': self.adaptive_parameters,
            'motion_patterns': {
                key: {
                    'direction': pattern.direction,
                    'speed': pattern.speed,
                    'confidence': pattern.confidence,
                    'frequency': pattern.frequency,
                    'last_seen': pattern.last_seen
                } for key, pattern in self.motion_patterns.items()
            },
            'learning_status': self.get_learning_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_learning_state(self, filepath: str):
        """Load learning state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.adaptive_parameters = state.get('adaptive_parameters', self._initialize_parameters())
            
            # Reconstruct motion patterns
            self.motion_patterns = {}
            for key, pattern_data in state.get('motion_patterns', {}).items():
                self.motion_patterns[key] = MotionPattern(**pattern_data)
            
            print(f"[INFO] Loaded adaptive learning state from {filepath}")
            
        except FileNotFoundError:
            print(f"[INFO] No existing learning state found at {filepath}")
        except Exception as e:
            print(f"[WARNING] Failed to load learning state: {e}")
