from typing import Dict, List, Tuple, Optional
import numpy as np
from geo import latlon_to_xy_m, xy_m_to_latlon, haversine_m
from core_types import LatLon

class PositionEstimator:
    def __init__(self, lat0: float, lon0: float, pos_alpha: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self.pos_alpha = pos_alpha
        self.pred_s: Optional[LatLon] = None
        self.prev_s: Optional[LatLon] = None
        
        # Enhanced position tracking
        self.position_history = []
        self.max_history = 10
        self.flow_position_history = []
        self.detection_confidence_history = []

    def estimate(self,
                 class_weights: List[float],
                 class_locs: List[LatLon],
                 dt: float,
                 flow_position: Optional[LatLon] = None,
                 flow_confidence: float = 0.0) -> Tuple[Optional[LatLon], float, float]:
        """
        Enhanced position estimation with optical flow integration.
        Returns: (smoothed_position, speed_mps, speed_kmh)
        """
        detection_confidence = 0.0
        
        if not class_weights:
            # Use optical flow if no building detections
            if flow_position is not None and flow_confidence > 0.3:
                if self.pred_s is None:
                    self.pred_s = flow_position
                else:
                    # Blend with optical flow
                    blend_factor = min(0.5, flow_confidence)
                    self.pred_s = LatLon(
                        lat=self.pred_s.lat * (1 - blend_factor) + flow_position.lat * blend_factor,
                        lon=self.pred_s.lon * (1 - blend_factor) + flow_position.lon * blend_factor
                    )
                detection_confidence = flow_confidence
            else:
                return self.pred_s, 0.0, 0.0
        else:
            # Building-based detection
            w = np.asarray(class_weights, dtype=float)
            w = w / (w.sum() + 1e-9)
            detection_confidence = float(np.mean(w))

            xs, ys = [], []
            for w_i, ll in zip(w, class_locs):
                x, y = latlon_to_xy_m(ll.lat, ll.lon, self.lat0, self.lon0)
                xs.append(w_i * x); ys.append(w_i * y)
            xw, yw = float(np.sum(xs)), float(np.sum(ys))
            lat, lon = xy_m_to_latlon(xw, yw, self.lat0, self.lon0)
            pred = LatLon(lat, lon)

            if self.pred_s is None:
                self.pred_s = pred
            else:
                # Enhanced smoothing based on detection confidence
                alpha = self.pos_alpha * (1.0 + detection_confidence * 0.5)
                alpha = min(0.8, alpha)  # Cap the smoothing factor
                
                self.pred_s = LatLon(
                    lat=self.pred_s.lat * (1 - alpha) + pred.lat * alpha,
                    lon=self.pred_s.lon * (1 - alpha) + pred.lon * alpha
                )
                
                # Integrate optical flow if available and building detection is weak
                if flow_position is not None and flow_confidence > 0.4 and detection_confidence < 0.6:
                    flow_blend = min(0.3, flow_confidence * 0.5)
                    self.pred_s = LatLon(
                        lat=self.pred_s.lat * (1 - flow_blend) + flow_position.lat * flow_blend,
                        lon=self.pred_s.lon * (1 - flow_blend) + flow_position.lon * flow_blend
                    )

        # Store position history for analysis
        self.position_history.append(self.pred_s)
        self.detection_confidence_history.append(detection_confidence)
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
            self.detection_confidence_history.pop(0)

        # Calculate speed with enhanced smoothing
        speed_mps = 0.0
        if self.prev_s is not None and dt > 0:
            d_m = haversine_m(self.prev_s.lat, self.prev_s.lon, self.pred_s.lat, self.pred_s.lon)
            speed_mps = d_m / dt
            
            # Smooth speed based on history
            if len(self.position_history) >= 3:
                # Calculate average speed from recent positions
                recent_speeds = []
                for i in range(1, min(4, len(self.position_history))):
                    if i < len(self.position_history):
                        prev_pos = self.position_history[-i-1]
                        curr_pos = self.position_history[-i]
                        if prev_pos is not None and curr_pos is not None:
                            dist = haversine_m(prev_pos.lat, prev_pos.lon, curr_pos.lat, curr_pos.lon)
                            recent_speeds.append(dist / dt)
                
                if recent_speeds:
                    avg_speed = np.mean(recent_speeds)
                    speed_mps = speed_mps * 0.7 + avg_speed * 0.3  # Blend current and historical speed
        
        self.prev_s = self.pred_s
        return self.pred_s, speed_mps, speed_mps * 3.6
