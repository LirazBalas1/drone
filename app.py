import cv2
import pandas as pd
from collections import defaultdict
from typing import List, Tuple
from config import AppCfg
from core_types import LatLon, BoxDet
from srt import load_srt_latlon
from tiles import CachedHttpTileProvider
from hud import HudRenderer
from detector.yolo_ultra import YoloUltranyxDetector
from stability import StabilityGate
from estimator import PositionEstimator
from dedup import deduplicate_by_class
from video_io import Cv2VideoWriter
from geo import haversine_m
from optical_flow import OpticalFlowTracker
from north_calibration import NorthCalibration
from adaptive_learning import AdaptiveLearningSystem, PerformanceMetrics
from motion_predictor import MotionPredictor

def overlay_bottom_right(frame, overlay_img, pad=12):
    H, W = frame.shape[:2]
    h, w = overlay_img.shape[:2]
    x1 = W - w - pad
    y1 = H - h - pad
    frame[y1:y1+h, x1:x1+w] = overlay_img
    return frame

def _get_screen_resolution():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080

def _resize_for_screen(img, margin_w=80, margin_h=120):
    sw, sh = _get_screen_resolution()
    h, w = img.shape[:2]
    max_w = max(sw - margin_w, 1)
    max_h = max(sh - margin_h, 1)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def main():
    cfg = AppCfg()

    # --- Load label->GPS (pure IO) ---
    df = pd.read_excel(cfg.data.excel_locations_path, header=None)
    df.columns = ["Label", "Latitude", "Longitude"]
    label2gps = {str(r.Label): LatLon(float(r.Latitude), float(r.Longitude)) for _, r in df.iterrows()}
    nodes_all: List[Tuple[str, LatLon]] = [(lbl, ll) for lbl, ll in label2gps.items()]
    print(f"[INFO] Loaded {len(label2gps)} label->GPS entries")

    # --- SRT (ground truth positions for accuracy only) ---
    srt = load_srt_latlon(cfg.data.srt_path)
    if not srt:
        raise RuntimeError(f"No SRT entries parsed from {cfg.data.srt_path}")
    print(f"[INFO] Loaded {len(srt)} SRT frames")
    lat0, lon0 = srt[0]["lat"], srt[0]["lon"]
    # derive dt/fps from SRT DiffTime when available to avoid double-reading video
    dt_ms_vals = [e.get("dt_ms") for e in srt[:120] if isinstance(e.get("dt_ms"), (int, float))]
    if dt_ms_vals:
        avg_dt_ms = sum(dt_ms_vals) / len(dt_ms_vals)
        fps = 1000.0 / max(avg_dt_ms, 1e-3)
    else:
        fps = 30.0
    dt = 1.0 / float(fps)

    # --- Detector ---
    det = YoloUltranyxDetector(cfg.model)
    
    # --- North Calibration ---
    north_calibration = NorthCalibration(
        image_width=1920,  # Will be updated from first frame
        image_height=1080,
        camera_angle_deg=60.0
    )
    
    # --- Adaptive Learning System ---
    adaptive_learning = AdaptiveLearningSystem(
        learning_rate=0.1,
        memory_size=1000,
        adaptation_threshold=0.1
    )
    
    # --- Motion Predictor ---
    motion_predictor = MotionPredictor(
        prediction_horizon=2.0,  # 2 seconds ahead
        history_size=50,
        learning_system=adaptive_learning
    )
    
    # --- Optical Flow Tracker ---
    flow_tracker = OpticalFlowTracker(
        camera_angle_deg=60.0,  # 60 degrees downward
        north_orientation_deg=0.0,  # Will be updated by north calibration
        flow_alpha=0.3,
        camera_height_m=100.0  # Default altitude
    )

    # --- Tile/HUD ---
    tp = CachedHttpTileProvider(
        base_url=cfg.hud.tile_provider_url,
        cache_dir=cfg.hud.tile_cache_dir,
        user_agent=cfg.hud.user_agent,
        tile_size=cfg.hud.tile_size
    )
    hud = HudRenderer(tp, cfg.hud.tile_size, cfg.hud.hud_size, cfg.hud.hud_zoom)

    # --- Stability / Estimator ---
    gate = StabilityGate(
        history_len=cfg.stability.history_len,
        presence_gamma=cfg.stability.presence_gamma,
        conf_alpha=cfg.stability.conf_alpha,
        ema_beta=cfg.stability.ema_beta,
        lock_thresh=cfg.stability.lock_thresh,
        unlock_thresh=cfg.stability.unlock_thresh,
    )
    est = PositionEstimator(lat0, lon0, pos_alpha=cfg.smooth.pos_alpha)

    # --- Video Output (initialized on first frame to avoid double-reading for size) ---
    wtr = None
    W = H = None  # determined from first frame
    print(f"[INFO] Target FPS (from SRT): {fps:.2f}")

    path_pred: List[LatLon] = []
    frame_idx = 0

    try:
        for frame_det in det.stream(cfg.video.input_path):
            frame = frame_det.frame_bgr
            names = frame_det.names
            
            # lazily initialize writer with first frame size
            if wtr is None:
                H, W = frame.shape[:2]
                wtr = Cv2VideoWriter(cfg.video.output_path, fps=fps, size=(W, H))
                print(f"[INFO] Video parameters: {W}x{H} @ {fps:.2f}fps")

            # --- Deduplicate per class (except "other") ---
            kept, class_conf = deduplicate_by_class(frame_det.boxes, names, "other")
            # filter out 'other' from stability, drawing, and estimation
            class_conf = {k: v for k, v in class_conf.items() if k != "other"}

            # --- Stability update & hysteresis ---
            presence_now = {names[b.cls_id]: 1 for b in kept if names[b.cls_id] != "other"}
            score = gate.update(presence_now, class_conf)

            # Keep only locked classes
            kept_locked: List[BoxDet] = [b for b in kept if names[b.cls_id] != "other" and gate.is_locked(names[b.cls_id])]
            
            # --- Update North Calibration ---
            # Prepare building data for north detection
            buildings_data = []
            for b in kept_locked:
                x1, y1, x2, y2 = b.xyxy.astype(int)
                buildings_data.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': b.conf
                })
            
            # Detect north direction
            north_angle, north_confidence = north_calibration.detect_north_direction(
                frame, buildings=buildings_data
            )
            
            # Update optical flow tracker with north direction
            if north_confidence > 0.3:
                flow_tracker.north_orientation_deg = north_angle
                flow_tracker.camera_calibration.north_orientation_deg = north_angle
            
            # --- Update Optical Flow ---
            flow_result = flow_tracker.update(frame)
            flow_x, flow_y, flow_confidence = 0.0, 0.0, 0.0
            if flow_result is not None:
                flow_x, flow_y, flow_confidence = flow_result

            # --- Draw Ultralytics style overlays (minimal, no blur tricks) ---
            # Avoid rescaling; draw directly on `frame`.
            for b in kept_locked:
                x1, y1, x2, y2 = b.xyxy.astype(int).tolist()
                cname = names[b.cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(frame, f"{cname} {b.conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            # --- Class center dots + label GPS ---
            class_weights = []
            class_locs = []
            for b in kept_locked:
                x1, y1, x2, y2 = b.xyxy
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                cname = names[b.cls_id]
                cv2.circle(frame, (cx, cy), 6, (50, 200, 50), -1)

                ll = label2gps.get(cname)
                if ll:
                    cv2.putText(frame, f"{ll.lat:.6f}, {ll.lon:.6f}", (cx+8, cy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 50), 2, cv2.LINE_AA)
                    # weights: score * (1+0.5*area_norm) * conf
                    w = (score.get(cname, 0.0)) * (1.0 + 0.5 * (((x2-x1)*(y2-y1)) / (W*H))) * max(b.conf, 1e-3)
                    class_weights.append(w)
                    class_locs.append(ll)

            # --- Optical Flow Position Estimation ---
            flow_position = None
            if flow_confidence > 0.3:
                # Get optical flow position estimate
                pixel_to_meter_ratio = 0.1  # Adjust based on altitude and camera specs
                if pred_s is not None:
                    flow_position = flow_tracker.get_movement_estimate(pred_s, pixel_to_meter_ratio)
                elif len(path_pred) > 0:
                    # Use last known position for flow estimation
                    flow_position = flow_tracker.get_movement_estimate(path_pred[-1], pixel_to_meter_ratio)
            
            # --- Enhanced Position Estimation with Optical Flow Integration ---
            pred_s, spd_mps, spd_kmh = est.estimate(
                class_weights, class_locs, dt, 
                flow_position=flow_position, 
                flow_confidence=flow_confidence
            )
            
            # --- Adaptive Learning and Motion Prediction ---
            if pred_s is not None:
                # Update motion predictor
                current_time = frame_idx * dt
                motion_state = motion_predictor.update_motion_state(pred_s, current_time)
                
                # Get motion prediction
                motion_prediction = motion_predictor.predict_motion(current_time, use_learning=True)
                
                # Apply adaptive parameters to estimator
                adaptive_params = adaptive_learning.get_adaptive_parameters()
                if adaptive_params:
                    # Update estimator with adaptive parameters
                    est.pos_alpha = adaptive_params.get('stability_alpha', est.pos_alpha)
                
                # Update prediction accuracy for learning
                if len(path_pred) > 0 and motion_prediction:
                    motion_predictor.update_prediction_accuracy(
                        motion_prediction.predicted_position, pred_s
                    )
                
                path_pred.append(pred_s)

            # --- Draw Optical Flow Visualization ---
            frame = flow_tracker.draw_flow_visualization(frame)
            
            # --- Draw North Direction Indicator ---
            frame = north_calibration.draw_north_indicator(frame)
            
            # --- Overlays: PRED, SPD, SRT, ERR, FLOW ---
            y0 = 30
            if pred_s is not None:
                cv2.putText(frame, f"PRED {pred_s.lat:.6f}, {pred_s.lon:.6f}", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                y0 += 30
                cv2.putText(frame, f"SPD  {spd_mps:5.1f} m/s  ({spd_kmh:5.1f} km/h)", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                y0 += 30
                
                # Optical Flow information
                if flow_confidence > 0.1:
                    cv2.putText(frame, f"FLOW {flow_x:6.1f}, {flow_y:6.1f} (conf: {flow_confidence:.2f})", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2, cv2.LINE_AA)
                    y0 += 25
                
                # Adaptive Learning information
                learning_status = adaptive_learning.get_learning_status()
                if learning_status['is_learning']:
                    cv2.putText(frame, f"LEARNING: {learning_status['learning_confidence']:.2f} (adapt: {learning_status['adaptation_count']})", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)
                    y0 += 20
                
                # Motion Prediction information
                motion_stats = motion_predictor.get_motion_statistics()
                if motion_stats:
                    cv2.putText(frame, f"PREDICT: {motion_stats.get('prediction_accuracy', 0):.2f} (speed: {motion_stats.get('avg_speed', 0):.1f})", (20, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)
                    y0 += 20

            if frame_idx < len(srt) and pred_s is not None:
                gt_lat = srt[frame_idx]["lat"]; gt_lon = srt[frame_idx]["lon"]
                err_m = haversine_m(pred_s.lat, pred_s.lon, gt_lat, gt_lon)
                cv2.putText(frame, f"SRT  {gt_lat:.6f}, {gt_lon:.6f}", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
                y0 += 30
                cv2.putText(frame, f"ERR  {err_m:6.1f} m", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)

            # --- Camera center reticle ---
            ch, cw = frame.shape[:2]
            ccx, ccy = cw // 2, ch // 2
            cv2.circle(frame, (ccx, ccy), 10, (255,255,255), 2)
            cv2.line(frame, (ccx-18, ccy), (ccx-4, ccy), (255,255,255), 2)
            cv2.line(frame, (ccx+4, ccy), (ccx+18, ccy), (255,255,255), 2)
            cv2.line(frame, (ccx, ccy-18), (ccx, ccy-4), (255,255,255), 2)
            cv2.line(frame, (ccx, ccy+4), (ccx, ccy+18), (255,255,255), 2)

            # --- HUD ---
            if pred_s is not None:
                # filter nodes by radius on the fly (no side effects)
                nearby = []
                for name, ll in nodes_all:
                    d = haversine_m(pred_s.lat, pred_s.lon, ll.lat, ll.lon)
                    if d <= cfg.hud.building_radius_m:
                        nearby.append((name, ll))
                hud_img = hud.render(pred_s, path_pred, nearby)
                frame = overlay_bottom_right(frame, hud_img, pad=12)

            # --- IO ---
            if wtr is not None:
                wtr.write(frame)
            if cfg.video.preview:
                disp = _resize_for_screen(frame)
                cv2.imshow("YOLO + HUD + Stability + Speed", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    finally:
        if wtr is not None:
            wtr.close()
        if cfg.video.preview:
            cv2.destroyAllWindows()
        
        # Save adaptive learning state
        try:
            adaptive_learning.save_learning_state("adaptive_learning_state.json")
            print(f"[INFO] Saved adaptive learning state")
        except Exception as e:
            print(f"[WARNING] Failed to save learning state: {e}")
        
        print(f"[DONE] Saved: {cfg.video.output_path}")

if __name__ == "__main__":
    main()
