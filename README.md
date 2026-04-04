# Drone Video Analysis

A Python-based computer vision system for drone video analysis, featuring optical flow estimation, speed calculation, camera calibration, and real-time HUD overlay.

## Features

- **Optical Flow Estimation** -- Compute motion vectors from drone video frames using OpenCV
- **Speed Estimation** -- Calculate ground speed from optical flow with robust outlier rejection
- **Camera Calibration** -- Lens distortion correction and intrinsic parameter estimation
- **Adaptive Learning** -- Motion prediction with adaptive noise filtering
- **North Calibration** -- Compass heading estimation from video metadata (SRT parsing)
- **Tile-Based Processing** -- Efficient frame processing using spatial tiling
- **HUD Overlay** -- Real-time heads-up display with speed, heading, and stability indicators

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Main application entry point |
| `flow.py` | Core optical flow computation |
| `optical_flow.py` | Advanced optical flow pipeline |
| `estimator.py` | Speed estimation engine |
| `robust_predictor.py` | Robust motion prediction with outlier rejection |
| `robust_speed_estimator.py` | Enhanced speed estimation |
| `camera_calibration.py` | Camera intrinsic calibration |
| `north_calibration.py` | Heading/compass calibration |
| `adaptive_learning.py` | Adaptive noise filtering |
| `motion_predictor.py` | Motion trajectory prediction |
| `hud.py` | Heads-up display rendering |
| `geo.py` | Geographic coordinate utilities |
| `srt.py` | DJI SRT subtitle parser |
| `tiles.py` | Tile-based frame processing |
| `video_io.py` | Video input/output handling |
| `stability.py` | Frame stability analysis |
| `config.py` | Configuration parameters |

## Tech Stack

- **Python 3**
- **OpenCV** -- Computer vision and optical flow
- **NumPy** -- Numerical computation

## Getting Started

```bash
git clone https://github.com/LirazBalas1/drone.git
cd drone
pip install -r requirements.txt
python app.py
```
