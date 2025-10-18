from dataclasses import dataclass

@dataclass(frozen=True)
class ModelCfg:
    weights_path: str = "data/best.pt"
    conf: float = 0.12  # Even lower confidence for better building detection
    img_size: int = 1280
    use_half: bool = True
    device_index: int = 0
    # Building-specific parameters
    building_conf_thresh: float = 0.15  # Lower threshold for buildings
    temporal_consistency: bool = True  # Enable temporal consistency
    multi_scale_detection: bool = True  # Enable multi-scale detection
    shape_analysis: bool = True  # Enable building shape analysis
    feature_detection: bool = True  # Enable building feature detection

@dataclass(frozen=True)
class VideoCfg:
    input_path: str = "inputs/DJI_0072wrongangle.MP4"
    output_path: str = "predictions/DJI_0072wrongangle_output.mp4"
    preview: bool = True

@dataclass(frozen=True)
class DataCfg:
    srt_path: str = "inputs/DJI_0072wrongangle.srt"
    excel_locations_path: str = "data/locations _ariel_uni.xlsx"  # Label | Latitude | Longitude

@dataclass(frozen=True)
class HudCfg:
    tile_size: int = 256
    hud_size: int = 320
    hud_zoom: int = 18
    building_radius_m: float = 400.0
    tile_provider_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    tile_cache_dir: str = "tile_cache"
    user_agent: str = "DroneHUD/1.0 (educational)"

@dataclass(frozen=True)
class StabilityCfg:
    history_len: int = 15  # Longer history for better stability
    presence_gamma: float = 1.2  # Reduced for more sensitive detection
    conf_alpha: float = 0.8  # Higher confidence weight
    ema_beta: float = 0.25  # Slower adaptation
    lock_thresh: float = 0.55  # Lower threshold for easier locking
    unlock_thresh: float = 0.45  # Lower threshold for easier unlocking

@dataclass(frozen=True)
class SmoothCfg:
    pos_alpha: float = 0.15  # Even slower temporal EMA for smoother tracking
    flow_blend_alpha: float = 0.4  # Higher optical flow blending factor
    building_weight_boost: float = 2.0  # Higher boost weight for building detections
    north_calibration_alpha: float = 0.3  # North calibration smoothing
    altitude_adaptation: bool = True  # Enable altitude adaptation

@dataclass(frozen=True)
class AppCfg:
    model: ModelCfg = ModelCfg()
    video: VideoCfg = VideoCfg()
    data: DataCfg = DataCfg()
    hud: HudCfg = HudCfg()
    stability: StabilityCfg = StabilityCfg()
    smooth: SmoothCfg = SmoothCfg()
