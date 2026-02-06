import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml


Point = Tuple[int, int]


@dataclass
class CameraConfig:
    index: int
    width: int
    height: int
    fps: int
    video_file: Optional[str]


@dataclass
class ROIConfig:
    mode: str  # "aruco" or "manual"
    warp_size: int
    aruco_dictionary: str
    aruco_ids: List[int]
    roi_cache_path: str


@dataclass
class PreviewConfig:
    show_raw: bool
    show_warp: bool
    show_fps: bool
    show_diff: bool
    show_mask: bool
    show_overlay: bool
    window_size: int
    mosaic: bool
    show_controls: bool
    simple_controls: bool


@dataclass
class BaselineConfig:
    path: str
    capture_frames: int
    capture_delay_ms: int


@dataclass
class DetectionConfig:
    mode: str
    preview_all: bool
    preview_only: bool
    blur_ksize: int
    diff_threshold: int
    morph_open_ksize: int
    morph_close_ksize: int
    use_clahe: bool
    clahe_clip: float
    clahe_grid: int
    min_area: int
    max_area: int
    use_aspect: bool
    aspect_ratio_min: float
    aspect_ratio_max: float
    use_length: bool
    orientation_mode: str
    use_orientation: bool
    angle_tolerance_deg: float
    length_min_cm: float
    length_max_cm: float
    red_hue_low1: int
    red_hue_high1: int
    red_hue_low2: int
    red_hue_high2: int
    red_sat_min: int
    red_val_min: int


@dataclass
class TrackerConfig:
    match_distance_px: float
    min_persist_frames: int
    cooldown_frames: int


@dataclass
class FreezeMaskConfig:
    enabled: bool
    dilate_ksize: int


@dataclass
class AppConfig:
    camera: CameraConfig
    roi: ROIConfig
    preview: PreviewConfig
    baseline: BaselineConfig
    detection: DetectionConfig
    tracker: TrackerConfig
    freeze_mask: FreezeMaskConfig


class VideoSource:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        if cfg.video_file:
            self.cap = cv2.VideoCapture(cfg.video_file)
        else:
            self.cap = cv2.VideoCapture(cfg.index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()


class ROIAligner:
    def __init__(self, cfg: ROIConfig) -> None:
        self.cfg = cfg
        self.manual_points: List[Point] = []
        self._load_cached_points()

    def _load_cached_points(self) -> None:
        cache_path = Path(self.cfg.roi_cache_path)
        if not cache_path.exists():
            return
        try:
            data = json.loads(cache_path.read_text())
            pts = data.get("points", [])
            if len(pts) == 4:
                self.manual_points = [(int(p[0]), int(p[1])) for p in pts]
        except Exception:
            self.manual_points = []

    def _save_cached_points(self) -> None:
        cache_path = Path(self.cfg.roi_cache_path)
        cache_path.write_text(json.dumps({"points": self.manual_points}))

    def set_manual_point(self, pt: Point) -> None:
        if len(self.manual_points) < 4:
            self.manual_points.append(pt)
            if len(self.manual_points) == 4:
                self._save_cached_points()

    def reset_manual_points(self) -> None:
        self.manual_points = []

    def _aruco_dict(self) -> int:
        name = self.cfg.aruco_dictionary
        if not hasattr(cv2.aruco, name):
            raise ValueError(f"Unknown ArUco dictionary: {name}")
        return getattr(cv2.aruco, name)

    def _find_aruco_corners(self, frame: np.ndarray) -> Optional[List[Point]]:
        if not hasattr(cv2, "aruco"):
            return None
        aruco_dict = cv2.aruco.getPredefinedDictionary(self._aruco_dict())
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            return None
        id_to_center = {}
        for corner, marker_id in zip(corners, ids.flatten()):
            if marker_id in self.cfg.aruco_ids:
                c = corner[0]
                center = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                id_to_center[int(marker_id)] = center
        if len(id_to_center) != 4:
            return None
        ordered = [id_to_center[i] for i in self.cfg.aruco_ids]
        return ordered

    def get_warp(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.cfg.mode == "aruco":
            src_pts = self._find_aruco_corners(frame)
        else:
            src_pts = self.manual_points if len(self.manual_points) == 4 else None
        if src_pts is None:
            return None, None
        dst_size = self.cfg.warp_size
        dst_pts = np.array(
            [(0, 0), (dst_size - 1, 0), (dst_size - 1, dst_size - 1), (0, dst_size - 1)],
            dtype=np.float32,
        )
        src = np.array(src_pts, dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, dst_pts)
        warped = cv2.warpPerspective(frame, H, (dst_size, dst_size))
        return warped, H


class Visualizer:
    def __init__(self, cfg: PreviewConfig) -> None:
        self.cfg = cfg
        self.last_time = time.time()
        self.fps = 0.0

    def update_fps(self) -> None:
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

    def draw_fps(self, frame: np.ndarray) -> None:
        if not self.cfg.show_fps:
            return
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


class BaselineManager:
    def __init__(self, cfg: BaselineConfig) -> None:
        self.cfg = cfg
        self.baseline: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        path = Path(self.cfg.path)
        if not path.exists():
            return
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            self.baseline = img

    def save(self) -> None:
        if self.baseline is None:
            return
        cv2.imwrite(self.cfg.path, self.baseline)

    def capture(self, get_frame_fn) -> bool:
        frames: List[np.ndarray] = []
        for _ in range(max(1, self.cfg.capture_frames)):
            frame = get_frame_fn()
            if frame is None:
                continue
            frames.append(frame.astype(np.float32))
            if self.cfg.capture_delay_ms > 0:
                cv2.waitKey(self.cfg.capture_delay_ms)
        if not frames:
            return False
        avg = np.mean(frames, axis=0).astype(np.uint8)
        self.baseline = avg
        self.save()
        return True


@dataclass
class Blob:
    contour: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: float
    centroid: Tuple[float, float]
    angle_deg: float
    length_px: float


@dataclass
class Track:
    track_id: int
    centroid: Tuple[float, float]
    bbox: Tuple[int, int, int, int]
    area: float
    angle_deg: float
    length_px: float
    frames_seen: int
    last_seen_frame: int
    confirmed: bool


class StrokeTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        self.cfg = cfg
        self.tracks: List[Track] = []
        self.next_id = 1
        self.count = 0
        self.last_confirm_frame = -1_000_000

    def reset(self) -> None:
        self.tracks = []
        self.next_id = 1
        self.count = 0
        self.last_confirm_frame = -1_000_000

    def update(self, blobs: List[Blob], frame_idx: int) -> List[Track]:
        assigned = set()
        for track in self.tracks:
            best_idx = -1
            best_dist = float("inf")
            for i, blob in enumerate(blobs):
                if i in assigned:
                    continue
                dx = blob.centroid[0] - track.centroid[0]
                dy = blob.centroid[1] - track.centroid[1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0 and best_dist <= self.cfg.match_distance_px:
                blob = blobs[best_idx]
                assigned.add(best_idx)
                track.centroid = blob.centroid
                track.bbox = blob.bbox
                track.area = blob.area
                track.angle_deg = blob.angle_deg
                track.length_px = blob.length_px
                track.frames_seen += 1
                track.last_seen_frame = frame_idx

        for i, blob in enumerate(blobs):
            if i in assigned:
                continue
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    centroid=blob.centroid,
                    bbox=blob.bbox,
                    area=blob.area,
                    angle_deg=blob.angle_deg,
                    length_px=blob.length_px,
                    frames_seen=1,
                    last_seen_frame=frame_idx,
                    confirmed=False,
                )
            )
            self.next_id += 1

        alive: List[Track] = []
        for track in self.tracks:
            if frame_idx - track.last_seen_frame <= self.cfg.min_persist_frames + 2:
                alive.append(track)
        self.tracks = alive

        confirmed_now: List[Track] = []
        for track in self.tracks:
            if track.confirmed:
                continue
            if track.frames_seen >= self.cfg.min_persist_frames:
                if frame_idx - self.last_confirm_frame >= self.cfg.cooldown_frames:
                    track.confirmed = True
                    self.count += 1
                    self.last_confirm_frame = frame_idx
                    confirmed_now.append(track)
        return confirmed_now


def load_config(path: Path) -> AppConfig:
    data = yaml.safe_load(path.read_text())
    cam = data["camera"]
    roi = data["roi"]
    preview = data["preview"]
    baseline = data["baseline"]
    detection = data["detection"]
    tracker = data["tracker"]
    freeze_mask = data["freeze_mask"]
    return AppConfig(
        camera=CameraConfig(
            index=int(cam["index"]),
            width=int(cam["width"]),
            height=int(cam["height"]),
            fps=int(cam["fps"]),
            video_file=cam.get("video_file"),
        ),
        roi=ROIConfig(
            mode=str(roi["mode"]),
            warp_size=int(roi["warp_size"]),
            aruco_dictionary=str(roi["aruco_dictionary"]),
            aruco_ids=list(roi["aruco_ids"]),
            roi_cache_path=str(roi["roi_cache_path"]),
        ),
        preview=PreviewConfig(
            show_raw=bool(preview["show_raw"]),
            show_warp=bool(preview["show_warp"]),
            show_fps=bool(preview["show_fps"]),
            show_diff=bool(preview["show_diff"]),
            show_mask=bool(preview["show_mask"]),
            show_overlay=bool(preview["show_overlay"]),
            window_size=int(preview["window_size"]),
            mosaic=bool(preview["mosaic"]),
            show_controls=bool(preview["show_controls"]),
            simple_controls=bool(preview.get("simple_controls", True)),
        ),
        baseline=BaselineConfig(
            path=str(baseline["path"]),
            capture_frames=int(baseline["capture_frames"]),
            capture_delay_ms=int(baseline["capture_delay_ms"]),
        ),
        detection=DetectionConfig(
            mode=str(detection["mode"]),
            preview_all=bool(detection.get("preview_all", False)),
            preview_only=bool(detection.get("preview_only", False)),
            blur_ksize=int(detection["blur_ksize"]),
            diff_threshold=int(detection["diff_threshold"]),
            morph_open_ksize=int(detection["morph_open_ksize"]),
            morph_close_ksize=int(detection["morph_close_ksize"]),
            use_clahe=bool(detection["use_clahe"]),
            clahe_clip=float(detection["clahe_clip"]),
            clahe_grid=int(detection["clahe_grid"]),
            min_area=int(detection["min_area"]),
            max_area=int(detection["max_area"]),
            use_aspect=bool(detection["use_aspect"]),
            aspect_ratio_min=float(detection["aspect_ratio_min"]),
            aspect_ratio_max=float(detection["aspect_ratio_max"]),
            use_length=bool(detection["use_length"]),
            orientation_mode=str(detection["orientation_mode"]),
            use_orientation=bool(detection["use_orientation"]),
            angle_tolerance_deg=float(detection["angle_tolerance_deg"]),
            length_min_cm=float(detection["length_min_cm"]),
            length_max_cm=float(detection["length_max_cm"]),
            red_hue_low1=int(detection["red_hue_low1"]),
            red_hue_high1=int(detection["red_hue_high1"]),
            red_hue_low2=int(detection["red_hue_low2"]),
            red_hue_high2=int(detection["red_hue_high2"]),
            red_sat_min=int(detection["red_sat_min"]),
            red_val_min=int(detection["red_val_min"]),
        ),
        tracker=TrackerConfig(
            match_distance_px=float(tracker["match_distance_px"]),
            min_persist_frames=int(tracker["min_persist_frames"]),
            cooldown_frames=int(tracker["cooldown_frames"]),
        ),
        freeze_mask=FreezeMaskConfig(
            enabled=bool(freeze_mask["enabled"]),
            dilate_ksize=int(freeze_mask["dilate_ksize"]),
        ),
    )


def save_config(path: Path, cfg: AppConfig) -> None:
    data = {
        "camera": {
            "index": cfg.camera.index,
            "width": cfg.camera.width,
            "height": cfg.camera.height,
            "fps": cfg.camera.fps,
            "video_file": cfg.camera.video_file,
        },
        "roi": {
            "mode": cfg.roi.mode,
            "warp_size": cfg.roi.warp_size,
            "aruco_dictionary": cfg.roi.aruco_dictionary,
            "aruco_ids": cfg.roi.aruco_ids,
            "roi_cache_path": cfg.roi.roi_cache_path,
        },
        "preview": {
            "show_raw": cfg.preview.show_raw,
            "show_warp": cfg.preview.show_warp,
            "show_fps": cfg.preview.show_fps,
            "show_diff": cfg.preview.show_diff,
            "show_mask": cfg.preview.show_mask,
            "show_overlay": cfg.preview.show_overlay,
            "window_size": cfg.preview.window_size,
            "mosaic": cfg.preview.mosaic,
            "show_controls": cfg.preview.show_controls,
            "simple_controls": cfg.preview.simple_controls,
        },
        "baseline": {
            "path": cfg.baseline.path,
            "capture_frames": cfg.baseline.capture_frames,
            "capture_delay_ms": cfg.baseline.capture_delay_ms,
        },
        "detection": {
            "mode": cfg.detection.mode,
            "preview_all": cfg.detection.preview_all,
            "preview_only": cfg.detection.preview_only,
            "blur_ksize": cfg.detection.blur_ksize,
            "diff_threshold": cfg.detection.diff_threshold,
            "morph_open_ksize": cfg.detection.morph_open_ksize,
            "morph_close_ksize": cfg.detection.morph_close_ksize,
            "use_clahe": cfg.detection.use_clahe,
            "clahe_clip": cfg.detection.clahe_clip,
            "clahe_grid": cfg.detection.clahe_grid,
            "min_area": cfg.detection.min_area,
            "max_area": cfg.detection.max_area,
            "use_aspect": cfg.detection.use_aspect,
            "aspect_ratio_min": cfg.detection.aspect_ratio_min,
            "aspect_ratio_max": cfg.detection.aspect_ratio_max,
            "use_length": cfg.detection.use_length,
            "orientation_mode": cfg.detection.orientation_mode,
            "use_orientation": cfg.detection.use_orientation,
            "angle_tolerance_deg": cfg.detection.angle_tolerance_deg,
            "length_min_cm": cfg.detection.length_min_cm,
            "length_max_cm": cfg.detection.length_max_cm,
            "red_hue_low1": cfg.detection.red_hue_low1,
            "red_hue_high1": cfg.detection.red_hue_high1,
            "red_hue_low2": cfg.detection.red_hue_low2,
            "red_hue_high2": cfg.detection.red_hue_high2,
            "red_sat_min": cfg.detection.red_sat_min,
            "red_val_min": cfg.detection.red_val_min,
        },
        "freeze_mask": {
            "enabled": cfg.freeze_mask.enabled,
            "dilate_ksize": cfg.freeze_mask.dilate_ksize,
        },
        "tracker": {
            "match_distance_px": cfg.tracker.match_distance_px,
            "min_persist_frames": cfg.tracker.min_persist_frames,
            "cooldown_frames": cfg.tracker.cooldown_frames,
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    source = VideoSource(cfg.camera)
    aligner = ROIAligner(cfg.roi)
    viz = Visualizer(cfg.preview)
    baseline_mgr = BaselineManager(cfg.baseline)
    tracker = StrokeTracker(cfg.tracker)
    freeze_mask = None
    simple_controls = cfg.preview.simple_controls
    help_overlay = True
    preview_all = cfg.detection.preview_all
    preview_only = cfg.detection.preview_only

    window_raw = "raw"
    window_warp = "warp"
    window_diff = "diff"
    window_mask = "mask"
    window_overlay = "overlay"
    window_mosaic = "mosaic"
    window_controls = "controls"

    def on_mouse(event, x, y, _flags, _param):
        if cfg.roi.mode != "manual":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            aligner.set_manual_point((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            aligner.reset_manual_points()

    def on_mouse_mosaic(event, x, y, _flags, _param):
        if cfg.roi.mode != "manual":
            return
        size = cfg.preview.window_size
        if x >= size or y >= size:
            return
        if frame is None:
            return
        scale_x = frame.shape[1] / float(size)
        scale_y = frame.shape[0] / float(size)
        mx = int(x * scale_x)
        my = int(y * scale_y)
        if event == cv2.EVENT_LBUTTONDOWN:
            aligner.set_manual_point((mx, my))
        if event == cv2.EVENT_RBUTTONDOWN:
            aligner.reset_manual_points()

    if cfg.preview.show_raw:
        cv2.namedWindow(window_raw, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_raw, on_mouse)
    if cfg.preview.show_warp and not cfg.preview.mosaic:
        cv2.namedWindow(window_warp, cv2.WINDOW_NORMAL)
    if cfg.preview.show_diff and not cfg.preview.mosaic:
        cv2.namedWindow(window_diff, cv2.WINDOW_NORMAL)
    if cfg.preview.show_mask and not cfg.preview.mosaic:
        cv2.namedWindow(window_mask, cv2.WINDOW_NORMAL)
    if cfg.preview.show_overlay and not cfg.preview.mosaic:
        cv2.namedWindow(window_overlay, cv2.WINDOW_NORMAL)
    if cfg.preview.mosaic:
        cv2.namedWindow(window_mosaic, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_mosaic, on_mouse_mosaic)

    if not cfg.preview.mosaic:
        if cfg.preview.show_raw:
            cv2.resizeWindow(window_raw, cfg.preview.window_size, cfg.preview.window_size)
        if cfg.preview.show_warp:
            cv2.resizeWindow(window_warp, cfg.preview.window_size, cfg.preview.window_size)
        if cfg.preview.show_diff:
            cv2.resizeWindow(window_diff, cfg.preview.window_size, cfg.preview.window_size)
        if cfg.preview.show_mask:
            cv2.resizeWindow(window_mask, cfg.preview.window_size, cfg.preview.window_size)
        if cfg.preview.show_overlay:
            cv2.resizeWindow(window_overlay, cfg.preview.window_size, cfg.preview.window_size)
    else:
        cv2.resizeWindow(window_mosaic, cfg.preview.window_size * 2, cfg.preview.window_size * 2)

    def build_controls(simple: bool) -> None:
        cv2.namedWindow(window_controls, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_controls, 420, 640)
        cv2.createTrackbar("mode 0=C 1=R", window_controls, 0 if cfg.detection.mode == "contrast" else 1, 1, lambda _v: None)
        cv2.createTrackbar("thr", window_controls, cfg.detection.diff_threshold, 255, lambda _v: None)
        cv2.createTrackbar("minA", window_controls, cfg.detection.min_area, 200000, lambda _v: None)
        cv2.createTrackbar("maxA", window_controls, cfg.detection.max_area, 200000, lambda _v: None)
        cv2.createTrackbar("blur", window_controls, cfg.detection.blur_ksize, 25, lambda _v: None)
        cv2.createTrackbar("open", window_controls, cfg.detection.morph_open_ksize, 21, lambda _v: None)
        cv2.createTrackbar("close", window_controls, cfg.detection.morph_close_ksize, 21, lambda _v: None)
        if not simple:
            cv2.createTrackbar("useAsp", window_controls, 1 if cfg.detection.use_aspect else 0, 1, lambda _v: None)
            cv2.createTrackbar("aspMinX10", window_controls, int(cfg.detection.aspect_ratio_min * 10), 400, lambda _v: None)
            cv2.createTrackbar("aspMaxX10", window_controls, int(cfg.detection.aspect_ratio_max * 10), 400, lambda _v: None)
            cv2.createTrackbar("useLen", window_controls, 1 if cfg.detection.use_length else 0, 1, lambda _v: None)
            cv2.createTrackbar("lenMinX10", window_controls, int(cfg.detection.length_min_cm * 10), 400, lambda _v: None)
            cv2.createTrackbar("lenMaxX10", window_controls, int(cfg.detection.length_max_cm * 10), 400, lambda _v: None)
            cv2.createTrackbar("useOri", window_controls, 1 if cfg.detection.use_orientation else 0, 1, lambda _v: None)
            cv2.createTrackbar("angTol", window_controls, int(cfg.detection.angle_tolerance_deg), 90, lambda _v: None)
            cv2.createTrackbar("clahe", window_controls, 1 if cfg.detection.use_clahe else 0, 1, lambda _v: None)
            cv2.createTrackbar("clpX10", window_controls, int(cfg.detection.clahe_clip * 10), 50, lambda _v: None)
            cv2.createTrackbar("grid", window_controls, int(cfg.detection.clahe_grid), 16, lambda _v: None)

    if cfg.preview.show_controls:
        build_controls(simple_controls)

    paused = False
    frame_idx = 0
    while True:
        if not paused:
            frame = source.read()
            if frame is None:
                break
            viz.update_fps()
        if frame is None:
            break

        if cfg.preview.show_controls:
            cfg.detection.mode = "contrast" if cv2.getTrackbarPos("mode 0=C 1=R", window_controls) == 0 else "red"
            cfg.detection.diff_threshold = cv2.getTrackbarPos("thr", window_controls)
            cfg.detection.blur_ksize = max(1, cv2.getTrackbarPos("blur", window_controls))
            cfg.detection.morph_open_ksize = cv2.getTrackbarPos("open", window_controls)
            cfg.detection.morph_close_ksize = cv2.getTrackbarPos("close", window_controls)
            cfg.detection.min_area = cv2.getTrackbarPos("minA", window_controls)
            cfg.detection.max_area = cv2.getTrackbarPos("maxA", window_controls)
            if not simple_controls:
                cfg.detection.use_aspect = cv2.getTrackbarPos("useAsp", window_controls) == 1
                cfg.detection.aspect_ratio_min = cv2.getTrackbarPos("aspMinX10", window_controls) / 10.0
                cfg.detection.aspect_ratio_max = cv2.getTrackbarPos("aspMaxX10", window_controls) / 10.0
                cfg.detection.use_length = cv2.getTrackbarPos("useLen", window_controls) == 1
                cfg.detection.length_min_cm = cv2.getTrackbarPos("lenMinX10", window_controls) / 10.0
                cfg.detection.length_max_cm = cv2.getTrackbarPos("lenMaxX10", window_controls) / 10.0
                cfg.detection.use_orientation = cv2.getTrackbarPos("useOri", window_controls) == 1
                cfg.detection.angle_tolerance_deg = cv2.getTrackbarPos("angTol", window_controls)
                cfg.detection.use_clahe = cv2.getTrackbarPos("clahe", window_controls) == 1
                cfg.detection.clahe_clip = cv2.getTrackbarPos("clpX10", window_controls) / 10.0
                cfg.detection.clahe_grid = max(2, cv2.getTrackbarPos("grid", window_controls))

        raw_vis = frame.copy()
        if cfg.roi.mode == "manual" and len(aligner.manual_points) > 0:
            for pt in aligner.manual_points:
                cv2.circle(raw_vis, pt, 6, (0, 255, 255), -1)
        viz.draw_fps(raw_vis)

        warped, _H = aligner.get_warp(frame)

        if cfg.preview.show_raw:
            cv2.imshow(window_raw, raw_vis)
        if cfg.preview.show_warp and not cfg.preview.mosaic:
            if warped is None:
                blank = np.zeros((cfg.roi.warp_size, cfg.roi.warp_size, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for ROI...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.imshow(window_warp, blank)
            else:
                warp_vis = warped.copy()
                if baseline_mgr.baseline is not None:
                    cv2.putText(
                        warp_vis,
                        "Baseline loaded",
                        (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_warp, warp_vis)

        diff_vis = None
        mask_vis = None
        overlay_vis = None
        blobs: List[Blob] = []
        if warped is not None and baseline_mgr.baseline is not None:
            if freeze_mask is None or freeze_mask.shape[:2] != warped.shape[:2]:
                freeze_mask = np.zeros(warped.shape[:2], dtype=np.uint8)

            k = max(1, cfg.detection.blur_ksize)
            if k % 2 == 0:
                k += 1

            if cfg.detection.mode == "red":
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
                hsv_base = cv2.cvtColor(baseline_mgr.baseline, cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(
                    hsv,
                    (cfg.detection.red_hue_low1, cfg.detection.red_sat_min, cfg.detection.red_val_min),
                    (cfg.detection.red_hue_high1, 255, 255),
                )
                mask2 = cv2.inRange(
                    hsv,
                    (cfg.detection.red_hue_low2, cfg.detection.red_sat_min, cfg.detection.red_val_min),
                    (cfg.detection.red_hue_high2, 255, 255),
                )
                mask_red = cv2.bitwise_or(mask1, mask2)
                base1 = cv2.inRange(
                    hsv_base,
                    (cfg.detection.red_hue_low1, cfg.detection.red_sat_min, cfg.detection.red_val_min),
                    (cfg.detection.red_hue_high1, 255, 255),
                )
                base2 = cv2.inRange(
                    hsv_base,
                    (cfg.detection.red_hue_low2, cfg.detection.red_sat_min, cfg.detection.red_val_min),
                    (cfg.detection.red_hue_high2, 255, 255),
                )
                base_red = cv2.bitwise_or(base1, base2)
                mask = cv2.bitwise_and(mask_red, cv2.bitwise_not(base_red))
                diff_vis = mask.copy()
            else:
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                base_gray = cv2.cvtColor(baseline_mgr.baseline, cv2.COLOR_BGR2GRAY)
                if k > 1:
                    gray = cv2.GaussianBlur(gray, (k, k), 0)
                    base_gray = cv2.GaussianBlur(base_gray, (k, k), 0)
                if cfg.detection.use_clahe:
                    clahe = cv2.createCLAHE(clipLimit=cfg.detection.clahe_clip, tileGridSize=(cfg.detection.clahe_grid, cfg.detection.clahe_grid))
                    gray = clahe.apply(gray)
                    base_gray = clahe.apply(base_gray)
                diff = cv2.subtract(base_gray, gray)
                _, mask = cv2.threshold(diff, cfg.detection.diff_threshold, 255, cv2.THRESH_BINARY)
                diff_vis = diff

            if cfg.detection.morph_open_ksize > 1:
                ok = cfg.detection.morph_open_ksize
                if ok % 2 == 0:
                    ok += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if cfg.detection.morph_close_ksize > 1:
                ck = cfg.detection.morph_close_ksize
                if ck % 2 == 0:
                    ck += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            if cfg.freeze_mask.enabled and freeze_mask is not None:
                mask[freeze_mask > 0] = 0

            mask_vis = mask
            overlay_vis = warped.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                if w == 0:
                    continue
                if not preview_all:
                    if area < cfg.detection.min_area or area > cfg.detection.max_area:
                        continue
                if cfg.detection.use_aspect and not preview_all:
                    aspect = h / float(w)
                    if aspect < cfg.detection.aspect_ratio_min or aspect > cfg.detection.aspect_ratio_max:
                        continue
                length_px = float(max(w, h))
                px_per_cm = cfg.roi.warp_size / 40.0
                length_cm = length_px / px_per_cm
                if cfg.detection.use_length and not preview_all:
                    if length_cm < cfg.detection.length_min_cm or length_cm > cfg.detection.length_max_cm:
                        continue
                angle_deg = 90.0
                if len(cnt) >= 5:
                    pts = cnt.reshape(-1, 2).astype(np.float32)
                    _mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                    vx, vy = eigenvectors[0]
                    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
                if cfg.detection.use_orientation and cfg.detection.orientation_mode == "vertical" and not preview_all:
                    angle_abs = abs((angle_deg + 90.0) % 180.0 - 90.0)
                    if angle_abs > cfg.detection.angle_tolerance_deg:
                        continue
                cx = x + w / 2.0
                cy = y + h / 2.0
                if not preview_all:
                    blobs.append(
                        Blob(
                            contour=cnt,
                            bbox=(x, y, w, h),
                            area=area,
                            centroid=(cx, cy),
                            angle_deg=angle_deg,
                            length_px=length_px,
                        )
                    )
                    cv2.rectangle(overlay_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    cv2.rectangle(overlay_vis, (x, y), (x + w, y + h), (0, 255, 255), 1)

        confirmed = []
        if not preview_all and not preview_only:
            confirmed = tracker.update(blobs, frame_idx=frame_idx)
            frame_idx += 1
            for tr in confirmed:
                print(
                    f"EVENT stroke_id={tr.track_id} centroid=({tr.centroid[0]:.1f},{tr.centroid[1]:.1f}) "
                    f"bbox={tr.bbox} angle={tr.angle_deg:.1f} length_px={tr.length_px:.1f}"
                )
                if cfg.freeze_mask.enabled and freeze_mask is not None:
                    x, y, w, h = tr.bbox
                    cv2.rectangle(freeze_mask, (x, y), (x + w, y + h), 255, -1)
                    dk = max(1, cfg.freeze_mask.dilate_ksize)
                    if dk % 2 == 0:
                        dk += 1
                    if dk > 1:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
                        freeze_mask = cv2.dilate(freeze_mask, kernel)

        if overlay_vis is not None:
            cv2.putText(
                overlay_vis,
                f"Count: {tracker.count} | Mode: {cfg.detection.mode} | PreviewAll: {'on' if preview_all else 'off'} | PreviewOnly: {'on' if preview_only else 'off'}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            for tr in tracker.tracks:
                if not tr.confirmed:
                    continue
                x, y, w, h = tr.bbox
                cv2.rectangle(overlay_vis, (x, y), (x + w, y + h), (255, 255, 0), 2)
            y0 = 48
            step = 18
            info_lines = [
                f"thr={cfg.detection.diff_threshold} area={cfg.detection.min_area}-{cfg.detection.max_area}",
                f"blur={cfg.detection.blur_ksize} open={cfg.detection.morph_open_ksize} close={cfg.detection.morph_close_ksize}",
                f"aspect={'on' if cfg.detection.use_aspect else 'off'} len={'on' if cfg.detection.use_length else 'off'} orient={'on' if cfg.detection.use_orientation else 'off'}",
                f"len_cm={cfg.detection.length_min_cm:.1f}-{cfg.detection.length_max_cm:.1f} angle_tol={cfg.detection.angle_tolerance_deg:.0f}",
            ]
            for line in info_lines:
                cv2.putText(
                    overlay_vis,
                    line,
                    (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )
                y0 += step
            if help_overlay:
                help_lines = [
                    "Keys: b=baseline, c=clear pts, r=reset count, f=clear mask",
                    "e=expert controls, h=toggle help, w=save config, a=preview all, t=preview only",
                    "thr=diff threshold, minA/maxA=blob area",
                    "blur/open/close=denoise controls",
                    "asp/len/orient filters shown when expert on",
                ]
                for line in help_lines:
                    cv2.putText(
                        overlay_vis,
                        line,
                        (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y0 += step

        if cfg.preview.show_diff and not cfg.preview.mosaic:
            if diff_vis is None:
                blank = np.zeros((cfg.roi.warp_size, cfg.roi.warp_size), dtype=np.uint8)
                cv2.imshow(window_diff, blank)
            else:
                cv2.imshow(window_diff, diff_vis)
        if cfg.preview.show_mask and not cfg.preview.mosaic:
            if mask_vis is None:
                blank = np.zeros((cfg.roi.warp_size, cfg.roi.warp_size), dtype=np.uint8)
                cv2.imshow(window_mask, blank)
            else:
                cv2.imshow(window_mask, mask_vis)
        if cfg.preview.show_overlay and not cfg.preview.mosaic:
            if overlay_vis is None:
                blank = np.zeros((cfg.roi.warp_size, cfg.roi.warp_size, 3), dtype=np.uint8)
                cv2.imshow(window_overlay, blank)
            else:
                cv2.imshow(window_overlay, overlay_vis)

        if cfg.preview.mosaic:
            size = cfg.preview.window_size
            def _prep(img, is_color=True):
                if img is None:
                    if is_color:
                        img = np.zeros((cfg.roi.warp_size, cfg.roi.warp_size, 3), dtype=np.uint8)
                    else:
                        img = np.zeros((cfg.roi.warp_size, cfg.roi.warp_size), dtype=np.uint8)
                if not is_color and img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return cv2.resize(img, (size, size))

            a = _prep(raw_vis, True)
            b = _prep(warped, True)
            c = _prep(diff_vis, False)
            if cfg.preview.show_overlay:
                d = _prep(overlay_vis, True)
            else:
                d = _prep(mask_vis, False)
            top = cv2.hconcat([a, b])
            bottom = cv2.hconcat([c, d])
            mosaic = cv2.vconcat([top, bottom])
            cv2.imshow(window_mosaic, mosaic)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("c"):
            aligner.reset_manual_points()
        if key == ord("b"):
            if warped is not None:
                def _get_warped():
                    frm = source.read()
                    if frm is None:
                        return None
                    w, _ = aligner.get_warp(frm)
                    return w
                ok = baseline_mgr.capture(_get_warped)
                if ok:
                    print("Baseline captured and saved.")
                    if freeze_mask is not None:
                        freeze_mask[:] = 0
                else:
                    print("Baseline capture failed.")
        if key == ord("r"):
            tracker.reset()
        if key == ord("f"):
            if freeze_mask is not None:
                freeze_mask[:] = 0
        if key == ord("w"):
            save_config(config_path, cfg)
            print("Config saved.")
        if key == ord("h"):
            help_overlay = not help_overlay
        if key == ord("e"):
            if cfg.preview.show_controls:
                simple_controls = not simple_controls
                cfg.preview.simple_controls = simple_controls
                cv2.destroyWindow(window_controls)
                build_controls(simple_controls)
        if key == ord("a"):
            preview_all = not preview_all
            if preview_all:
                preview_only = False
                cfg.detection.preview_only = False
            cfg.detection.preview_all = preview_all
        if key == ord("t"):
            preview_only = not preview_only
            if preview_only:
                preview_all = False
                cfg.detection.preview_all = False
            cfg.detection.preview_only = preview_only

    source.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
