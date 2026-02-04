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


@dataclass
class BaselineConfig:
    path: str
    capture_frames: int
    capture_delay_ms: int


@dataclass
class DetectionConfig:
    blur_ksize: int
    diff_threshold: int
    min_area: int
    max_area: int
    aspect_ratio_min: float
    aspect_ratio_max: float
    orientation_mode: str
    angle_tolerance_deg: float
    length_min_cm: float
    length_max_cm: float


@dataclass
class TrackerConfig:
    match_distance_px: float
    min_persist_frames: int
    cooldown_frames: int


@dataclass
class AppConfig:
    camera: CameraConfig
    roi: ROIConfig
    preview: PreviewConfig
    baseline: BaselineConfig
    detection: DetectionConfig
    tracker: TrackerConfig


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
        ),
        baseline=BaselineConfig(
            path=str(baseline["path"]),
            capture_frames=int(baseline["capture_frames"]),
            capture_delay_ms=int(baseline["capture_delay_ms"]),
        ),
        detection=DetectionConfig(
            blur_ksize=int(detection["blur_ksize"]),
            diff_threshold=int(detection["diff_threshold"]),
            min_area=int(detection["min_area"]),
            max_area=int(detection["max_area"]),
            aspect_ratio_min=float(detection["aspect_ratio_min"]),
            aspect_ratio_max=float(detection["aspect_ratio_max"]),
            orientation_mode=str(detection["orientation_mode"]),
            angle_tolerance_deg=float(detection["angle_tolerance_deg"]),
            length_min_cm=float(detection["length_min_cm"]),
            length_max_cm=float(detection["length_max_cm"]),
        ),
        tracker=TrackerConfig(
            match_distance_px=float(tracker["match_distance_px"]),
            min_persist_frames=int(tracker["min_persist_frames"]),
            cooldown_frames=int(tracker["cooldown_frames"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    source = VideoSource(cfg.camera)
    aligner = ROIAligner(cfg.roi)
    viz = Visualizer(cfg.preview)
    baseline_mgr = BaselineManager(cfg.baseline)
    tracker = StrokeTracker(cfg.tracker)

    window_raw = "raw"
    window_warp = "warp"
    window_diff = "diff"
    window_mask = "mask"
    window_overlay = "overlay"
    window_mosaic = "mosaic"

    def on_mouse(event, x, y, _flags, _param):
        if cfg.roi.mode != "manual":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            aligner.set_manual_point((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            aligner.reset_manual_points()

    if cfg.preview.show_raw and not cfg.preview.mosaic:
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
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            base_gray = cv2.cvtColor(baseline_mgr.baseline, cv2.COLOR_BGR2GRAY)
            k = max(1, cfg.detection.blur_ksize)
            if k % 2 == 0:
                k += 1
            if k > 1:
                gray = cv2.GaussianBlur(gray, (k, k), 0)
                base_gray = cv2.GaussianBlur(base_gray, (k, k), 0)
            diff = cv2.subtract(base_gray, gray)
            _, mask = cv2.threshold(diff, cfg.detection.diff_threshold, 255, cv2.THRESH_BINARY)
            diff_vis = diff
            mask_vis = mask

            overlay_vis = warped.copy()
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < cfg.detection.min_area or area > cfg.detection.max_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w == 0:
                    continue
                aspect = h / float(w)
                if aspect < cfg.detection.aspect_ratio_min or aspect > cfg.detection.aspect_ratio_max:
                    continue
                length_px = float(max(w, h))
                px_per_cm = cfg.roi.warp_size / 40.0
                length_cm = length_px / px_per_cm
                if length_cm < cfg.detection.length_min_cm or length_cm > cfg.detection.length_max_cm:
                    continue
                angle_deg = 90.0
                if len(cnt) >= 5:
                    pts = cnt.reshape(-1, 2).astype(np.float32)
                    mean, eigenvectors = cv2.PCACompute(pts, mean=None)
                    vx, vy = eigenvectors[0]
                    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
                if cfg.detection.orientation_mode == "vertical":
                    angle_abs = abs((angle_deg + 90.0) % 180.0 - 90.0)
                    if angle_abs > cfg.detection.angle_tolerance_deg:
                        continue
                cx = x + w / 2.0
                cy = y + h / 2.0
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

        confirmed = tracker.update(blobs, frame_idx=frame_idx)
        frame_idx += 1
        for tr in confirmed:
            print(
                f"EVENT stroke_id={tr.track_id} centroid=({tr.centroid[0]:.1f},{tr.centroid[1]:.1f}) "
                f"bbox={tr.bbox} angle={tr.angle_deg:.1f} length_px={tr.length_px:.1f}"
            )

        if overlay_vis is not None:
            cv2.putText(
                overlay_vis,
                f"Count: {tracker.count}",
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
                else:
                    print("Baseline capture failed.")
        if key == ord("r"):
            tracker.reset()

    source.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
