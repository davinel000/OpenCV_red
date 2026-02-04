# OpenCV Stroke Counter (Milestone 1)

This project will detect and count new pen strokes on a 40x40 cm sheet under a fixed camera.  
Milestone 1 implements camera capture, ROI alignment (manual or ArUco), and preview windows.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --config config.yaml
```

## Controls

- `p` pause/unpause
- `q` or `Esc` quit
- `c` clear manual ROI points
- `b` capture baseline (Milestone 2)
- `r` reset tracker/count (Milestone 4)

## ROI Alignment

### Manual mode

- Set `roi.mode: "manual"` in `config.yaml`.
- Click four points in the raw window in this order:
  1. Top-left
  2. Top-right
  3. Bottom-right
  4. Bottom-left
- The points are cached to `roi_points.json`.
- Right-click clears current points.

### ArUco mode (optional)

- Set `roi.mode: "aruco"`.
- Place four ArUco markers on the sheet corners matching `roi.aruco_ids`.
- The IDs are interpreted in order: top-left, top-right, bottom-right, bottom-left.
- Requires `opencv-contrib-python` (already in `requirements.txt`).

## Baseline (Milestone 2)

- Press `b` to capture a baseline from the warped ROI.
- Baseline is averaged over `baseline.capture_frames` and saved to `baseline.path`.
- If `baseline.path` exists at startup, it is loaded automatically.

## Detection Preview (Milestone 3)

- Diff and mask windows are enabled by default.
- Use `detection.diff_threshold`, `detection.min_area`, and `detection.aspect_ratio_*` to tune sensitivity.
- Use `preview.window_size` to control initial window size.
- Use `preview.mosaic: true` for a single 2x2 view.

## Tracking (Milestone 4)

- New strokes trigger events after persisting `tracker.min_persist_frames`.
- Orientation filtering is controlled by `detection.orientation_mode`.

## Next Milestone

Milestone 3 will add contrast-mode diff, mask, blob detection, and overlay.
