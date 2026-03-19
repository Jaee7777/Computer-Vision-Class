# Project 4: Calilbration and Augmented Reality
- Jaee Oh

## Requirements
- OpenCV 4.x with `opencv_contrib` (required for SURF)
```
# Ubuntu/Debian
sudo apt install libopencv-contrib-dev

# macOS with Homebrew
brew install opencv
```
- A printed 10x7 checkerboard (9x6 inner corners)
- Measure one square on your board and update `square_size_mm` in `vidDisplay.cpp`

## Build
```bash
mkdir build && cd build
cmake ..
make
```

## Run
```bash
./vidDisplay
```

## Calibration (do this first)
1. Hold the checkerboard in front of the camera
2. Press **`s`** to save the current frame (need at least 5, aim for 10–15)
3. Move the board to different angles and distances between saves
4. Press **`c`** to run calibration — results saved to `calibration.yml`
5. On future runs, `calibration.yml` is loaded automatically

## Controls
| Key | Action |
|-----|--------|
| `s` | Save calibration frame (board must be visible) |
| `c` | Run calibration (requires 5+ saved frames) |
| `h` | Toggle Harris corner detection |
| `u` | Toggle SURF feature detection |
| `q` | Quit |

## Features
- **Checkerboard detection** — detects and highlights inner corners in real time
- **Camera calibration** — computes camera matrix and distortion coefficients
- **AR axes** — draws X/Y/Z axes anchored to the board once calibrated
- **Virtual cube** — renders a colored 3D cube sitting on the board
- **Harris corners** — overlays detected corners across the full frame
- **SURF features** — overlays scale- and rotation-invariant keypoints

## Output Files
| File | Description |
|------|-------------|
| `calibration.yml` | Camera matrix and distortion coefficients |
| `calib_0.jpg`, `calib_1.jpg`, ... | Saved calibration frames with corners drawn |